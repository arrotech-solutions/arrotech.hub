"""
WhatsApp Workflow Trigger Service.
Fires workflows based on WhatsApp events (new message, new contact, keyword match).
Includes real estate specific triggers for property inquiries, maintenance, and payments.
"""

import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import tenant_session
from ..models import (
    Workflow, WorkflowStatus, WorkflowTriggerType,
    WhatsAppContact, WhatsAppMessage, WhatsAppMessageDirection
)
from ..services.workflow_builder_service import WorkflowBuilderService
from ..observability.tracer import trace_span

logger = logging.getLogger(__name__)

_STORAGE_CONFIG_KEYS = (
    "storage_provider",
    "storage_spreadsheet_id",
    "storage_orders_sheet_name",
    "storage_customers_sheet_name",
    "storage_transactions_sheet_name",
    "storage_reservations_sheet_name",
    "storage_airtable_base_id",
    "storage_airtable_orders_table",
    "storage_airtable_customers_table",
    "storage_airtable_transactions_table",
    "storage_airtable_reservations_table",
)

# Agent settings stored as flat workflow.variables — merged into runtime config
# so older workflows pick up new keys without recreating from the Library template.
_AGENT_CONFIG_KEYS = _STORAGE_CONFIG_KEYS + (
    "kb_id",
    "business_name",
    "business_phone",
    "business_email",
    "business_telegram_chat_id",
    "order_type",
    "currency",
    "delivery_methods",
    "reservations_enabled",
    "system_prompt",
    "auto_escalation_enabled",
    "supported_languages",
    "human_handoff_ttl_hours",
    "agent_mode",
)


def _merge_workflow_storage_into_config(
    wf_config: Dict[str, Any],
    workflow_variables: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Backfill agent + storage settings from top-level workflow variables."""
    from .whatsapp_ordering_helpers import normalize_agent_storage_settings

    merged = dict(wf_config or {})
    variables = workflow_variables or {}
    nested_config = variables.get("config") if isinstance(variables.get("config"), dict) else {}

    for key in _AGENT_CONFIG_KEYS:
        # Legacy nested config fills gaps only; top-level workflow.variables always wins.
        if merged.get(key) in (None, "", []):
            if nested_config.get(key) not in (None, "", []):
                merged[key] = nested_config[key]
        if variables.get(key) not in (None, "", []):
            merged[key] = variables[key]

    return normalize_agent_storage_settings(merged)


# Real estate keyword groups for trigger matching
RE_KEYWORDS = {
    "property_inquiry": [
        "rent", "kodi", "buy", "nunua", "apartment", "flat", "house", "nyumba",
        "bedsitter", "bedsitta", "plot", "shamba", "land", "2br", "3br", "1br",
        "bedroom", "studio", "office", "shop", "duka", "godown", "commercial",
        "for sale", "for rent", "available", "vacancy", "vacant"
    ],
    "viewing_request": [
        "view", "visit", "see the", "tazama", "viewing", "angalia", "come see",
        "can i see", "schedule", "book viewing", "show me"
    ],
    "maintenance_request": [
        "broken", "leak", "repair", "fix", "maintenance", "plumbing", "pipe",
        "electrical", "water", "maji", "stima", "haribika", "vunja", "blocked",
        "not working", "damaged", "crack", "flood", "toilet", "sink"
    ],
    "payment_confirmation": [
        "confirmed", "paid", "mpesa", "m-pesa", "transaction", "receipt",
        "nimelipa", "payment sent", "sent payment"
    ],
    "lease_inquiry": [
        "lease", "contract", "renew", "renewal", "extend", "move out",
        "vacate", "notice", "leaving"
    ],
}


class WhatsAppWorkflowTrigger:
    """Service to trigger workflows based on WhatsApp events."""

    @classmethod
    async def has_active_conversational_agent(
        cls,
        user_id: uuid.UUID,
        db: AsyncSession,
    ) -> bool:
        """True if user has an active WhatsApp workflow with a conversational_agent step."""
        from ..models import WorkflowStep

        result = await db.execute(
            select(Workflow).where(
                and_(
                    Workflow.user_id == user_id,
                    Workflow.status.in_([WorkflowStatus.ACTIVE, WorkflowStatus.ACTIVE.value]),
                    Workflow.trigger_type.in_([WorkflowTriggerType.EVENT, WorkflowTriggerType.EVENT.value, WorkflowTriggerType.WEBHOOK, WorkflowTriggerType.WEBHOOK.value]),
                )
            )
        )
        workflows = result.scalars().all()
        for workflow in workflows:
            trigger_config = workflow.trigger_config or {}
            if trigger_config.get("event_type") != "whatsapp_message_received":
                continue
            step_result = await db.execute(
                select(WorkflowStep).where(
                    and_(
                        WorkflowStep.workflow_id == workflow.id,
                        WorkflowStep.tool_name == "conversational_agent",
                    )
                )
            )
            if step_result.scalar_one_or_none():
                return True
        return False
    
    TRIGGER_EVENTS = [
        "whatsapp_message_received",
        "whatsapp_new_contact",
        "whatsapp_keyword_detected",
        # Real estate specific triggers
        "whatsapp_property_inquiry",
        "whatsapp_viewing_request",
        "whatsapp_maintenance_request",
        "whatsapp_payment_confirmation",
        "whatsapp_lease_inquiry",
    ]
    
    @classmethod
    def _detect_real_estate_intent(cls, message_content: str) -> Optional[str]:
        """Detect real estate intent from message content using whole-word
        matching, so a keyword like 'lease' does NOT match inside 'please'
        (which previously mis-tagged food/cart messages as 'lease_inquiry')."""
        if not message_content:
            return None

        content_lower = message_content.lower()

        for intent, keywords in RE_KEYWORDS.items():
            for keyword in keywords:
                kw = keyword.lower().strip()
                if not kw:
                    continue
                # (?<!\w) ... (?!\w) = phrase/whole-word boundaries; avoids false
                # positives such as 'lease' in 'please' or 'notice' in 'notices'.
                if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", content_lower):
                    return intent

        return None
    
    @classmethod
    async def on_message_received(
        cls,
        user_id: uuid.UUID,
        contact: WhatsAppContact,
        message: WhatsAppMessage
    ):
        """
        Called when a new WhatsApp message is received.
        Checks for matching workflows and triggers them.
        Includes real estate intent detection.
        """
        async with tenant_session(user_id) as db:
            try:
                # Detect real estate intent from message
                re_intent = cls._detect_real_estate_intent(message.content)
                re_event_type = f"whatsapp_{re_intent}" if re_intent else None
                
                # ── CCM: Persist incoming message to conversation session ──
                session_key = ""
                try:
                    from .conversation_context_manager import context_manager

                    session = await context_manager.get_or_create_session(
                        platform="whatsapp",
                        owner_user_id=str(user_id),
                        sender_id=contact.phone_number,
                        metadata={
                            "contact_name": contact.name or contact.profile_name or "",
                            "contact_phone": contact.phone_number,
                        }
                    )
                    session_key = session.session_key

                    # Add incoming message to history
                    await context_manager.add_message(
                        session, "user", message.content or ""
                    )
                except Exception as ccm_err:
                    logger.warning(f"[WA_TRIGGER] CCM session init failed (non-blocking): {ccm_err}")

                # Find workflows with WhatsApp triggers
                result = await db.execute(
                    select(Workflow)
                    .options(selectinload(Workflow.steps))
                    .where(
                        and_(
                            Workflow.user_id == user_id,
                            Workflow.status.in_([WorkflowStatus.ACTIVE, WorkflowStatus.ACTIVE.value]),
                            Workflow.trigger_type.in_([WorkflowTriggerType.EVENT, WorkflowTriggerType.EVENT.value, WorkflowTriggerType.WEBHOOK, WorkflowTriggerType.WEBHOOK.value]),
                        )
                    )
                )
                workflows = result.scalars().all()

                # Handoff TTL from first active ordering workflow config (matches agent)
                if session_key:
                    try:
                        from ..config import settings

                        ttl_hours = int(
                            getattr(settings, "AGENT_HUMAN_HANDOFF_TTL_HOURS", 24) or 0
                        )
                        for wf in workflows:
                            tc = wf.trigger_config or {}
                            if tc.get("event_type") != "whatsapp_message_received":
                                continue
                            cfg = (wf.variables or {}).get("config") or {}
                            if cfg.get("human_handoff_ttl_hours") is not None:
                                try:
                                    ttl_hours = int(cfg["human_handoff_ttl_hours"])
                                except (TypeError, ValueError):
                                    pass
                                break
                        if ttl_hours > 0:
                            expired = await context_manager.maybe_expire_human_handoff(
                                session_key, ttl_hours * 3600
                            )
                            if expired:
                                logger.info(
                                    "[WA_TRIGGER] Handoff TTL expired for %s — AI resumed",
                                    contact.phone_number,
                                )
                    except Exception as handoff_err:
                        logger.warning(
                            f"[WA_TRIGGER] Handoff TTL check failed (continuing): {handoff_err}"
                        )

                def _workflow_has_conversational_agent(wf: Workflow) -> bool:
                    return any(
                        s.tool_name == "conversational_agent" for s in (wf.steps or [])
                    )

                matched: List[tuple] = []
                for workflow in workflows:
                    trigger_config = workflow.trigger_config or {}
                    event_type = trigger_config.get("event_type", "")
                    should_trigger = False

                    if event_type == "whatsapp_message_received":
                        should_trigger = True
                    elif event_type == "whatsapp_new_contact":
                        if contact.message_count == 1:
                            should_trigger = True
                    elif event_type == "whatsapp_keyword_detected":
                        keywords = trigger_config.get("keywords", [])
                        content = (message.content or "").lower()
                        for keyword in keywords:
                            if keyword.lower() in content:
                                should_trigger = True
                                break
                    elif re_event_type and event_type == re_event_type:
                        should_trigger = True

                    if should_trigger:
                        matched.append((workflow, event_type))

                # Run at most one whatsapp_message_received ordering workflow per message
                wa_general = [
                    (w, et) for w, et in matched if et == "whatsapp_message_received"
                ]
                others = [
                    (w, et) for w, et in matched if et != "whatsapp_message_received"
                ]
                to_execute: List[tuple] = list(others)
                if wa_general:
                    preferred = next(
                        (w for w, _ in wa_general if _workflow_has_conversational_agent(w)),
                        wa_general[0][0],
                    )
                    to_execute.append((preferred, "whatsapp_message_received"))
                    if len(wa_general) > 1:
                        logger.warning(
                            "[WA_TRIGGER] %s whatsapp_message_received workflows matched; "
                            "running only '%s'",
                            len(wa_general),
                            preferred.name,
                        )

                if not to_execute:
                    from ..observability.logger import log_event
                    import logging as _logging
                    log_event(
                        level=_logging.INFO,
                        event_type="NO_WORKFLOW_MATCHED",
                        message=f"No active workflow matched for message from {contact.phone_number}",
                        payload={"matched_workflows_count": len(matched)}
                    )

                for workflow, event_type in to_execute:
                    # Global handoff gate — skip AI when a human agent owns the thread
                    if session_key:
                        try:
                            from ..services.conversation_context_manager import context_manager

                            if await context_manager.is_human_handoff_active(session_key):
                                from ..observability.logger import log_event
                                import logging as _logging
                                log_event(
                                    level=_logging.INFO,
                                    event_type="WORKFLOW_SKIPPED_HANDOFF",
                                    message=f"Skipping workflow '{workflow.name}' due to human handoff",
                                    workflow_id=str(workflow.id)
                                )
                                logger.info(
                                    "[WA_TRIGGER] Skipping workflow '%s' — human handoff active for %s",
                                    workflow.name,
                                    contact.phone_number,
                                )
                                continue
                        except Exception as handoff_gate_err:
                            logger.warning(
                                "[WA_TRIGGER] Handoff gate check failed (continuing): %s",
                                handoff_gate_err,
                            )

                    wf_config = dict((workflow.variables or {}).get("config", {}) or {})
                    wf_config = _merge_workflow_storage_into_config(
                        wf_config, workflow.variables
                    )
                    wf_config.setdefault("customer_phone", contact.phone_number)
                    wf_config.setdefault(
                        "customer_name",
                        contact.name or contact.profile_name or "Customer",
                    )
                    wf_config.setdefault("platform", "whatsapp")

                    agent_mode = str(wf_config.get("agent_mode", "") or "").strip().lower()
                    if agent_mode == "support":
                        try:
                            from ..services.whatsapp_compliance_service import WhatsAppComplianceService

                            if not await WhatsAppComplianceService.is_business_open(user_id, db):
                                logger.info(
                                    "[WA_TRIGGER] Outside business hours — skipping support workflow '%s'",
                                    workflow.name,
                                )
                                continue
                        except Exception as bh_err:
                            logger.warning("[WA_TRIGGER] Business hours check failed: %s", bh_err)

                    input_vars = {
                        "whatsapp_contact_id": str(contact.id) if contact.id else None,
                        "whatsapp_contact_phone": contact.phone_number,
                        "whatsapp_contact_name": contact.name or contact.profile_name or "Customer",
                        "whatsapp_message_id": str(message.id) if message.id else None,
                        "whatsapp_message_content": message.content or "",
                        "whatsapp_message_type": message.message_type or "",
                        "whatsapp_is_location": message.message_type == "location",
                        # Meta's actual message ID (wam.xxx) for idempotency.
                        # Distinct from whatsapp_message_id which is the internal DB UUID.
                        "meta_message_id": message.whatsapp_message_id or "",
                        "timestamp": datetime.utcnow().isoformat(),
                        "session_key": session_key,
                        "platform": "whatsapp",
                        "config": wf_config,
                    }

                    # Only attach real-estate intent to real-estate workflows, so
                    # a food/retail tenant is never tagged with RE intents.
                    order_type_val = str(wf_config.get("order_type", "")).strip().lower()
                    is_real_estate_wf = order_type_val in (
                        "real_estate", "realestate", "property", "properties",
                        "rental", "rentals", "lease", "housing",
                    )
                    attach_re_intent = bool(re_intent) and (
                        is_real_estate_wf or event_type == re_event_type
                    )
                    if attach_re_intent:
                        input_vars["real_estate_intent"] = re_intent
                        input_vars["real_estate_event"] = re_event_type

                    from ..observability.logger import log_event
                    import logging as _logging
                    log_event(
                        level=_logging.INFO,
                        event_type="WORKFLOW_MATCH",
                        message=f"Firing workflow '{workflow.name}'",
                        workflow_id=str(workflow.id),
                        payload={"trigger_type": event_type, "re_intent": re_intent}
                    )
                    
                    logger.info(
                        f"[WA_TRIGGER] Firing workflow '{workflow.name}' for contact {contact.phone_number}"
                        + (f" (RE intent: {re_intent})" if attach_re_intent else "")
                    )

                    try:
                        with trace_span("execute_workflow", payload={"workflow_id": str(workflow.id), "workflow_name": workflow.name}):
                            log_event(
                                level=_logging.INFO,
                                event_type="WORKFLOW_EXECUTION_START",
                                message=f"Executing workflow '{workflow.name}'",
                                workflow_id=str(workflow.id)
                            )
                            builder = WorkflowBuilderService()
                            await builder.execute_workflow(
                                workflow_id=workflow.id,
                                user_id=user_id,
                                db=db,
                                input_data=input_vars,
                                trigger_type=event_type or "whatsapp_message_received",
                            )
                            log_event(
                                level=_logging.INFO,
                                event_type="WORKFLOW_EXECUTION_SUCCESS",
                                message=f"Successfully executed workflow '{workflow.name}'",
                                workflow_id=str(workflow.id)
                            )
                    except Exception as e:
                        log_event(
                            level=_logging.ERROR,
                            event_type="WORKFLOW_EXECUTION_ERROR",
                            status="failed",
                            message=f"Failed to execute workflow {workflow.id}: {e}",
                            workflow_id=str(workflow.id),
                            error_message=str(e)
                        )
                        logger.error(
                            f"[WA_TRIGGER] Failed to execute workflow {workflow.id}: {e}"
                        )
                            
            except Exception as e:
                logger.error(f"[WA_TRIGGER] Error checking workflows: {e}")


async def register_whatsapp_workflow_actions():
    """
    Register WhatsApp actions that can be used in workflows.
    These are added to the MCP tools registry.
    """
    # This is called at startup to register WhatsApp as workflow actions
    workflow_actions = {
        "whatsapp_send_message": {
            "description": "Send a WhatsApp message to a contact",
            "parameters": {
                "contact_id": {"type": "integer", "description": "Contact ID to send to"},
                "message": {"type": "string", "description": "Message content"}
            }
        },
        "whatsapp_send_template": {
            "description": "Send a WhatsApp template message",
            "parameters": {
                "contact_id": {"type": "integer", "description": "Contact ID to send to"},
                "template_name": {"type": "string", "description": "Template name"},
                "variables": {"type": "object", "description": "Template variables"}
            }
        },
        "whatsapp_add_tag": {
            "description": "Add a tag to a WhatsApp contact",
            "parameters": {
                "contact_id": {"type": "integer", "description": "Contact ID"},
                "tag": {"type": "string", "description": "Tag to add"}
            }
        },
        # Real estate specific actions
        "whatsapp_send_rent_reminder": {
            "description": "Send a formatted rent reminder via WhatsApp",
            "parameters": {
                "contact_id": {"type": "integer", "description": "Tenant contact ID"},
                "amount": {"type": "number", "description": "Rent amount in KES"},
                "due_date": {"type": "string", "description": "Payment due date"},
                "reminder_level": {"type": "string", "enum": ["first", "second", "final"]}
            }
        },
        "whatsapp_send_maintenance_ack": {
            "description": "Send a maintenance request acknowledgement via WhatsApp",
            "parameters": {
                "contact_id": {"type": "integer", "description": "Tenant contact ID"},
                "category": {"type": "string", "description": "Maintenance category"},
                "priority": {"type": "string", "description": "Priority level"}
            }
        },
        "whatsapp_send_viewing_slots": {
            "description": "Send available property viewing slots via WhatsApp",
            "parameters": {
                "contact_id": {"type": "integer", "description": "Prospect contact ID"},
                "property_description": {"type": "string"},
                "slots": {"type": "array", "items": {"type": "string"}}
            }
        },
    }
    
    return workflow_actions


async def execute_whatsapp_action(
    action_name: str,
    user_id: uuid.UUID,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a WhatsApp workflow action.
    Called by the workflow executor when a workflow step uses a WhatsApp action.
    """
    from ..services import WhatsAppService
    from ..config import settings
    from ..models import Connection
    
    async with tenant_session(user_id) as db:
        with trace_span(f"action_{action_name}", payload={"parameters": parameters}):
            try:
                    # Get user's WhatsApp connection config
                conn_result = await db.execute(
                    select(Connection).where(
                        and_(
                            Connection.user_id == user_id,
                            Connection.platform == "whatsapp",
                            Connection.status == "active"
                        )
                    )
                )
                connection = conn_result.scalar_one_or_none()
                wa_config = connection.config if connection and connection.config else {
                    "access_token": settings.WHATSAPP_TOKEN,
                    "phone_number_id": settings.WHATSAPP_PHONE_NUMBER_ID
                }

                if action_name == "whatsapp_send_message":
                    contact_id = parameters.get("contact_id")
                    message_content = parameters.get("message")
                
                    # Get contact
                    result = await db.execute(
                        select(WhatsAppContact).where(
                            and_(
                                WhatsAppContact.id == contact_id,
                                WhatsAppContact.user_id == user_id
                            )
                        )
                    )
                    contact = result.scalar_one_or_none()
                
                    if not contact:
                        return {"success": False, "error": "Contact not found"}
                
                    # Send message
                    wa_service = WhatsAppService()
                    result = await wa_service.send_message(
                        to_number=contact.phone_number,
                        message=message_content,
                        message_type="text",
                        config=wa_config
                    )

                    if result.get("success") and message_content:
                        try:
                            from ..services.whatsapp_inbox_service import record_outbound_message

                            await record_outbound_message(
                                db,
                                user_id=user_id,
                                phone_number=contact.phone_number,
                                content=message_content,
                                whatsapp_message_id=result.get("message_id"),
                                contact_id=contact.id,
                                is_agent=True,
                            )
                        except Exception as inbox_err:
                            logger.warning(
                                "[WA_ACTION] Failed to persist outbound to inbox: %s",
                                inbox_err,
                            )
                
                    return {"success": True, "message_id": result.get("message_id")}
                
                elif action_name == "whatsapp_add_tag":
                    contact_id = parameters.get("contact_id")
                    tag = parameters.get("tag")
                
                    result = await db.execute(
                        select(WhatsAppContact).where(
                            and_(
                                WhatsAppContact.id == contact_id,
                                WhatsAppContact.user_id == user_id
                            )
                        )
                    )
                    contact = result.scalar_one_or_none()
                
                    if not contact:
                        return {"success": False, "error": "Contact not found"}
                
                    # Add tag
                    current_tags = contact.tags or []
                    if tag not in current_tags:
                        current_tags.append(tag)
                        contact.tags = current_tags
                        await db.commit()
                
                    return {"success": True, "tags": contact.tags}
            
                elif action_name == "whatsapp_send_rent_reminder":
                    # Use real estate tools to format, then send via WhatsApp
                    from .real_estate_service import RealEstateService
                    re_service = RealEstateService()
                
                    formatted = await re_service.format_rent_reminder(
                        tenant_name=parameters.get("tenant_name", "Tenant"),
                        amount=parameters.get("amount", 0),
                        due_date=parameters.get("due_date", ""),
                        reminder_level=parameters.get("reminder_level", "first"),
                        paybill=parameters.get("paybill", ""),
                        account_number=parameters.get("account_number", ""),
                    )
                
                    if not formatted.get("success"):
                        return formatted
                
                    contact_id = parameters.get("contact_id")
                    result_contact = await db.execute(
                        select(WhatsAppContact).where(
                            and_(WhatsAppContact.id == contact_id, WhatsAppContact.user_id == user_id)
                        )
                    )
                    contact = result_contact.scalar_one_or_none()
                
                    if not contact:
                        return {"success": False, "error": "Contact not found"}
                
                    wa_service = WhatsAppService()
                    send_result = await wa_service.send_message(
                        to_number=contact.phone_number,
                        message=formatted["message"],
                        message_type="text",
                        config=wa_config
                    )
                
                    return {"success": True, "message_id": send_result.get("message_id"), "formatted_message": formatted["message"]}
            
                elif action_name == "whatsapp_send_maintenance_ack":
                    from .real_estate_service import RealEstateService
                    re_service = RealEstateService()
                
                    formatted = await re_service.format_maintenance_response(
                        tenant_name=parameters.get("tenant_name", "Tenant"),
                        category=parameters.get("category", "general"),
                        priority=parameters.get("priority", "normal"),
                    )
                
                    if not formatted.get("success"):
                        return formatted
                
                    contact_id = parameters.get("contact_id")
                    result_contact = await db.execute(
                        select(WhatsAppContact).where(
                            and_(WhatsAppContact.id == contact_id, WhatsAppContact.user_id == user_id)
                        )
                    )
                    contact = result_contact.scalar_one_or_none()
                
                    if not contact:
                        return {"success": False, "error": "Contact not found"}
                
                    wa_service = WhatsAppService()
                    send_result = await wa_service.send_message(
                        to_number=contact.phone_number,
                        message=formatted["message"],
                        message_type="text",
                        config=wa_config
                    )
                
                    return {"success": True, "message_id": send_result.get("message_id"), "ticket_id": formatted.get("ticket_id")}
            
                elif action_name == "whatsapp_send_viewing_slots":
                    from .real_estate_service import RealEstateService
                    re_service = RealEstateService()
                
                    formatted = await re_service.format_viewing_slots(
                        property_description=parameters.get("property_description", "the property"),
                        slots=parameters.get("slots"),
                        location=parameters.get("location", ""),
                        agent_name=parameters.get("agent_name", ""),
                    )
                
                    if not formatted.get("success"):
                        return formatted
                
                    contact_id = parameters.get("contact_id")
                    result_contact = await db.execute(
                        select(WhatsAppContact).where(
                            and_(WhatsAppContact.id == contact_id, WhatsAppContact.user_id == user_id)
                        )
                    )
                    contact = result_contact.scalar_one_or_none()
                
                    if not contact:
                        return {"success": False, "error": "Contact not found"}
                
                    wa_service = WhatsAppService()
                    send_result = await wa_service.send_message(
                        to_number=contact.phone_number,
                        message=formatted["message"],
                        message_type="text",
                        config=wa_config
                    )
                
                    return {"success": True, "message_id": send_result.get("message_id"), "slots": formatted.get("available_slots")}
                
                else:
                    return {"success": False, "error": f"Unknown action: {action_name}"}
                
            except Exception as e:
                logger.error(f"[WA_ACTION] Error executing {action_name}: {e}")
            return {"success": False, "error": str(e)}

