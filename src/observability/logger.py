import json
import logging
import datetime
import asyncio
from typing import Any, Dict, Optional
import sys

from .tracer import get_trace_id, get_span_id, get_parent_span_id, get_customer_id, get_phone_number_hash

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for production logs."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "trace_id": get_trace_id(),
            "span_id": get_span_id(),
            "parent_span_id": get_parent_span_id(),
            "customer_id": get_customer_id(),
            "phone_number_hash": get_phone_number_hash(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra attributes from record if they exist
        if hasattr(record, "event_type"):
            log_data["event_type"] = record.event_type
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "status"):
            log_data["status"] = record.status
        
        # Add exception info if present
        if record.exc_info:
            log_data["error_type"] = record.exc_info[0].__name__
            log_data["error_message"] = str(record.exc_info[1])
            log_data["stack_trace"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

# Global queue for async DB logging
_log_queue: Optional[asyncio.Queue] = None

def get_log_queue() -> asyncio.Queue:
    global _log_queue
    if _log_queue is None:
        # Create queue on first access, binding to the current active event loop
        _log_queue = asyncio.Queue()
    return _log_queue

async def flush_all_logs_async():
    """Flush all currently queued logs to the database immediately."""
    from ..database import get_session_maker
    from ..models import ObservabilityLog
    
    q = get_log_queue()
    if q.empty():
        return 0
        
    session_maker = get_session_maker()
    logs_to_persist = []
    
    # Try to get logs quickly (up to 100 or until queue empty)
    for _ in range(100):
        try:
            logs_to_persist.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
            
    if not logs_to_persist:
        return 0
        
    # Persist batch
    try:
        async with session_maker() as session:
            for log_data in logs_to_persist:
                db_log = ObservabilityLog(
                    level=log_data.get("level", "INFO"),
                    trace_id=log_data.get("trace_id"),
                    span_id=log_data.get("span_id"),
                    parent_span_id=log_data.get("parent_span_id"),
                    event_type=log_data.get("event_type", "GENERIC"),
                    customer_id=log_data.get("customer_id"),
                    phone_number_hash=log_data.get("phone_number_hash"),
                    agent_id=log_data.get("agent_id"),
                    workflow_id=log_data.get("workflow_id"),
                    tool_name=log_data.get("tool_name"),
                    step_name=log_data.get("step_name"),
                    status=log_data.get("status"),
                    duration_ms=log_data.get("duration_ms"),
                    retry_count=log_data.get("retry_count", 0),
                    payload=log_data.get("payload"),
                    error_type=log_data.get("error_type"),
                    error_message=log_data.get("error_message"),
                    stack_trace=log_data.get("stack_trace")
                )
                session.add(db_log)
            await session.commit()
            
        # Mark tasks as done
        for _ in range(len(logs_to_persist)):
            q.task_done()
            
        return len(logs_to_persist)
        
    except Exception as e:
        print(f"CRITICAL: Failed to persist logs to DB: {e}", file=sys.stderr)
        return 0

async def db_log_worker():
    """Background worker to persist logs from queue to database (FastAPI only)."""
    while True:
        # Avoid race condition by just peeking/sleeping, then flushing
        q = get_log_queue()
        if q.empty():
            await asyncio.sleep(0.5)
            continue
            
        await flush_all_logs_async()
        await asyncio.sleep(0.5)  # Small backoff before checking again

async def log_cleanup_job(retention_days: int = 14):
    """Periodically delete logs older than the retention period."""
    from ..database import get_session_maker
    from ..models import ObservabilityLog
    from sqlalchemy import delete
    import datetime
    
    session_maker = get_session_maker()
    
    while True:
        try:
            cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=retention_days)
            # More aggressive cleanup for high-volume logs (e.g. WhatsApp read receipts, webhooks)
            fast_cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=3)
            high_volume_events = ["WA_API_SEND", "WA_API_READ_RECEIPT", "WEBHOOK_RECEIVED", "WEBHOOK_DISPATCH", "MESSAGE_RECEIVED", "MESSAGE_SAVED"]
            
            async with session_maker() as session:
                # Delete old logs (standard retention for ERROR/CRITICAL etc)
                result = await session.execute(
                    delete(ObservabilityLog).where(ObservabilityLog.timestamp < cutoff)
                )
                
                # Delete high-volume INFO logs sooner
                result_fast = await session.execute(
                    delete(ObservabilityLog).where(
                        ObservabilityLog.timestamp < fast_cutoff,
                        ObservabilityLog.event_type.in_(high_volume_events)
                    )
                )
                
                await session.commit()
                deleted_count = result.rowcount + result_fast.rowcount
                if deleted_count > 0:
                    logging.info(f"Cleaned up {deleted_count} old logs from database.")
            
            # Run cleanup once a day
            await asyncio.sleep(86400)
            
        except Exception as e:
            logging.error(f"Error in log cleanup job: {e}")
            await asyncio.sleep(3600) # Retry in an hour if it fails

def setup_observability_logging():
    """Configure structured logging and start DB worker."""
    root_logger = logging.getLogger()
    
    # Standard output handler (JSON)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(JSONFormatter())
    
    # Clear existing handlers and add ours
    root_logger.handlers = [stdout_handler]
    root_logger.setLevel(logging.INFO)
    
    return root_logger

def log_event(
    level: int,
    event_type: str,
    message: str,
    status: str = "success",
    duration_ms: Optional[int] = None,
    payload: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """Helper to log structured events to both stdout and DB queue."""
    logger = logging.getLogger("observability")
    
    # Extra data for JSON formatter (stdout)
    extra = {
        "event_type": event_type,
        "status": status,
        "duration_ms": duration_ms,
        **kwargs
    }
    
    logger.log(level, message, extra=extra)
    
    # Add to DB queue for persistence
    db_data = {
        "level": logging.getLevelName(level),
        "trace_id": get_trace_id(),
        "span_id": get_span_id(),
        "parent_span_id": get_parent_span_id(),
        "customer_id": get_customer_id(),
        "phone_number_hash": get_phone_number_hash(),
        "event_type": event_type,
        "status": status,
        "duration_ms": duration_ms,
        "payload": payload,
        **kwargs
    }
    
    try:
        # Non-blocking add to queue
        loop = asyncio.get_event_loop()
        if loop.is_running():
            get_log_queue().put_nowait(db_data)
    except Exception:
        # If no loop or queue full, we still have stdout log
        pass
