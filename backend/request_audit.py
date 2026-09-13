"""Capture serialized Ollama request bodies without authentication headers."""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

AUDIT_PATH = Path(__file__).resolve().parents[1] / "logs" / "ollama_requests.jsonl"
_lock = threading.Lock()
logger = logging.getLogger(__name__)


def capture_ollama_request(request, *, mode=None, call_type=None):
    """HTTPX request hook: records SDK serialization, not server receipt."""
    if request.method != "POST" or request.url.path != "/api/chat":
        return
    try:
        payload = json.loads(request.content)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "outgoing_ollama_request",
            "host": request.url.host,
            "path": request.url.path,
            "mode": mode,
            "call_type": call_type,
            "payload": {key: payload[key] for key in (
                "model", "messages", "think", "options", "stream", "format", "tools",
            ) if key in payload},
        }
        with _lock:
            AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with AUDIT_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.warning("Could not capture outgoing Ollama request", exc_info=True)
