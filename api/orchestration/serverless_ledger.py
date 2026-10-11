"""Private, atomic job-ID journal. Contains identifiers/status only, never payloads."""
import json
import os
from pathlib import Path
import tempfile
import threading

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
FIELDS = {"record_id", "execution_id", "worker_id", "participant_id", "endpoint_id", "job_id", "status", "created_at", "updated_at", "error", "prompt_id"}


class JobLedger:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def _read(self):
        if self.path.is_symlink():
            raise RuntimeError("Unsafe serverless job journal path")
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or any(not isinstance(v, dict) or set(v) - FIELDS for v in data.values()):
                raise ValueError()
            return data
        except (ValueError, OSError):
            raise RuntimeError("Serverless job journal is unreadable; preserve it for recovery before submitting more work") from None

    def all(self):
        with self._lock:
            return list(self._read().values())

    def update(self, record_id, **fields):
        if set(fields) - FIELDS:
            raise ValueError("Unexpected serverless journal field")
        with self._lock:
            data = self._read()
            data.setdefault(record_id, {"record_id": record_id}).update(fields)
            # Keep every unresolved job, plus a bounded terminal history.
            completed = sorted((k for k, v in data.items() if v.get("status") in TERMINAL), key=lambda k: data[k].get("updated_at", 0))
            for key in completed[:-200]:
                data.pop(key)
            if self.path.parent.is_symlink():
                raise RuntimeError("Unsafe serverless journal directory")
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".jobs-", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(data, stream, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return dict(data[record_id])
