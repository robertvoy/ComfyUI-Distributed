"""Background cloudflared process output reader."""

import asyncio
import re
import threading

from ..constants import CLOUDFLARE_LOG_BUFFER_SIZE
from ..logging import debug_log

PUBLIC_URL_PATTERN = re.compile(
    r"(https?://[\w.-]+\.(?:trycloudflare\.com|cloudflare\.dev))",
    re.IGNORECASE,
)
class ProcessReader:
    def __init__(self, log_file=None):
        self._process = None
        self._thread = None
        self._loop = None
        self._url_event = None
        self._public_url = None
        self._last_error = None
        self._recent_logs = []
        self._log_file = log_file

    def set_log_file(self, log_file):
        self._log_file = log_file

    def _append_log(self, line):
        if self._log_file:
            try:
                with open(self._log_file, "a", encoding="utf-8", errors="replace") as f:
                    f.write(line + "\n")
            except Exception as exc:  # pragma: no cover
                debug_log(f"Failed to write tunnel log: {exc}")

        self._recent_logs.append(line)
        if len(self._recent_logs) > CLOUDFLARE_LOG_BUFFER_SIZE:
            self._recent_logs = self._recent_logs[-CLOUDFLARE_LOG_BUFFER_SIZE:]

    def _reader(self):
        process = self._process
        if process is None:
            return

        loop = self._loop
        # Captured once: stop() may clear self._url_event while this thread runs.
        url_event = self._url_event
        for raw_line in iter(process.stdout.readline, ""):
            line = raw_line.strip()
            if not line:
                continue

            self._append_log(line)
            match = PUBLIC_URL_PATTERN.search(line)
            if match and not self._public_url:
                self._public_url = match.group(1).rstrip("/")
                if url_event and loop:
                    try:
                        loop.call_soon_threadsafe(url_event.set)
                    except RuntimeError:
                        pass  # The loop is already closed (shutdown).

            if "error" in line.lower() and not self._last_error:
                self._last_error = line

        if url_event and loop:
            if not self._last_error and not self._public_url:
                self._last_error = "Cloudflare tunnel exited before becoming ready"
            try:
                loop.call_soon_threadsafe(url_event.set)
            except RuntimeError:
                pass  # The loop is already closed (shutdown).

    def start(self, process, loop):
        self._process = process
        self._loop = loop
        self._url_event = asyncio.Event()
        self._public_url = None
        self._last_error = None
        self._recent_logs = []
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    async def wait_for_url(self, timeout):
        if not self._url_event:
            return None
        await asyncio.wait_for(self._url_event.wait(), timeout=timeout)
        return self._public_url

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._thread = None
        self._process = None
        self._loop = None
        self._url_event = None

    def get_url(self):
        return self._public_url

    def get_last_error(self):
        return self._last_error

    def get_recent_logs(self):
        return list(self._recent_logs)
