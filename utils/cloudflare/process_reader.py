"""Background cloudflared process output reader."""

import asyncio
import re
import threading
from typing import Any

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
        self._task = None
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

    @staticmethod
    def _normalize_line(raw_line):
        if isinstance(raw_line, bytes):
            return raw_line.decode("utf-8", errors="replace").strip()
        return str(raw_line).strip()

    def _process_log_line(self, line):
        self._append_log(line)
        match = PUBLIC_URL_PATTERN.search(line)
        if match and not self._public_url:
            self._public_url = match.group(1).rstrip("/")
            return True

        if "error" in line.lower() and not self._last_error:
            self._last_error = line
        return False

    def _reader(self):
        process = self._process
        if process is None:
            return

        loop = self._loop
        for raw_line in iter(process.stdout.readline, ""):
            line = self._normalize_line(raw_line)
            if not line:
                continue

            found_url = self._process_log_line(line)
            if found_url and self._url_event and loop:
                loop.call_soon_threadsafe(self._url_event.set)

        if self._url_event and loop:
            if not self._last_error and not self._public_url:
                self._last_error = "Cloudflare tunnel exited before becoming ready"
            loop.call_soon_threadsafe(self._url_event.set)

    async def _async_reader(self):
        process = self._process
        if process is None or process.stdout is None:
            if self._url_event:
                self._url_event.set()
            return

        while True:
            raw_line = await process.stdout.readline()
            if not raw_line:
                break

            line = self._normalize_line(raw_line)
            if not line:
                continue

            found_url = self._process_log_line(line)
            if found_url and self._url_event:
                self._url_event.set()

        if self._url_event:
            if not self._last_error and not self._public_url:
                self._last_error = "Cloudflare tunnel exited before becoming ready"
            self._url_event.set()

    def start(self, process: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._process = process
        self._loop = loop
        self._url_event = asyncio.Event()
        self._public_url = None
        self._last_error = None
        self._recent_logs = []
        self._thread = None
        self._task = None

        stdout = getattr(process, "stdout", None)
        if stdout is None:
            return

        if asyncio.iscoroutinefunction(getattr(stdout, "readline", None)):
            self._task = loop.create_task(self._async_reader())
        else:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    async def wait_for_url(self, timeout):
        if not self._url_event:
            return None
        await asyncio.wait_for(self._url_event.wait(), timeout=timeout)
        return self._public_url

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

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
