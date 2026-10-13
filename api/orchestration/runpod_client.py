"""Small Runpod queue API client. Never retries a potentially billable POST."""
import asyncio
import json
import re

import aiohttp

API_BASE = "https://api.runpod.ai/v2"
MAX_REQUEST_BYTES = 10_000_000
_ID = re.compile(r"[A-Za-z0-9_-]{1,200}\Z")


class RunpodError(RuntimeError):
    """Sanitized provider failure safe to display in the panel."""


class AmbiguousSubmission(RunpodError):
    """Submission may have been accepted: never retry it automatically."""


def serialize_request(workflow, images, timeout_seconds):
    """Serialize and size-check the exact JSON body used by Runpod ``/run``."""
    payload = {
        "input": {"workflow": workflow, "images": images},
        "policy": {
            "executionTimeout": timeout_seconds * 1000,
            "ttl": timeout_seconds * 1000,
        },
    }
    body = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    size = len(body)
    if size > MAX_REQUEST_BYTES:
        raise ValueError(
            f"Serialized serverless request body is {size:,} bytes; Runpod allows at most "
            f"{MAX_REQUEST_BYTES:,} bytes. Base64-encoded images add about 33% over raw "
            "image bytes; reduce image or workflow size."
        )
    return body


def checked_id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid Runpod endpoint or job ID")
    return value


class RunpodClient:
    def __init__(self, session, endpoint_id, api_key):
        self.session = session
        self.endpoint_id = checked_id(endpoint_id)
        if not isinstance(api_key, str) or not api_key or any(c in api_key for c in "\r\n"):
            raise ValueError("A valid Runpod API key is required")
        self._api_key = api_key

    async def _request(self, method, path, payload=None, request_body=None):
        # Base origin is code-owned, never a user-supplied URL. Do not follow redirects
        # with credentials, or include response bodies/transport errors in exceptions.
        url = f"{API_BASE}/{self.endpoint_id}/{path}"
        attempts = 3 if method == "GET" else 1
        for attempt in range(attempts):
            try:
                request_kwargs = {
                    "headers": {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    "timeout": aiohttp.ClientTimeout(total=30),
                    "allow_redirects": False,
                }
                if request_body is None:
                    request_kwargs["json"] = payload
                else:
                    request_kwargs["data"] = request_body
                async with self.session.request(
                    method, url,
                    **request_kwargs,
                ) as response:
                    status = response.status
                    if status != 200:
                        if method == "GET" and (status == 429 or status >= 500) and attempt + 1 < attempts:
                            await asyncio.sleep(0.5 * (2 ** attempt))
                            continue
                        if method == "POST" and path == "run" and status >= 500:
                            raise AmbiguousSubmission("Runpod submission returned a server error; acceptance is unknown. Check the provider before retrying.")
                        raise RunpodError(f"Runpod {path.split('/')[0]} request failed (HTTP {status})")
                    try:
                        result = await response.json()
                    except (ValueError, aiohttp.ContentTypeError):
                        if method == "POST" and path == "run":
                            raise AmbiguousSubmission("Runpod submission returned an unreadable response; check the provider before retrying.") from None
                        raise RunpodError("Runpod returned an invalid JSON response") from None
                    if not isinstance(result, dict):
                        if method == "POST" and path == "run":
                            raise AmbiguousSubmission("Runpod submission did not return a job record; check the provider before retrying.")
                        raise RunpodError("Runpod returned an invalid response")
                    return result
            except (asyncio.TimeoutError, aiohttp.ClientError, OSError):
                if method == "POST" and path == "run":
                    raise AmbiguousSubmission("Runpod submission connection was interrupted; acceptance is unknown. Check the provider before retrying.") from None
                if method == "GET" and attempt + 1 < attempts:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                raise RunpodError(f"Runpod {path.split('/')[0]} connection failed") from None
        raise RunpodError("Runpod request failed")

    async def health(self):
        return await self._request("GET", "health")

    async def submit(self, workflow, images, timeout_seconds):
        body = serialize_request(workflow, images, timeout_seconds)
        result = await self._request("POST", "run", request_body=body)
        try:
            return checked_id(result.get("id"))
        except ValueError:
            raise AmbiguousSubmission("Runpod submission returned no valid job ID; check the provider before retrying.") from None

    async def status(self, job_id):
        return await self._request("GET", f"status/{checked_id(job_id)}")

    async def cancel(self, job_id):
        return await self._request("POST", f"cancel/{checked_id(job_id)}")
