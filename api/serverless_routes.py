"""Master-only Runpod pool monitoring. Health checks never submit inference."""
from aiohttp import web
import server

from ..utils.config import load_config
from .orchestration.serverless_runtime import get_manager
from .orchestration.runpod_client import RunpodError


def _pool(worker_id):
    return next((w for w in load_config().get("workers", [])
                 if w.get("id") == worker_id and w.get("type") == "serverless"), None)


async def _status(worker_id, fresh=False):
    worker = _pool(worker_id)
    if worker is None:
        return web.json_response({"ok": False, "message": "Serverless worker not found"}, status=404)
    try:
        manager = get_manager()
        result = await manager.health(worker, fresh=fresh)
        return web.json_response(dict(result, jobs=manager.public_jobs(worker_id)))
    except Exception as exc:
        message = str(exc) if isinstance(exc, RunpodError) else "Check the saved endpoint ID and API key"
        return web.json_response({"ok": False, "queueRemaining": 0, "message": message})


@server.PromptServer.instance.routes.post("/distributed/serverless/test")
async def test_serverless_endpoint(request):
    try:
        data = await request.json()
        worker_id = data.get("worker_id")
    except Exception:
        return web.json_response({"ok": False, "message": "Invalid request"}, status=400)
    return await _status(worker_id, fresh=True)


@server.PromptServer.instance.routes.get("/distributed/serverless/status/{worker_id}")
async def serverless_status_endpoint(request):
    return await _status(request.match_info["worker_id"])


@server.PromptServer.instance.routes.post("/distributed/serverless/cancel")
async def cancel_serverless_endpoint(request):
    try:
        data = await request.json()
        worker_id = data.get("worker_id")
    except Exception:
        return web.json_response({"ok": False, "message": "Invalid request"}, status=400)
    if _pool(worker_id) is None:
        return web.json_response({"ok": False, "message": "Serverless worker not found"}, status=404)
    try:
        manager = get_manager()
        count = await manager.cancel(worker_id=worker_id)
        return web.json_response({"ok": True, "cancelled": count, "jobs": manager.public_jobs(worker_id),
                                  "message": "Cancellation requested for this master's jobs only"})
    except Exception:
        return web.json_response({"ok": False, "message": "Cancellation could not be confirmed; check provider jobs"}, status=502)
