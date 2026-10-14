import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = copy.deepcopy(payload)
        self.status = status


class _FakeRequest:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    async def json(self):
        if self.error is not None:
            raise self.error
        return self.payload


class _Routes:
    def get(self, _path):
        return lambda handler: handler

    def post(self, _path):
        return lambda handler: handler


def _load_modules():
    repo = Path(__file__).resolve().parents[1]
    package_name = "dist_serverless_routes_testpkg"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]

    root = types.ModuleType(package_name)
    root.__path__ = [str(repo)]
    sys.modules[package_name] = root
    api_pkg = types.ModuleType(f"{package_name}.api")
    api_pkg.__path__ = [str(repo / "api")]
    sys.modules[f"{package_name}.api"] = api_pkg
    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = [str(repo / "utils")]
    sys.modules[f"{package_name}.utils"] = utils_pkg
    orchestration_pkg = types.ModuleType(f"{package_name}.api.orchestration")
    orchestration_pkg.__path__ = [str(repo / "api" / "orchestration")]
    sys.modules[f"{package_name}.api.orchestration"] = orchestration_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.log = lambda *_args, **_kwargs: None
    logging_module.debug_log = lambda *_args, **_kwargs: None
    sys.modules[logging_module.__name__] = logging_module
    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.HEARTBEAT_TIMEOUT = 30
    sys.modules[constants_module.__name__] = constants_module

    def load_file(module_name, path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    config_module = load_file(f"{package_name}.utils.config", repo / "utils" / "config.py")
    serverless_config = load_file(
        f"{package_name}.utils.serverless_config", repo / "utils" / "serverless_config.py"
    )

    runtime = types.ModuleType(f"{package_name}.api.orchestration.serverless_runtime")
    runtime.secret_store = None
    runtime.manager = types.SimpleNamespace(tasks={}, runs={}, pool_is_active=lambda _: False)
    runtime.get_secret_store = lambda: runtime.secret_store
    runtime.get_manager = lambda: runtime.manager
    sys.modules[runtime.__name__] = runtime

    import aiohttp  # Ensure the real dependency is loaded; never replace global aiohttp.
    previous_server = sys.modules.get("server")
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes()))
    sys.modules["server"] = server

    network = types.ModuleType(f"{package_name}.utils.network")

    async def handle_api_error(_request, error, status=500):
        return _FakeResponse({"status": "error", "message": str(error)}, status)

    network.handle_api_error = handle_api_error
    network.normalize_host = lambda value: value
    sys.modules[network.__name__] = network

    routes = load_file(f"{package_name}.api.config_routes", repo / "api" / "config_routes.py")
    routes.web = types.SimpleNamespace(
        json_response=lambda payload, status=200: _FakeResponse(payload, status)
    )

    if previous_server is None:
        sys.modules.pop("server", None)
    else:
        sys.modules["server"] = previous_server
    return config_module, serverless_config, runtime, routes


config_module, serverless_config, runtime_module, config_routes = _load_modules()


class _RunningTask:
    def done(self):
        return False


class _ConfigRouteFixture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        config_module.CONFIG_FILE = str(self.root / "gpu_config.json")
        config_module.invalidate_config_cache()
        self.config = config_module.get_default_config()
        self.config["workers"] = []
        self.assertTrue(config_module.save_config(self.config))
        self.secrets = serverless_config.SecretStore(self.root / "private" / "credentials.json")
        runtime_module.secret_store = self.secrets
        runtime_module.manager = types.SimpleNamespace(tasks={}, runs={}, pool_is_active=lambda _: False)

    def tearDown(self):
        config_module.invalidate_config_cache()
        self.tempdir.cleanup()

    def read_config_file(self):
        return json.loads(Path(config_module.CONFIG_FILE).read_text(encoding="utf-8"))


class ServerlessConfigRouteTests(_ConfigRouteFixture):
    async def test_get_config_redacts_keys_and_reports_only_configured_boolean(self):
        self.secrets.set("pool-1", "private-api-key")
        self.config["workers"] = [
            {"id": "pool-1", "type": "serverless", "provider": "runpod", "endpoint_id": "ep-1",
             "api_key": "inline-api-key", "name": "GPU"}
        ]
        Path(config_module.CONFIG_FILE).write_text(json.dumps(self.config), encoding="utf-8")
        config_module.invalidate_config_cache()

        response = await config_routes.get_config_endpoint(_FakeRequest())

        self.assertEqual(response.status, 200)
        worker = response.payload["workers"][0]
        self.assertTrue(worker["api_key_configured"])
        self.assertNotIn("api_key", worker)
        self.assertNotIn("private-api-key", json.dumps(response.payload))
        self.assertNotIn("inline-api-key", json.dumps(response.payload))

    async def test_get_config_redacts_normalized_secret_field_aliases(self):
        aliases = {
            "apiKey": "camel-api-key",
            "APIKEY": "upper-api-key",
            "clearApiKey": "camel-clear-key",
            "CLEAR_API_KEY": "upper-clear-key",
            "runpod_CREDENTIAL": "provider-credential",
        }
        self.config["workers"] = [{
            "id": "pool-1", "type": "serverless", "endpoint_id": "ep-123",
            **aliases,
            "nested": {"credential_note": "nested-credential"},
        }]
        Path(config_module.CONFIG_FILE).write_text(json.dumps(self.config), encoding="utf-8")
        config_module.invalidate_config_cache()

        response = await config_routes.get_config_endpoint(_FakeRequest())

        self.assertEqual(response.status, 200)
        worker = response.payload["workers"][0]
        for alias, value in aliases.items():
            self.assertNotIn(alias, worker)
            self.assertNotIn(value, json.dumps(response.payload))
        self.assertNotIn("credential_note", worker["nested"])
        self.assertNotIn("nested-credential", json.dumps(response.payload))

    async def test_bulk_config_rejects_normalized_secret_field_aliases(self):
        for secret_field in ("apiKey", "APIKEY", "clearApiKey", "RUNPOD_CREDENTIAL"):
            with self.subTest(secret_field=secret_field):
                secret = f"private-{secret_field}"
                response = await config_routes.update_config_endpoint(_FakeRequest({
                    "workers": [{"id": "pool-1", "type": "local", secret_field: secret}]
                }))
                self.assertEqual(response.status, 400)
                self.assertNotIn(secret, json.dumps(response.payload))
        self.assertEqual(self.read_config_file()["workers"], [])

    async def test_create_serverless_worker_without_local_fields_and_store_key_privately(self):
        secret = "runpod-secret-never-returned"
        payload = {
            "worker_id": "pool-1", "name": "GPU pool", "type": "serverless", "provider": "runpod",
            "endpoint_id": "ep-123", "jobs_per_run": 3, "concurrency": 2,
            "timeout_seconds": 600, "enabled": True, "api_key": secret,
        }

        response = await config_routes.update_worker_endpoint(_FakeRequest(payload))

        self.assertEqual(response.status, 200)
        self.assertTrue(response.payload["api_key_configured"])
        self.assertNotIn(secret, json.dumps(response.payload))
        saved = self.read_config_file()["workers"][0]
        self.assertEqual(saved["endpoint_id"], "ep-123")
        self.assertEqual(saved["jobs_per_run"], 3)
        # Legacy field from older browser tabs is accepted but not persisted.
        self.assertNotIn("concurrency", saved)
        self.assertEqual(saved["timeout_seconds"], 600)
        for obsolete in ("host", "port", "cuda_device", "extra_args", "api_key", "clear_api_key"):
            self.assertNotIn(obsolete, saved)
        self.assertEqual(self.secrets.get("pool-1"), secret)

    async def test_media_profile_persists_and_is_public_without_changing_credentials(self):
        for profile in ("legacy-image-v1", "comfy-media-v1"):
            with self.subTest(profile=profile):
                response = await config_routes.update_worker_endpoint(_FakeRequest({
                    "worker_id": "pool-1", "type": "serverless", "endpoint_id": "ep-123",
                    "media_profile": profile, "api_key": "saved-key", "enabled": False,
                }))
                self.assertEqual(response.status, 200)
                self.assertEqual(self.read_config_file()["workers"][0]["media_profile"], profile)
                config_module.invalidate_config_cache()
                public = await config_routes.get_config_endpoint(_FakeRequest())
                self.assertEqual(public.status, 200)
                self.assertEqual(public.payload["workers"][0]["media_profile"], profile)
                self.assertTrue(public.payload["workers"][0]["api_key_configured"])
                self.assertNotIn("saved-key", json.dumps(public.payload))
                self.assertEqual(self.secrets.get("pool-1"), "saved-key")

    async def test_old_clients_omit_media_profile_without_downgrading_saved_profile(self):
        for stored in ({}, {"media_profile": "comfy-media-v1"}):
            with self.subTest(stored=stored):
                worker = {"id": "pool-1", "type": "serverless", "endpoint_id": "ep-123", **stored}
                self.config["workers"] = [worker]
                self.assertTrue(config_module.save_config(self.config))
                response = await config_routes.update_worker_endpoint(_FakeRequest({
                    "worker_id": "pool-1", "name": "Old client save", "jobs_per_run": 2,
                }))
                self.assertEqual(response.status, 200)
                saved = self.read_config_file()["workers"][0]
                self.assertEqual({k: v for k, v in saved.items() if k == "media_profile"}, stored)
                # The bulk route is also used by older clients with no profile field.
                response = await config_routes.update_config_endpoint(_FakeRequest({"workers": [{
                    "id": "pool-1", "type": "serverless", "endpoint_id": "ep-123", "jobs_per_run": 3,
                }]}))
                self.assertEqual(response.status, 200)
                saved = self.read_config_file()["workers"][0]
                self.assertEqual({k: v for k, v in saved.items() if k == "media_profile"}, stored)

    async def test_invalid_media_profile_is_rejected_by_worker_and_bulk_routes(self):
        for profile in ("auto", "comfy-media-v2", None, False, 1, [], {}):
            with self.subTest(profile=profile):
                response = await config_routes.update_worker_endpoint(_FakeRequest({
                    "worker_id": "pool-1", "type": "serverless", "endpoint_id": "ep-123",
                    "media_profile": profile, "api_key": "never-save",
                }))
                self.assertEqual(response.status, 400)
                response = await config_routes.update_config_endpoint(_FakeRequest({"workers": [{
                    "id": "pool-1", "type": "serverless", "endpoint_id": "ep-123", "media_profile": profile,
                }]}))
                self.assertEqual(response.status, 400)
                self.assertEqual(self.read_config_file()["workers"], [])
                self.assertIsNone(self.secrets.get("pool-1"))

    async def test_active_pool_blocks_media_profile_changes_in_both_directions(self):
        for old_profile, new_profile in ((None, "comfy-media-v1"), ("comfy-media-v1", "legacy-image-v1")):
            with self.subTest(old_profile=old_profile):
                worker = {"id": "pool-1", "type": "serverless", "endpoint_id": "ep-123"}
                if old_profile is not None:
                    worker["media_profile"] = old_profile
                self.config["workers"] = [worker]
                self.assertTrue(config_module.save_config(self.config))
                original = self.read_config_file()
                with patch.object(config_routes, "_serverless_pool_is_active", return_value=True):
                    response = await config_routes.update_worker_endpoint(_FakeRequest({
                        "worker_id": "pool-1", "media_profile": new_profile,
                    }))
                    self.assertEqual(response.status, 409)
                    response = await config_routes.update_config_endpoint(_FakeRequest({
                        "workers": [dict(worker, media_profile=new_profile)],
                    }))
                    self.assertEqual(response.status, 409)
                self.assertEqual(self.read_config_file(), original)

    async def test_enable_only_update_preserves_serverless_settings_and_saved_key(self):
        self.secrets.set("pool-1", "saved-secret")
        self.config["workers"] = [{
            "id": "pool-1", "type": "serverless", "provider": "runpod", "name": "Pool",
            "endpoint_id": "ep-123", "jobs_per_run": 4, "concurrency": 3,
            "timeout_seconds": 900, "enabled": False,
        }]
        self.assertTrue(config_module.save_config(self.config))

        response = await config_routes.update_worker_endpoint(
            _FakeRequest({"worker_id": "pool-1", "enabled": True})
        )

        worker = self.read_config_file()["workers"][0]
        self.assertTrue(worker["enabled"])
        self.assertEqual(worker["endpoint_id"], "ep-123")
        self.assertEqual(worker["jobs_per_run"], 4)
        self.assertNotIn("concurrency", worker)
        self.assertEqual(worker["timeout_seconds"], 900)
        self.assertEqual(self.secrets.get("pool-1"), "saved-secret")
        self.assertTrue(response.payload["api_key_configured"])

    async def test_invalid_serverless_record_is_not_persisted_or_secret_written(self):
        secret = "must-not-leak"
        response = await config_routes.update_worker_endpoint(_FakeRequest({
            "worker_id": "pool-1", "type": "serverless", "provider": "runpod",
            "endpoint_id": "ep-123", "jobs_per_run": True, "api_key": secret,
        }))

        self.assertEqual(response.status, 400)
        self.assertNotIn(secret, json.dumps(response.payload))
        self.assertEqual(self.read_config_file()["workers"], [])
        self.assertIsNone(self.secrets.get("pool-1"))

    async def test_json_decode_error_is_generic_and_does_not_echo_body(self):
        secret = "body-secret-value"
        response = await config_routes.update_worker_endpoint(
            _FakeRequest(error=ValueError(f"bad json near {secret}"))
        )
        self.assertEqual(response.status, 400)
        self.assertNotIn(secret, json.dumps(response.payload))

    async def test_bulk_config_rejects_api_keys_at_any_depth_without_persisting(self):
        secret = "bulk-secret-never-write"
        response = await config_routes.update_config_endpoint(_FakeRequest({
            "workers": [{
                "id": "pool-1", "type": "serverless", "provider": "runpod",
                "endpoint_id": "ep-123", "nested": {"api_key": secret},
            }]
        }))

        self.assertEqual(response.status, 400)
        self.assertNotIn(secret, json.dumps(response.payload))
        self.assertEqual(self.read_config_file()["workers"], [])
        self.assertFalse(self.secrets.path.exists())

    async def test_bulk_update_validates_and_returns_redacted_serverless_config(self):
        self.secrets.set("pool-1", "saved-secret")
        response = await config_routes.update_config_endpoint(_FakeRequest({
            "workers": [{
                "id": "pool-1", "type": "serverless", "provider": "runpod",
                "endpoint_id": "ep-123", "name": "Pool", "enabled": True,
                "host": "obsolete-host", "port": 8188, "cuda_device": 0,
            }]
        }))

        self.assertEqual(response.status, 200)
        self.assertTrue(response.payload["config"]["workers"][0]["api_key_configured"])
        saved = self.read_config_file()["workers"][0]
        for obsolete in ("host", "port", "cuda_device"):
            self.assertNotIn(obsolete, saved)
        self.assertNotIn("api_key", response.payload["config"]["workers"][0])
        self.assertEqual(self.secrets.get("pool-1"), "saved-secret")

    async def test_bulk_type_conversion_clears_secret_and_serverless_fields(self):
        self.secrets.set("pool-1", "bulk-clear-key")
        self.config["workers"] = [{
            "id": "pool-1", "type": "serverless", "provider": "runpod", "endpoint_id": "ep-123",
        }]
        self.assertTrue(config_module.save_config(self.config))

        response = await config_routes.update_config_endpoint(_FakeRequest({
            "workers": [{
                "id": "pool-1", "type": "local", "port": 8188,
                "provider": "runpod", "endpoint_id": "legacy-ep", "api_key_configured": True,
                "media_profile": "comfy-media-v1",
            }]
        }))

        self.assertEqual(response.status, 200)
        self.assertIsNone(self.secrets.get("pool-1"))
        saved = self.read_config_file()["workers"][0]
        for field in ("provider", "endpoint_id", "api_key_configured", "media_profile"):
            self.assertNotIn(field, saved)

    async def test_bulk_invalid_serverless_record_does_not_replace_configuration(self):
        existing = {"id": "local-1", "type": "local", "port": 8188}
        self.config["workers"] = [existing]
        self.assertTrue(config_module.save_config(self.config))

        response = await config_routes.update_config_endpoint(_FakeRequest({
            "workers": [{"id": "pool-1", "type": "serverless", "endpoint_id": "bad/id"}]
        }))

        self.assertEqual(response.status, 400)
        self.assertEqual(self.read_config_file()["workers"], [existing])

    async def test_delete_serverless_worker_clears_saved_key(self):
        self.secrets.set("pool-1", "delete-me")
        self.config["workers"] = [{
            "id": "pool-1", "type": "serverless", "provider": "runpod", "endpoint_id": "ep-123"
        }]
        self.assertTrue(config_module.save_config(self.config))

        response = await config_routes.delete_worker_endpoint(_FakeRequest({"worker_id": "pool-1"}))

        self.assertEqual(response.status, 200)
        self.assertEqual(self.read_config_file()["workers"], [])
        self.assertIsNone(self.secrets.get("pool-1"))

    async def test_type_conversion_clears_secret_and_removes_serverless_fields(self):
        self.secrets.set("pool-1", "old-key")
        self.config["workers"] = [{
            "id": "pool-1", "name": "Pool", "type": "serverless", "provider": "runpod",
            "endpoint_id": "ep-123", "jobs_per_run": 2, "concurrency": 2, "timeout_seconds": 600,
            "media_profile": "comfy-media-v1",
        }]
        self.assertTrue(config_module.save_config(self.config))

        response = await config_routes.update_worker_endpoint(_FakeRequest({
            "worker_id": "pool-1", "type": "local", "name": "Local", "port": 8188,
            "cuda_device": 0, "enabled": False,
        }))

        self.assertEqual(response.status, 200)
        self.assertIsNone(self.secrets.get("pool-1"))
        worker = self.read_config_file()["workers"][0]
        self.assertEqual(worker["type"], "local")
        for serverless_field in ("provider", "endpoint_id", "jobs_per_run", "concurrency", "timeout_seconds", "media_profile"):
            self.assertNotIn(serverless_field, worker)

    async def test_enabled_toggle_allowed_during_active_run_but_endpoint_change_is_rejected(self):
        self.secrets.set("pool-1", "saved-key")
        self.config["workers"] = [{
            "id": "pool-1", "type": "serverless", "provider": "runpod", "endpoint_id": "ep-123",
            "enabled": False,
        }]
        self.assertTrue(config_module.save_config(self.config))
        runtime_module.manager = types.SimpleNamespace(
            tasks={"run-1": _RunningTask()},
            runs={"run-1": {"worker": {"id": "serverless:pool-1:1", "pool_id": "pool-1"}}},
            pool_is_active=lambda worker_id: worker_id == "pool-1",
        )

        enabled = await config_routes.update_worker_endpoint(
            _FakeRequest({"worker_id": "pool-1", "enabled": True})
        )
        blocked = await config_routes.update_worker_endpoint(_FakeRequest({
            "worker_id": "pool-1", "endpoint_id": "ep-456",
        }))

        self.assertEqual(enabled.status, 200)
        self.assertEqual(blocked.status, 409)
        self.assertEqual(self.read_config_file()["workers"][0]["endpoint_id"], "ep-123")

    async def test_blank_api_key_keeps_saved_key_and_clear_api_key_deletes_it(self):
        self.secrets.set("pool-1", "saved-key")
        self.config["workers"] = [{
            "id": "pool-1", "type": "serverless", "provider": "runpod", "endpoint_id": "ep-123",
        }]
        self.assertTrue(config_module.save_config(self.config))

        keep = await config_routes.update_worker_endpoint(_FakeRequest({
            "worker_id": "pool-1", "api_key": "",
        }))
        clear = await config_routes.update_worker_endpoint(_FakeRequest({
            "worker_id": "pool-1", "clear_api_key": True,
        }))

        self.assertTrue(keep.payload["api_key_configured"])
        self.assertFalse(clear.payload["api_key_configured"])
        self.assertIsNone(self.secrets.get("pool-1"))

    async def test_config_save_refuses_inline_api_keys_and_invalid_serverless_records(self):
        with self.subTest(case="inline key"):
            config = config_module.get_default_config()
            config["workers"] = [{"id": "pool-1", "type": "local", "api_key": "private"}]
            self.assertFalse(config_module.save_config(config))
        with self.subTest(case="invalid serverless record"):
            config = config_module.get_default_config()
            config["workers"] = [{"id": "pool-1", "type": "serverless", "endpoint_id": "bad/id"}]
            self.assertFalse(config_module.save_config(config))

    async def test_active_pool_blocks_credential_changes_type_conversion_and_deletion(self):
        self.secrets.set("pool-1", "saved-key")
        self.config["workers"] = [{
            "id": "pool-1", "type": "serverless", "provider": "runpod", "endpoint_id": "ep-123",
        }]
        self.assertTrue(config_module.save_config(self.config))
        runtime_module.manager = types.SimpleNamespace(
            tasks={"run-1": _RunningTask()},
            runs={"run-1": {"worker": {"id": "serverless:pool-1:1", "pool_id": "pool-1"}}},
            pool_is_active=lambda worker_id: worker_id == "pool-1",
        )

        credential_change = await config_routes.update_worker_endpoint(_FakeRequest({
            "worker_id": "pool-1", "api_key": "new-key",
        }))
        type_change = await config_routes.update_worker_endpoint(_FakeRequest({
            "worker_id": "pool-1", "type": "local", "port": 8188, "cuda_device": 0,
        }))
        deletion = await config_routes.delete_worker_endpoint(_FakeRequest({"worker_id": "pool-1"}))

        self.assertEqual(credential_change.status, 409)
        self.assertEqual(type_change.status, 409)
        self.assertEqual(deletion.status, 409)
        self.assertEqual(self.secrets.get("pool-1"), "saved-key")
        self.assertEqual(self.read_config_file()["workers"][0]["type"], "serverless")

    async def test_secret_write_is_rolled_back_when_config_persistence_fails(self):
        secret = "write-then-rollback"
        with patch.object(config_module, "save_config", return_value=False):
            response = await config_routes.update_worker_endpoint(_FakeRequest({
                "worker_id": "pool-1", "type": "serverless", "endpoint_id": "ep-123",
                "api_key": secret,
            }))

        self.assertEqual(response.status, 500)
        self.assertIsNone(self.secrets.get("pool-1"))
        self.assertEqual(self.read_config_file()["workers"], [])
        self.assertEqual(config_module.load_config()["workers"], [])

    async def test_bulk_secret_delete_is_rolled_back_when_config_persistence_fails(self):
        self.secrets.set("pool-1", "bulk-restore-key")
        self.config["workers"] = [{
            "id": "pool-1", "type": "serverless", "endpoint_id": "ep-123",
        }]
        self.assertTrue(config_module.save_config(self.config))

        with patch.object(config_module, "save_config", return_value=False):
            response = await config_routes.update_config_endpoint(_FakeRequest({
                "workers": [{"id": "pool-1", "type": "local", "port": 8188}]
            }))

        self.assertEqual(response.status, 500)
        self.assertEqual(self.secrets.get("pool-1"), "bulk-restore-key")
        self.assertEqual(self.read_config_file()["workers"][0]["type"], "serverless")

    async def test_secret_replacement_is_rolled_back_when_config_persistence_fails(self):
        self.secrets.set("pool-1", "prior-key")
        self.config["workers"] = [{
            "id": "pool-1", "type": "serverless", "endpoint_id": "ep-123",
        }]
        self.assertTrue(config_module.save_config(self.config))

        with patch.object(config_module, "save_config", return_value=False):
            response = await config_routes.update_worker_endpoint(_FakeRequest({
                "worker_id": "pool-1", "api_key": "replacement-key",
            }))

        self.assertEqual(response.status, 500)
        self.assertEqual(self.secrets.get("pool-1"), "prior-key")
        self.assertEqual(self.read_config_file()["workers"][0]["type"], "serverless")

    async def test_delete_secret_is_rolled_back_when_config_persistence_fails(self):
        self.secrets.set("pool-1", "delete-restore-key")
        self.config["workers"] = [{
            "id": "pool-1", "type": "serverless", "endpoint_id": "ep-123",
        }]
        self.assertTrue(config_module.save_config(self.config))

        with patch.object(config_module, "save_config", return_value=False):
            response = await config_routes.delete_worker_endpoint(_FakeRequest({"worker_id": "pool-1"}))

        self.assertEqual(response.status, 500)
        self.assertEqual(self.secrets.get("pool-1"), "delete-restore-key")
        self.assertEqual(self.read_config_file()["workers"][0]["id"], "pool-1")

    async def test_regular_worker_delete_does_not_require_posix_secret_storage(self):
        self.config["workers"] = [{"id": "local-1", "type": "local", "port": 8188}]
        self.assertTrue(config_module.save_config(self.config))
        with patch.object(config_routes, "_serverless_secret_store", side_effect=RuntimeError("POSIX only")):
            response = await config_routes.delete_worker_endpoint(_FakeRequest({"worker_id": "local-1"}))
        self.assertEqual(response.status, 200)
        self.assertEqual(self.read_config_file()["workers"], [])

    async def test_enabling_serverless_without_key_is_rejected(self):
        response = await config_routes.update_worker_endpoint(_FakeRequest({
            "worker_id": "pool-1", "type": "serverless", "endpoint_id": "ep-123", "enabled": True,
        }))
        self.assertEqual(response.status, 400)
        self.assertEqual(self.read_config_file()["workers"], [])

    async def test_bulk_endpoint_change_is_blocked_during_active_run(self):
        self.config["workers"] = [{"id": "pool-1", "type": "serverless", "endpoint_id": "ep-123"}]
        self.assertTrue(config_module.save_config(self.config))
        runtime_module.manager = types.SimpleNamespace(
            tasks={"run-1": _RunningTask()},
            runs={"run-1": {"worker": {"id": "serverless:pool-1:1", "pool_id": "pool-1"}}},
            pool_is_active=lambda worker_id: worker_id == "pool-1",
        )
        updated = copy.deepcopy(self.config["workers"])
        updated[0]["endpoint_id"] = "ep-other"
        response = await config_routes.update_config_endpoint(_FakeRequest({"workers": updated}))
        self.assertEqual(response.status, 409)
        self.assertEqual(self.read_config_file()["workers"][0]["endpoint_id"], "ep-123")

    async def test_non_serverless_update_behavior_is_preserved(self):
        self.config["workers"] = [{"id": "local-1", "name": "Local", "type": "local", "port": 8188}]
        self.assertTrue(config_module.save_config(self.config))

        response = await config_routes.update_worker_endpoint(_FakeRequest({
            "worker_id": "local-1", "name": "Renamed", "enabled": True,
        }))

        self.assertEqual(response.status, 200)
        worker = self.read_config_file()["workers"][0]
        self.assertEqual(worker["name"], "Renamed")
        self.assertTrue(worker["enabled"])

    async def test_config_normalization_removes_legacy_local_fields_from_serverless_records(self):
        raw = {
            "workers": [{
                "id": "pool-1", "type": "serverless", "endpoint_id": "ep-123",
                "host": "localhost", "port": 8188, "cuda_device": 2, "extra_args": "--old",
            }]
        }
        normalized = config_module._merge_with_defaults(raw, config_module.get_default_config())
        worker = normalized["workers"][0]
        for obsolete in ("host", "port", "cuda_device", "extra_args"):
            self.assertNotIn(obsolete, worker)


class PersistedPoolGuardTests(_ConfigRouteFixture):
    """Use the actual manager/ledger across restart, not a mocked idle result."""

    def setUp(self):
        super().setUp()
        from test_serverless_runtime import runtime
        from unittest.mock import AsyncMock
        self.runtime = runtime
        self.session_factory = AsyncMock()
        self.ledger_path = self.root / "jobs.json"
        self.manager = runtime.ServerlessManager(types.SimpleNamespace(), self.secrets, self.ledger_path, self.session_factory)
        runtime_module.manager = self.manager
        self.secrets.set("pool-1", "saved-key")
        self.config["workers"] = [{"id": "pool-1", "type": "serverless", "endpoint_id": "ep-123", "concurrency": 1}]
        self.assertTrue(config_module.save_config(self.config))

    async def test_retained_jobs_block_destructive_changes_after_restart(self):
        original = self.read_config_file()
        for status in ("UNKNOWN", "CANCEL_REQUESTED", "IN_QUEUE", "IN_PROGRESS", "DOWNLOADING", "SUBMITTING"):
            for restart in (False, True):
                with self.subTest(status=status, restart=restart):
                    self.manager.ledger.update("retained", worker_id="pool-1", endpoint_id="ep-123", status=status, job_id=None if status in {"UNKNOWN", "SUBMITTING"} else "owned-id")
                    if restart:
                        self.manager = self.runtime.ServerlessManager(types.SimpleNamespace(), self.secrets, self.ledger_path, self.session_factory)
                        runtime_module.manager = self.manager
                    else:
                        self.manager.tasks["retained"] = types.SimpleNamespace(done=lambda: True)
                    for fields in ({"api_key": "new-key"}, {"clear_api_key": True}, {"endpoint_id": "ep-other"}, {"type": "local", "port": 8188}, {"media_profile": "comfy-media-v1"}):
                        response = await config_routes.update_worker_endpoint(_FakeRequest(dict(fields, worker_id="pool-1")))
                        self.assertEqual(response.status, 409, fields)
                    response = await config_routes.delete_worker_endpoint(_FakeRequest({"worker_id": "pool-1"}))
                    self.assertEqual(response.status, 409)
                    worker = original["workers"][0]
                    for workers in ([], [dict(worker, type="local")], [dict(worker, endpoint_id="ep-other")], [dict(worker, media_profile="comfy-media-v1")]):
                        response = await config_routes.update_config_endpoint(_FakeRequest({"workers": workers}))
                        self.assertEqual(response.status, 409, workers)
                    self.assertEqual(self.read_config_file(), original)
                    self.assertEqual(self.secrets.get("pool-1"), "saved-key")
        self.session_factory.assert_not_awaited()

    async def test_retained_jobs_allow_same_or_omitted_media_profile(self):
        self.manager.ledger.update("retained", worker_id="pool-1", endpoint_id="ep-123", status="UNKNOWN")
        for profile in ("legacy-image-v1", "comfy-media-v1"):
            with self.subTest(profile=profile):
                worker = {"id": "pool-1", "type": "serverless", "endpoint_id": "ep-123"}
                # An absent legacy field and an explicit legacy selection are equivalent.
                if profile != "legacy-image-v1":
                    worker["media_profile"] = profile
                self.config["workers"] = [worker]
                self.assertTrue(config_module.save_config(self.config))
                for fields in ({"media_profile": profile}, {}):
                    response = await config_routes.update_worker_endpoint(_FakeRequest({
                        "worker_id": "pool-1", "name": "Same profile", **fields,
                    }))
                    self.assertEqual(response.status, 200)
                    response = await config_routes.update_config_endpoint(_FakeRequest({"workers": [{
                        "id": "pool-1", "type": "serverless", "endpoint_id": "ep-123", **fields,
                    }]}))
                    self.assertEqual(response.status, 200)
                    self.assertEqual(self.read_config_file()["workers"][0]["media_profile"], profile)
        self.session_factory.assert_not_awaited()

    async def test_retained_jobs_allow_images_per_run_and_legacy_concurrency_saves(self):
        # Upgrading from a config with concurrency=1 must not trip the busy guard,
        # and the image count stays editable while old jobs are unresolved.
        self.manager.ledger.update("retained", worker_id="pool-1", endpoint_id="ep-123", status="UNKNOWN")
        response = await config_routes.update_worker_endpoint(_FakeRequest({
            "worker_id": "pool-1", "endpoint_id": "ep-123", "jobs_per_run": 2, "concurrency": 2,
        }))
        self.assertEqual(response.status, 200)
        saved = self.read_config_file()["workers"][0]
        self.assertEqual(saved["jobs_per_run"], 2)
        self.assertNotIn("concurrency", saved)
        response = await config_routes.update_config_endpoint(_FakeRequest({"workers": [dict(saved, jobs_per_run=3, concurrency=5)]}))
        self.assertEqual(response.status, 200)
        self.assertEqual(self.read_config_file()["workers"][0]["jobs_per_run"], 3)

    async def test_retained_job_allows_disable_and_rename_but_not_key_clear(self):
        self.manager.ledger.update("retained", worker_id="pool-1", status="UNKNOWN")
        response = await config_routes.update_worker_endpoint(_FakeRequest({"worker_id": "pool-1", "enabled": False, "name": "Paused"}))
        self.assertEqual(response.status, 200)
        self.assertEqual(self.read_config_file()["workers"][0]["name"], "Paused")
        response = await config_routes.update_worker_endpoint(_FakeRequest({"worker_id": "pool-1", "clear_api_key": True}))
        self.assertEqual(response.status, 409)
        self.assertEqual(self.secrets.get("pool-1"), "saved-key")

    async def test_terminal_or_other_pool_records_do_not_block_deletion(self):
        self.manager.ledger.update("done", worker_id="pool-1", status="RESOLVED")
        self.manager.ledger.update("other", worker_id="other", status="UNKNOWN")
        response = await config_routes.delete_worker_endpoint(_FakeRequest({"worker_id": "pool-1"}))
        self.assertEqual(response.status, 200)
        self.assertIsNone(self.secrets.get("pool-1"))


if __name__ == "__main__":
    unittest.main()
