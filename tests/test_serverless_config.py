import importlib.util
import json
import os
import stat
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


def _load_serverless_config_module(*, without_fcntl=False):
    module_path = Path(__file__).resolve().parents[1] / "utils" / "serverless_config.py"
    module_name = (
        "dist_serverless_config_no_fcntl_test_module"
        if without_fcntl
        else "dist_serverless_config_test_module"
    )
    package_name = module_name + "_pkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(module_path.parent)]
    sys.modules[package_name] = package
    module_name = package_name + ".serverless_config"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if without_fcntl:
        with mock.patch.dict(sys.modules, {"fcntl": None}):
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)
    return module


serverless_config = _load_serverless_config_module()


class ValidateServerlessWorkerTests(unittest.TestCase):
    def test_returns_copy_with_runpod_defaults(self):
        worker = {"id": "worker-1", "endpoint_id": "ep_123-abc", "name": "GPU"}

        result = serverless_config.validate_serverless_worker(worker)

        self.assertEqual(
            result,
            {
                "id": "worker-1",
                "endpoint_id": "ep_123-abc",
                "name": "GPU",
                "type": "serverless",
                "provider": "runpod",
                "jobs_per_run": 1,
                "timeout_seconds": 1800,
            },
        )
        self.assertIsNot(result, worker)
        self.assertEqual(worker, {"id": "worker-1", "endpoint_id": "ep_123-abc", "name": "GPU"})

    def test_accepts_inclusive_numeric_bounds(self):
        for jobs, timeout in ((1, 30), (16, 86400)):
            with self.subTest(jobs=jobs, timeout=timeout):
                result = serverless_config.validate_serverless_worker(
                    {
                        "id": "worker-1",
                        "endpoint_id": "ep-123",
                        "jobs_per_run": jobs,
                        "timeout_seconds": timeout,
                    }
                )
                self.assertEqual(result["jobs_per_run"], jobs)
                self.assertEqual(result["timeout_seconds"], timeout)

    def test_legacy_concurrency_is_accepted_and_dropped(self):
        # Configs saved before Images per run replaced the separate setting.
        for legacy in (1, 2, 16, 17, True, "3"):
            with self.subTest(legacy=legacy):
                result = serverless_config.validate_serverless_worker(
                    {"id": "worker-1", "endpoint_id": "ep-123", "jobs_per_run": 2, "concurrency": legacy}
                )
                self.assertNotIn("concurrency", result)
                self.assertEqual(result["jobs_per_run"], 2)

    def test_endpoint_url_is_rejected_with_a_plain_message(self):
        with self.assertRaisesRegex(ValueError, "not a URL"):
            serverless_config.validate_serverless_worker(
                {"id": "worker-1", "endpoint_id": "https://api.runpod.ai/v2/abc123/run"}
            )

    def test_rejects_missing_or_empty_id_and_endpoint(self):
        base: dict[str, object] = {"id": "worker-1", "endpoint_id": "ep-123"}
        for field, value in (("id", None), ("id", "  "), ("endpoint_id", ""), ("endpoint_id", "https://api.runpod.io/ep-123")):
            with self.subTest(field=field, value=value):
                worker = dict(base)
                worker[field] = value
                with self.assertRaises(ValueError):
                    serverless_config.validate_serverless_worker(worker)

        worker = dict(base)
        del worker["id"]
        with self.assertRaises(ValueError):
            serverless_config.validate_serverless_worker(worker)

    def test_endpoint_id_must_be_ascii_opaque_identifier(self):
        for endpoint_id in ("ep/123", "ep 123", "ép-123", "ep:123", "https://example.test"):
            with self.subTest(endpoint_id=endpoint_id):
                with self.assertRaises(ValueError):
                    serverless_config.validate_serverless_worker(
                        {"id": "worker-1", "endpoint_id": endpoint_id}
                    )

    def test_endpoint_id_matches_runpod_client_length_limit(self):
        valid_id = "e" * 200
        result = serverless_config.validate_serverless_worker(
            {"id": "worker-1", "endpoint_id": valid_id}
        )
        self.assertEqual(result["endpoint_id"], valid_id)
        with self.assertRaises(ValueError):
            serverless_config.validate_serverless_worker(
                {"id": "worker-1", "endpoint_id": "e" * 201}
            )

    def test_enabled_must_be_boolean_when_supplied(self):
        base = {"id": "worker-1", "endpoint_id": "ep-123"}
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                self.assertIs(
                    serverless_config.validate_serverless_worker(
                        {**base, "enabled": enabled}
                    )["enabled"],
                    enabled,
                )
        for enabled in ("false", 0, 1, None):
            with self.subTest(enabled=enabled):
                with self.assertRaises(ValueError):
                    serverless_config.validate_serverless_worker(
                        {**base, "enabled": enabled}
                    )

    def test_rejects_unknown_provider_and_non_serverless_type(self):
        base = {"id": "worker-1", "endpoint_id": "ep-123"}
        for extra in ({"provider": "other"}, {"type": "local"}):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    serverless_config.validate_serverless_worker({**base, **extra})

    def test_numeric_fields_reject_bool_float_and_out_of_range(self):
        base = {"id": "worker-1", "endpoint_id": "ep-123"}
        invalid_values = {
            "jobs_per_run": (True, 1.0, 0, 17),
            "timeout_seconds": (True, 30.0, 29, 86401),
        }
        for field, values in invalid_values.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        serverless_config.validate_serverless_worker({**base, field: value})

    def test_requires_worker_object(self):
        for worker in (None, [], "worker"):
            with self.subTest(worker=worker):
                with self.assertRaises(ValueError):
                    serverless_config.validate_serverless_worker(worker)


class SecretStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.path = self.root / "private" / "runpod-secrets.json"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_set_get_has_delete_and_persists_as_json_mapping(self):
        store = serverless_config.SecretStore(self.path)
        store.set("worker-1", "secret-value")

        self.assertEqual(store.get("worker-1"), "secret-value")
        self.assertTrue(store.has("worker-1"))
        self.assertIsNone(store.get("missing"))
        self.assertFalse(store.has("missing"))
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"worker-1": "secret-value"})
        self.assertEqual(serverless_config.SecretStore(self.path).get("worker-1"), "secret-value")

        store.delete("worker-1")
        self.assertIsNone(store.get("worker-1"))
        self.assertFalse(store.has("worker-1"))

    def test_rejects_api_keys_with_newlines_or_over_4096_characters(self):
        store = serverless_config.SecretStore(self.path)
        for key in ("key\nwith-line-feed", "key\rwith-carriage-return", "k" * 4097):
            with self.subTest(key_length=len(key)):
                with self.assertRaises(serverless_config.SecretStoreError):
                    store.set("worker-1", key)
        store.set("worker-1", "k" * 4096)
        self.assertEqual(store.get("worker-1"), "k" * 4096)

    def test_new_secret_file_and_directory_have_restrictive_modes(self):
        store = serverless_config.SecretStore(self.path)
        store.set("worker-1", "secret-value")

        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.path.parent.stat().st_mode), 0o700)

    def test_missing_store_reads_as_empty_without_creating_files(self):
        store = serverless_config.SecretStore(self.path)

        self.assertIsNone(store.get("worker-1"))
        self.assertFalse(store.has("worker-1"))
        self.assertFalse(self.path.parent.exists())

    def test_corrupt_store_fails_closed_without_overwriting_or_leaking_contents(self):
        self.path.parent.mkdir()
        corrupt = b'{"partial": "NEVER-LEAK-THIS-secret'
        self.path.write_bytes(corrupt)
        store = serverless_config.SecretStore(self.path)

        for operation in (
            lambda: store.get("worker-1"),
            lambda: store.set("worker-1", "replacement-secret"),
            lambda: store.delete("worker-1"),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(Exception) as raised:
                    operation()
                self.assertNotIn("NEVER-LEAK-THIS", str(raised.exception))
                self.assertNotIn("replacement-secret", str(raised.exception))
                self.assertEqual(self.path.read_bytes(), corrupt)

    def test_rejects_non_mapping_or_non_string_store_values_without_leaking_data(self):
        for corrupt in (b'["opaque-stored-value"]', b'{"worker": 23}', b'{"": "opaque-stored-value"}'):
            with self.subTest(corrupt=corrupt):
                self.path.parent.mkdir(exist_ok=True)
                self.path.write_bytes(corrupt)
                with self.assertRaises(Exception) as raised:
                    serverless_config.SecretStore(self.path).get("worker")
                self.assertNotIn("opaque-stored-value", str(raised.exception))

    def test_rejects_symlink_file_and_preserves_link_target(self):
        target = self.root / "target.json"
        target.write_text('{"worker-1": "target-secret"}', encoding="utf-8")
        self.path.parent.mkdir()
        self.path.symlink_to(target)
        store = serverless_config.SecretStore(self.path)

        with self.assertRaises(Exception):
            store.get("worker-1")
        with self.assertRaises(Exception):
            store.set("worker-1", "replacement-secret")
        self.assertEqual(target.read_text(encoding="utf-8"), '{"worker-1": "target-secret"}')

    def test_rejects_symlinked_parent_directory(self):
        real_dir = self.root / "real"
        real_dir.mkdir()
        link_dir = self.root / "linked"
        link_dir.symlink_to(real_dir, target_is_directory=True)
        store = serverless_config.SecretStore(link_dir / "secrets.json")

        with self.assertRaises(Exception):
            store.set("worker-1", "secret-value")
        self.assertFalse((real_dir / "secrets.json").exists())

    def test_rejects_parent_traversal_in_store_path(self):
        store = serverless_config.SecretStore(self.root / "nested" / ".." / "secrets.json")

        with self.assertRaises(Exception):
            store.set("worker-1", "secret-value")
        self.assertFalse((self.root / "secrets.json").exists())

    def test_threaded_updates_from_independent_instances_are_not_lost(self):
        workers = 12
        barrier = threading.Barrier(workers)
        errors = []

        def write_one(index):
            try:
                barrier.wait(timeout=5)
                serverless_config.SecretStore(self.path).set(f"worker-{index}", f"key-{index}")
            except Exception as exc:  # collected to fail in the test thread
                errors.append(exc)

        threads = [threading.Thread(target=write_one, args=(i,)) for i in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads), "writer thread did not finish")
        self.assertEqual(errors, [])
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved, {f"worker-{i}": f"key-{i}" for i in range(workers)})


class PublicConfigTests(unittest.TestCase):
    def test_redacts_credentials_preserves_other_fields_and_marks_only_serverless(self):
        with tempfile.TemporaryDirectory() as tempdir:
            store = serverless_config.SecretStore(Path(tempdir) / "private" / "keys.json")
            store.set("serverless-1", "stored-api-key")
            config = {
                "workers": [
                    {
                        "id": "serverless-1",
                        "type": "serverless",
                        "provider": "runpod",
                        "endpoint_id": "ep-123",
                        "api_key": "inline-secret",
                        "clear_api_key": "another-secret",
                        "credential": "credential-secret",
                        "nested": {"runpod_credential": "nested-secret", "safe": "kept"},
                        "name": "remote GPU",
                    },
                    {"id": "serverless-2", "type": "serverless", "name": "not configured"},
                    {"id": "local-1", "type": "local", "api_key": "local-secret"},
                ],
                "settings": {"mode": "preserved", "credentials_note": "remove-me"},
            }
            original = json.loads(json.dumps(config))

            result = serverless_config.public_config(config, store)

        first, second, local = result["workers"]
        self.assertEqual(first["api_key_configured"], True)
        self.assertEqual(second["api_key_configured"], False)
        self.assertNotIn("api_key_configured", local)
        self.assertNotIn("api_key", first)
        self.assertNotIn("clear_api_key", first)
        self.assertNotIn("credential", first)
        self.assertNotIn("runpod_credential", first["nested"])
        self.assertEqual(first["nested"]["safe"], "kept")
        self.assertNotIn("api_key", local)
        self.assertNotIn("credentials_note", result["settings"])
        self.assertEqual(first["name"], "remote GPU")
        self.assertEqual(config, original)
        self.assertIsNot(result, config)
        self.assertIsNot(result["workers"], config["workers"])

    def test_redacts_normalized_secret_field_aliases(self):
        config = {
            "workers": [{
                "id": "serverless-1",
                "type": "serverless",
                "apiKey": "camel-api-key",
                "APIKEY": "upper-api-key",
                "clearApiKey": "camel-clear-key",
                "CLEAR_API_KEY": "upper-clear-key",
                "runpod_CREDENTIAL": "provider-credential",
                "nested": {"credential_note": "nested-credential"},
                "safe": "preserved",
            }]
        }

        result = serverless_config.public_config(
            config, serverless_config.SecretStore(self.root_path())
        )

        worker = result["workers"][0]
        self.assertEqual(worker, {
            "id": "serverless-1",
            "type": "serverless",
            "nested": {},
            "safe": "preserved",
            "api_key_configured": False,
        })

    def test_handles_non_dict_config_without_mutating_it(self):
        config = ["safe", {"api_key": "secret"}]
        result = serverless_config.public_config(config, serverless_config.SecretStore(self.root_path()))
        self.assertEqual(result, ["safe", {}])
        self.assertEqual(config, ["safe", {"api_key": "secret"}])

    def test_module_import_and_non_serverless_redaction_when_fcntl_unavailable(self):
        module = _load_serverless_config_module(without_fcntl=True)
        store = module.SecretStore(self.root_path())
        config = {
            "workers": [
                {"id": "local-1", "type": "local", "api_key": "inline-secret"}
            ]
        }

        result = module.public_config(config, store)

        self.assertEqual(result, {"workers": [{"id": "local-1", "type": "local"}]})
        self.assertEqual(config["workers"][0]["api_key"], "inline-secret")

    def test_secret_operations_fail_clearly_without_fcntl(self):
        module = _load_serverless_config_module(without_fcntl=True)
        store = module.SecretStore(self.root_path())
        unsupported_message = "Serverless credential storage is currently supported on POSIX masters only"

        for operation in (
            lambda: store.get("worker-1"),
            lambda: store.set("worker-1", "api-key"),
            lambda: store.delete("worker-1"),
            lambda: store.has("worker-1"),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(module.SecretStoreError, unsupported_message):
                    operation()

    @staticmethod
    def root_path():
        return Path(tempfile.gettempdir()) / "not-created-serverless-secret-store" / "secrets.json"


if __name__ == "__main__":
    unittest.main()
