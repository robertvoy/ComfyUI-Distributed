import importlib.util
import unittest
from pathlib import Path


def _load_queue_request_module():
    module_path = Path(__file__).resolve().parents[3] / "api" / "queue_request.py"
    spec = importlib.util.spec_from_file_location("queue_request", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


queue_request = _load_queue_request_module()
parse_queue_request_payload = queue_request.parse_queue_request_payload


class QueueRequestPayloadTests(unittest.TestCase):
    def _base_payload(self):
        return {
            "prompt": {"1": {"class_type": "Anything"}},
            "enabled_worker_ids": ["worker-1"],
            "client_id": "client-1",
        }

    def test_normalizes_enabled_worker_ids(self):
        payload_data = self._base_payload()
        payload_data["enabled_worker_ids"] = [" worker-a ", "worker-b", "worker-a"]
        payload_data["delegate_master"] = True
        payload = parse_queue_request_payload(
            payload_data
        )
        self.assertEqual(payload.enabled_worker_ids, ["worker-a", "worker-b"])
        self.assertTrue(payload.delegate_master)

    def test_supports_legacy_workers_field(self):
        payload_data = self._base_payload()
        payload_data.pop("enabled_worker_ids", None)
        payload_data["workers"] = [{"id": "w1"}, "w2", {"id": "w3"}, {"name": "no-id"}]
        payload = parse_queue_request_payload(
            payload_data
        )
        self.assertEqual(payload.enabled_worker_ids, ["w1", "w2", "w3"])

    def test_supports_workflow_prompt_fallback(self):
        payload_data = self._base_payload()
        payload_data.pop("prompt", None)
        payload_data["workflow"] = {
            "prompt": {
                "10": {"class_type": "DistributedCollector"},
            }
        }
        payload = parse_queue_request_payload(
            payload_data
        )
        self.assertIn("10", payload.prompt)

    def test_normalizes_trace_execution_id(self):
        payload_data = self._base_payload()
        payload_data["trace_execution_id"] = "  exec_123  "
        payload = parse_queue_request_payload(
            payload_data
        )
        self.assertEqual(payload.trace_execution_id, "exec_123")

    def test_blank_trace_execution_id_normalizes_to_none(self):
        payload_data = self._base_payload()
        payload_data["trace_execution_id"] = "   "
        payload = parse_queue_request_payload(
            payload_data
        )
        self.assertIsNone(payload.trace_execution_id)

    def test_workers_field_must_be_list(self):
        payload_data = self._base_payload()
        payload_data.pop("enabled_worker_ids", None)
        payload_data["workers"] = "worker-a"
        with self.assertRaisesRegex(ValueError, "Field 'workers' must be a list"):
            parse_queue_request_payload(payload_data)

    def test_trace_execution_id_must_be_string(self):
        payload_data = self._base_payload()
        payload_data["trace_execution_id"] = 123
        with self.assertRaisesRegex(ValueError, "trace_execution_id must be a string"):
            parse_queue_request_payload(payload_data)

    def test_auto_prepare_is_ignored_for_backward_compat(self):
        payload_data = self._base_payload()
        payload_data.pop("prompt", None)
        payload_data["auto_prepare"] = False
        payload_data["workflow"] = {
            "prompt": {"10": {"class_type": "DistributedCollector"}},
        }
        payload = parse_queue_request_payload(payload_data)
        self.assertIn("10", payload.prompt)

    def test_invalid_delegate_master_type_raises(self):
        payload_data = self._base_payload()
        payload_data["delegate_master"] = "yes"
        with self.assertRaisesRegex(ValueError, "delegate_master must be a boolean"):
            parse_queue_request_payload(payload_data)

    def test_invalid_enabled_worker_ids_type_raises(self):
        payload_data = self._base_payload()
        payload_data["enabled_worker_ids"] = "worker-a"
        with self.assertRaisesRegex(ValueError, "enabled_worker_ids must be a list"):
            parse_queue_request_payload(payload_data)

    def test_legacy_index_worker_tokens_are_rejected(self):
        payload_data = self._base_payload()
        payload_data["enabled_worker_ids"] = ["worker-a", "0", "worker_1"]
        with self.assertRaisesRegex(ValueError, "legacy index token"):
            parse_queue_request_payload(payload_data)

    def test_invalid_top_level_payload_raises(self):
        with self.assertRaisesRegex(ValueError, "Expected a JSON object body"):
            parse_queue_request_payload(["not", "an", "object"])

    def test_missing_prompt_raises(self):
        with self.assertRaisesRegex(ValueError, "Field 'prompt' must be an object"):
            parse_queue_request_payload(
                {
                    "workflow": {},
                    "enabled_worker_ids": ["worker-1"],
                    "client_id": "client-1",
                }
            )

    def test_missing_enabled_worker_ids_raises(self):
        payload_data = self._base_payload()
        payload_data.pop("enabled_worker_ids", None)
        with self.assertRaisesRegex(ValueError, "enabled_worker_ids required"):
            parse_queue_request_payload(payload_data)

    def test_missing_client_id_raises(self):
        payload_data = self._base_payload()
        payload_data.pop("client_id", None)
        with self.assertRaisesRegex(ValueError, "client_id required"):
            parse_queue_request_payload(payload_data)


if __name__ == "__main__":
    unittest.main()
