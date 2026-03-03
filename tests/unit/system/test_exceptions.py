import importlib.util
import unittest
from pathlib import Path


def _load_exceptions_module():
    module_path = Path(__file__).resolve().parents[3] / "utils" / "exceptions.py"
    spec = importlib.util.spec_from_file_location("dist_test_exceptions", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


exceptions = _load_exceptions_module()


class ExceptionsModuleTests(unittest.TestCase):
    def test_worker_error_tracks_worker_id(self):
        err = exceptions.WorkerError("worker failed", worker_id="w-1")
        self.assertIsInstance(err, exceptions.DistributedError)
        self.assertEqual(str(err), "worker failed")
        self.assertEqual(err.worker_id, "w-1")

    def test_process_error_tracks_pid_and_worker_id(self):
        err = exceptions.ProcessError("process failed", pid=1001, worker_id="worker-x")
        self.assertIsInstance(err, exceptions.DistributedError)
        self.assertEqual(err.pid, 1001)
        self.assertEqual(err.worker_id, "worker-x")

    def test_specialized_errors_inherit_expected_base_types(self):
        self.assertTrue(issubclass(exceptions.WorkerTimeoutError, exceptions.WorkerError))
        self.assertTrue(issubclass(exceptions.WorkerNotAvailableError, exceptions.WorkerError))
        self.assertTrue(issubclass(exceptions.JobQueueError, exceptions.DistributedError))
        self.assertTrue(issubclass(exceptions.TileCollectionError, exceptions.DistributedError))
        self.assertTrue(issubclass(exceptions.TunnelError, exceptions.DistributedError))


if __name__ == "__main__":
    unittest.main()
