import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path


def _load_worker_monitor_module():
    module_path = Path(__file__).resolve().parents[3] / "workers" / "worker_monitor.py"
    spec = importlib.util.spec_from_file_location("dist_test_worker_monitor", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


worker_monitor = _load_worker_monitor_module()


class _FakeWorkerProcess:
    def __init__(self, pid=9001, poll_sequence=None):
        self.pid = pid
        self._poll_sequence = list(poll_sequence or [None])
        self._poll_calls = 0
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self):
        if self._poll_calls < len(self._poll_sequence):
            value = self._poll_sequence[self._poll_calls]
        else:
            value = self._poll_sequence[-1]
        self._poll_calls += 1
        if value is not None:
            self.returncode = value
        return value

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout=None):
        _ = timeout
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class WorkerMonitorTests(unittest.TestCase):
    def setUp(self):
        self._orig_launch = worker_monitor.launch_process_with_timeout
        self._orig_is_alive = worker_monitor.is_process_alive
        self._orig_terminate = getattr(worker_monitor, "terminate_process", None)
        self._orig_sleep = worker_monitor.time.sleep
        self._orig_signal = worker_monitor.signal.signal
        self._env_backup = dict(os.environ)

        worker_monitor.time.sleep = lambda _seconds: None
        worker_monitor.signal.signal = lambda *_args, **_kwargs: None

    def tearDown(self):
        worker_monitor.launch_process_with_timeout = self._orig_launch
        worker_monitor.is_process_alive = self._orig_is_alive
        if self._orig_terminate is not None:
            worker_monitor.terminate_process = self._orig_terminate
        worker_monitor.time.sleep = self._orig_sleep
        worker_monitor.signal.signal = self._orig_signal
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_main_validates_required_env_and_args(self):
        os.environ.pop("COMFYUI_MASTER_PID", None)
        self.assertEqual(worker_monitor.main(["python", "main.py"]), 1)

        os.environ["COMFYUI_MASTER_PID"] = "not-an-int"
        self.assertEqual(worker_monitor.main(["python", "main.py"]), 1)

        os.environ["COMFYUI_MASTER_PID"] = "101"
        self.assertEqual(worker_monitor.main([]), 1)

    def test_main_delegates_to_monitor_and_run(self):
        calls = []

        def _monitor(master_pid, command):
            calls.append((master_pid, list(command)))
            return 7

        os.environ["COMFYUI_MASTER_PID"] = "202"
        original_monitor = worker_monitor.monitor_and_run
        worker_monitor.monitor_and_run = _monitor
        try:
            exit_code = worker_monitor.main(["python", "worker.py"])
        finally:
            worker_monitor.monitor_and_run = original_monitor

        self.assertEqual(exit_code, 7)
        self.assertEqual(calls, [(202, ["python", "worker.py"])])

    def test_monitor_and_run_returns_worker_exit_code(self):
        fake_process = _FakeWorkerProcess(poll_sequence=[None, 5])
        worker_monitor.launch_process_with_timeout = lambda *_args, **_kwargs: fake_process
        worker_monitor.is_process_alive = lambda _pid: True
        worker_monitor.terminate_process = lambda *_args, **_kwargs: None

        exit_code = worker_monitor.monitor_and_run(100, ["python", "worker.py"])

        self.assertEqual(exit_code, 5)

    def test_monitor_and_run_terminates_worker_when_master_dies(self):
        fake_process = _FakeWorkerProcess(poll_sequence=[None, None])
        term_calls = []

        worker_monitor.launch_process_with_timeout = lambda *_args, **_kwargs: fake_process
        worker_monitor.is_process_alive = lambda _pid: False

        def _terminate(process, timeout):
            term_calls.append((process.pid, timeout))
            process.returncode = 0

        worker_monitor.terminate_process = _terminate

        exit_code = worker_monitor.monitor_and_run(333, ["python", "worker.py"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(term_calls, [(9001, worker_monitor.PROCESS_TERMINATION_TIMEOUT)])

    def test_monitor_and_run_writes_pid_file_when_requested(self):
        fake_process = _FakeWorkerProcess(poll_sequence=[0])
        worker_monitor.launch_process_with_timeout = lambda *_args, **_kwargs: fake_process
        worker_monitor.is_process_alive = lambda _pid: True
        worker_monitor.terminate_process = lambda *_args, **_kwargs: None

        with tempfile.TemporaryDirectory() as tmpdir:
            pid_file = Path(tmpdir) / "worker.pid"
            os.environ["WORKER_PID_FILE"] = str(pid_file)
            exit_code = worker_monitor.monitor_and_run(444, ["python", "worker.py"])

            self.assertEqual(exit_code, 0)
            contents = pid_file.read_text(encoding="utf-8")
            monitor_pid, worker_pid = contents.split(",")
            self.assertTrue(monitor_pid.isdigit())
            self.assertEqual(worker_pid, "9001")


if __name__ == "__main__":
    unittest.main()
