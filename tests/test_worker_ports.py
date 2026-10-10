import importlib.util
import socket
import unittest
from pathlib import Path
from unittest.mock import patch


def load_ports():
    path = Path(__file__).resolve().parents[1] / "workers" / "ports.py"
    spec = importlib.util.spec_from_file_location("worker_ports_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkerPortTests(unittest.TestCase):
    def setUp(self):
        self.ports = load_ports()

    def test_allocation_starts_above_actual_master_and_skips_assigned_and_occupied(self):
        workers = [
            {"id": "existing", "host": "localhost", "port": 8190, "enabled": False},
            {"id": "remote", "host": "remote.example", "port": 8192},
        ]
        with patch.object(self.ports, "is_port_available", side_effect=lambda port: port != 8191):
            self.assertEqual(self.ports.allocate_worker_ports(8189, workers, 3), [8192, 8193, 8194])

    def test_local_host_forms_reserve_ports(self):
        for host in ["::1", "[::1]", "http://localhost/", "HTTPS://LOCALHOST", "0.0.0.0", None]:
            with self.subTest(host=host), patch.object(self.ports, "is_port_available", return_value=True):
                self.assertEqual(self.ports.allocate_worker_ports(8189, [{"id": "local", "host": host, "port": 8190}], 1), [8191])

    def test_exhaustion_is_explicit_not_a_partial_allocation(self):
        with patch.object(self.ports, "is_port_available", return_value=True):
            with self.assertRaisesRegex(ValueError, "available.*ports"):
                self.ports.allocate_worker_ports(65534, [], 2)

    def test_launch_rejects_master_port_without_reassigning_configuration(self):
        worker = {"id": "manual", "port": 8189}
        with self.assertRaisesRegex(ValueError, "master.*8189"):
            self.ports.validate_worker_port(worker, 8189, [])
        self.assertEqual(worker["port"], 8189)

    def test_launch_rejects_another_local_workers_reserved_port(self):
        worker = {"id": "manual", "port": 8190}
        other = {"id": "other", "host": None, "port": 8190, "enabled": False}
        with self.assertRaisesRegex(ValueError, "other"):
            self.ports.validate_worker_port(worker, 8189, [worker, other])

    def test_launch_ignores_itself_and_remote_workers_with_same_port(self):
        worker = {"id": "manual", "port": 8190}
        other = {"id": "remote", "host": "example.com", "port": 8190}
        with patch.object(self.ports, "is_port_available", return_value=True):
            self.ports.validate_worker_port(worker, 8189, [worker, other])

    def test_launch_rejects_an_occupied_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.assertFalse(self.ports.is_port_available(port))
            with self.assertRaisesRegex(ValueError, "already in use"):
                self.ports.validate_worker_port({"id": "manual", "port": port}, 8189, [])

    def test_invalid_ports_fail_clearly(self):
        for port in [0, 65536, "not-a-port", None, True, 8190.5]:
            with self.subTest(port=port), self.assertRaisesRegex(ValueError, "port"):
                self.ports.validate_worker_port({"id": "manual", "port": port}, 8189, [])


if __name__ == "__main__":
    unittest.main()
