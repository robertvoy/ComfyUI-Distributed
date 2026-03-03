import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from utils import logging as logging_utils


class LoggingUtilsTests(unittest.TestCase):
    def setUp(self):
        self.original_config_file = logging_utils.GPU_CONFIG_FILE
        self.state = logging_utils._debug_state()
        self.original_cache = self.state.enabled
        self.original_cache_time = self.state.updated_at

    def tearDown(self):
        logging_utils.GPU_CONFIG_FILE = self.original_config_file
        self.state.enabled = self.original_cache
        self.state.updated_at = self.original_cache_time

    def _reset_cache(self):
        self.state.enabled = None
        self.state.updated_at = 0.0

    def test_is_debug_enabled_reads_config_flag(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
            json.dump({"settings": {"debug": True}}, temp_file)
            temp_path = temp_file.name
        self.addCleanup(lambda: os.path.exists(temp_path) and os.unlink(temp_path))

        logging_utils.GPU_CONFIG_FILE = temp_path
        self._reset_cache()

        self.assertTrue(logging_utils.is_debug_enabled())

    def test_is_debug_enabled_handles_invalid_json(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
            temp_file.write("{invalid-json")
            temp_path = temp_file.name
        self.addCleanup(lambda: os.path.exists(temp_path) and os.unlink(temp_path))

        logging_utils.GPU_CONFIG_FILE = temp_path
        self._reset_cache()

        self.assertFalse(logging_utils.is_debug_enabled())

    def test_debug_log_emits_only_when_enabled(self):
        self.state.enabled = True
        self.state.updated_at = 1e18
        output = io.StringIO()
        with redirect_stdout(output):
            logging_utils.debug_log("hello")
        self.assertIn("[Distributed] hello", output.getvalue())

        self.state.enabled = False
        self.state.updated_at = 1e18
        output = io.StringIO()
        with redirect_stdout(output):
            logging_utils.debug_log("hidden")
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
