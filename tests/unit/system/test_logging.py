import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from utils import logging as logging_utils


class LoggingUtilsTests(unittest.TestCase):
    def test_is_debug_enabled_delegates_to_config(self):
        with patch.object(logging_utils, "get_debug_enabled", return_value=True) as mocked:
            self.assertTrue(logging_utils.is_debug_enabled())
        mocked.assert_called_once_with(default=False)

    def test_is_debug_enabled_handles_config_errors(self):
        with patch.object(logging_utils, "get_debug_enabled", side_effect=RuntimeError("boom")):
            self.assertFalse(logging_utils.is_debug_enabled())

    def test_debug_log_emits_only_when_enabled(self):
        output = io.StringIO()
        with patch.object(logging_utils, "is_debug_enabled", return_value=True), redirect_stdout(output):
            logging_utils.debug_log("hello")
        self.assertIn("[Distributed] hello", output.getvalue())

        output = io.StringIO()
        with patch.object(logging_utils, "is_debug_enabled", return_value=False), redirect_stdout(output):
            logging_utils.debug_log("hidden")
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
