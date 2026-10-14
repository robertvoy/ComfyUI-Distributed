import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_config_module():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "config.py"
    package_name = "image_list_timeout_config_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = [str(module_path.parent)]
    sys.modules[package_name] = root_pkg

    logging_module = types.ModuleType(f"{package_name}.logging")
    logging_module.log = lambda *_args, **_kwargs: None
    logging_module.debug_log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.logging"] = logging_module

    constants_module = types.ModuleType(f"{package_name}.constants")
    constants_module.HEARTBEAT_TIMEOUT = 30
    sys.modules[f"{package_name}.constants"] = constants_module

    spec = importlib.util.spec_from_file_location(f"{package_name}.config", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


config_module = _load_config_module()


SETTING_KEY = "image_list_job_timeout_seconds"


def _config_with_timeout(value):
    return {"settings": {SETTING_KEY: value}}


def test_default_config_sets_image_list_job_timeout_to_zero():
    assert config_module.get_default_config()["settings"][SETTING_KEY] == 0.0


def test_missing_value_defaults_to_disabled_timeout():
    assert config_module.get_image_list_job_timeout_seconds({}) == 0.0
    assert config_module.get_image_list_job_timeout_seconds({"settings": {}}) == 0.0


def test_none_config_loads_current_config(monkeypatch):
    cfg = _config_with_timeout("12.5")
    monkeypatch.setattr(config_module, "load_config", lambda: cfg)

    assert config_module.get_image_list_job_timeout_seconds() == 12.5


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (0.0, 0.0),
        ("0", 0.0),
        (" 0.0 ", 0.0),
        (1, 1.0),
        (2.5, 2.5),
        ("3", 3.0),
        (" 4.25 ", 4.25),
    ],
)
def test_accepts_zero_and_positive_numeric_values(value, expected):
    result = config_module.get_image_list_job_timeout_seconds(_config_with_timeout(value))

    assert result == expected
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        -1,
        -0.01,
        float("nan"),
        float("inf"),
        float("-inf"),
        "not-a-number",
        "",
        "   ",
        "-1",
        "nan",
        "inf",
        None,
        [],
        {},
    ],
)
def test_rejects_invalid_values_with_setting_name(value):
    with pytest.raises(ValueError, match=SETTING_KEY):
        config_module.get_image_list_job_timeout_seconds(_config_with_timeout(value))
