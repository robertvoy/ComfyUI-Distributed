# Native V3 node API

The package registers one `ComfyExtension` through `comfy_entrypoint()` and
imports the versioned `comfy_api.v0_0_2` API. Use a ComfyUI version that provides
that API; there is no V1 registration fallback.

## Compatibility

- All eight node IDs, display names, categories, visible input order, defaults
  and output order are retained. Existing execution algorithms remain in the
  private collector, utilities and upscale modules.
- The image/audio dividers explicitly declare ten typed outputs. Their existing
  frontend extensions still show only the selected number of outputs. This
  replaces the V1 `ByPassTypeTuple` indexing workaround without changing saved
  IMAGE/AUDIO socket indices or the ten returned values.
- Standard hidden context uses `cls.hidden`. Worker/orchestration metadata keeps
  its existing prompt-input names and Python defaults via `accept_all_inputs`;
  it does not become visible widgets.
- The collector retains list-input handling. Upscale retains its always-changing
  fingerprint and creates a private runtime helper for each V3 execution, so
  mutable helper state is not attached to sanitized V3 class clones.
- Routes, distributed state and the existing worker startup/shutdown hooks are
  initialized by `runtime/bootstrap.py` during `ComfyExtension.on_load()`.
  Worker mode still suppresses automatic worker launch. `distributed.py` is no
  longer an entrypoint.

## Verification

Run ordinary unit tests from this repository:

```bash
python -m pytest tests -q -o addopts=
```

Opt into real-framework acceptance using a ComfyUI checkout and its interpreter:

```bash
COMFYUI_SOURCE_ROOT=/path/to/ComfyUI \
  /path/to/ComfyUI/.venv/bin/python -m pytest tests -q -o addopts=
```

The acceptance subprocess uses CPU mode and the real ComfyUI loader, input
parser, V3 class preparation, prompt validation and `PromptExecutor`. It compares
all node schemas with the V1 fixture, checks all five bundled workflows' node
IDs and socket/link contracts, exercises injected worker metadata, image/audio
lists, collector aggregation and divider outputs, rejects invalid upscale enums,
and decodes an actual preview PNG referenced by executor history.

The upscale GPU/model boundary is mocked to check argument forwarding and
per-execution helper isolation. Full checkpoint inference, browser canvas
acceptance and multi-host HTTP transport are not exercised. No HTTP listener,
workers or model downloads are started; preview files use a temporary scratch
directory. The test does not install this branch into a live custom-node folder.
