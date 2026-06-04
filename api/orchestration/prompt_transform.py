import json
from collections import deque

from ...utils.logging import debug_log


class PromptIndex:
    """Cache prompt metadata for faster worker/master prompt preparation."""

    def __init__(self, prompt_obj):
        self._prompt_json = json.dumps(prompt_obj)
        self.nodes_by_class = {}
        self.class_by_node = {}
        self.inputs_by_node = {}
        for node_id, node in _iter_prompt_nodes(prompt_obj):
            class_type = node.get("class_type")
            node_id_str = str(node_id)
            if class_type:
                self.nodes_by_class.setdefault(class_type, []).append(node_id_str)
            self.class_by_node[node_id_str] = class_type
            self.inputs_by_node[node_id_str] = node.get("inputs", {})
        self._upstream_cache = {}

    def copy_prompt(self):
        return json.loads(self._prompt_json)

    def nodes_for_class(self, class_name):
        return self.nodes_by_class.get(class_name, [])

    def has_upstream(self, start_node_id, target_class):
        cache_key = (str(start_node_id), target_class)
        if cache_key in self._upstream_cache:
            return self._upstream_cache[cache_key]

        visited = set()
        stack = [str(start_node_id)]
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            inputs = self.inputs_by_node.get(node_id, {})
            for value in inputs.values():
                if isinstance(value, list) and len(value) == 2:
                    upstream_id = str(value[0])
                    if self.class_by_node.get(upstream_id) == target_class:
                        self._upstream_cache[cache_key] = True
                        return True
                    if upstream_id in self.inputs_by_node:
                        stack.append(upstream_id)

        self._upstream_cache[cache_key] = False
        return False


def _iter_prompt_nodes(prompt_obj):
    for node_id, node in prompt_obj.items():
        if isinstance(node, dict):
            yield str(node_id), node


def find_nodes_by_class(prompt_obj, class_name):
    nodes = []
    for node_id, node in _iter_prompt_nodes(prompt_obj):
        if node.get("class_type") == class_name:
            nodes.append(node_id)
    return nodes


def _find_downstream_nodes(prompt_obj, start_ids):
    """Return all nodes reachable downstream from the provided IDs."""
    adjacency = {}
    for node_id, node in _iter_prompt_nodes(prompt_obj):
        inputs = node.get("inputs", {})
        for value in inputs.values():
            if isinstance(value, list) and len(value) == 2:
                source_id = str(value[0])
                adjacency.setdefault(source_id, set()).add(str(node_id))

    connected = set(start_ids)
    queue = deque(start_ids)
    while queue:
        current = queue.popleft()
        for dependent in adjacency.get(current, ()):  # pragma: no branch - simple iteration
            if dependent not in connected:
                connected.add(dependent)
                queue.append(dependent)
    return connected


def _create_numeric_id_generator(prompt_obj):
    """Return a closure that yields new numeric string IDs."""
    max_id = 0
    for node_id in prompt_obj.keys():
        try:
            numeric = int(node_id)
        except (TypeError, ValueError):
            continue
        max_id = max(max_id, numeric)

    counter = max_id

    def _next_id():
        nonlocal counter
        counter += 1
        return str(counter)

    return _next_id


def _find_upstream_nodes(prompt_obj, start_ids):
    """Return all nodes reachable upstream from start_ids, including start nodes."""
    connected = set(str(node_id) for node_id in start_ids)
    queue = deque(connected)
    while queue:
        node_id = queue.popleft()
        node = prompt_obj.get(node_id) or {}
        inputs = node.get("inputs", {})
        for value in inputs.values():
            if isinstance(value, list) and len(value) == 2:
                source_id = str(value[0])
                if source_id in prompt_obj and source_id not in connected:
                    connected.add(source_id)
                    queue.append(source_id)
    return connected


_DELEGATE_MASTER_RETAINED_UPSTREAM_CLASSES = {
    "PrimitiveBoolean",
    "PrimitiveFloat",
    "PrimitiveInt",
    "PrimitiveNode",
    "PrimitiveString",
}

_DELEGATE_MASTER_ALWAYS_RETAINED_UPSTREAM_CLASSES = {
    "LoadImage",
}

_DELEGATE_MASTER_SAFE_SCALAR_TYPES = {"BOOLEAN", "FLOAT", "INT", "STRING"}
_DELEGATE_MASTER_SAFE_LIST_TYPES = {"LIST"}

# ComfyUI 0.23 exposes CreateList via the newer schema API rather than the
# legacy RETURN_TYPES/INPUT_TYPES attributes. Treat it as a safe config utility
# only after its connected inputs recursively prove safe.
_DELEGATE_MASTER_SAFE_DYNAMIC_CONFIG_OUTPUT_CLASSES = {"CreateList"}
_DELEGATE_MASTER_SAFE_DYNAMIC_CONFIG_INPUT_PREFIXES = {
    "CreateList": ("inputs.",),
}

# Test hook. At runtime this stays None and the ComfyUI node registry is loaded lazily.
_DELEGATE_MASTER_NODE_CLASS_MAPPINGS = None


def _get_delegate_master_node_class_mappings():
    """Return ComfyUI node-class mappings when available."""
    if _DELEGATE_MASTER_NODE_CLASS_MAPPINGS is not None:
        return _DELEGATE_MASTER_NODE_CLASS_MAPPINGS
    try:
        import nodes as comfy_nodes  # type: ignore
    except Exception:  # pragma: no cover - depends on ComfyUI runtime imports
        return {}
    return getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}


def _get_delegate_master_node_class(class_type):
    mappings = _get_delegate_master_node_class_mappings()
    return mappings.get(class_type) if isinstance(mappings, dict) else None


def _normalize_delegate_master_return_type(return_type):
    if return_type is None:
        return ""
    return str(return_type).strip().upper()


def _delegate_master_type_is_safe_scalar(type_name):
    return type_name in _DELEGATE_MASTER_SAFE_SCALAR_TYPES


def _delegate_master_type_is_safe_config(type_name):
    return _delegate_master_type_is_safe_scalar(type_name) or type_name in _DELEGATE_MASTER_SAFE_LIST_TYPES


def _delegate_master_output_is_safe_scalar(class_type, output_index):
    """Return True when a registered node output is lightweight config data."""
    if class_type in _DELEGATE_MASTER_SAFE_DYNAMIC_CONFIG_OUTPUT_CLASSES:
        return True
    node_class = _get_delegate_master_node_class(class_type)
    return_types = getattr(node_class, "RETURN_TYPES", ()) if node_class is not None else ()
    try:
        output_type = return_types[int(output_index)]
    except (IndexError, TypeError, ValueError):
        return False
    return _delegate_master_type_is_safe_config(_normalize_delegate_master_return_type(output_type))


def _get_delegate_master_input_types(class_type):
    node_class = _get_delegate_master_node_class(class_type)
    input_types = getattr(node_class, "INPUT_TYPES", None) if node_class is not None else None
    if callable(input_types):
        try:
            input_types = input_types()
        except TypeError:
            return {}
    return input_types if isinstance(input_types, dict) else {}


def _normalize_delegate_master_input_type(input_spec):
    if isinstance(input_spec, (list, tuple)) and input_spec:
        return _normalize_delegate_master_return_type(input_spec[0])
    return _normalize_delegate_master_return_type(input_spec)


def _delegate_master_input_is_safe_scalar(class_type, input_name):
    """Return True when a registered downstream input expects config data."""
    for prefix in _DELEGATE_MASTER_SAFE_DYNAMIC_CONFIG_INPUT_PREFIXES.get(class_type, ()):
        if input_name.startswith(prefix):
            return True
    input_types = _get_delegate_master_input_types(class_type)
    for section_name in ("required", "optional"):
        section = input_types.get(section_name, {})
        if isinstance(section, dict) and input_name in section:
            input_type = _normalize_delegate_master_input_type(section[input_name])
            return _delegate_master_type_is_safe_config(input_type)
    return False


def _is_delegate_master_always_retained_upstream_node(node):
    if not isinstance(node, dict):
        return False
    class_type = node.get("class_type")
    return isinstance(class_type, str) and class_type in _DELEGATE_MASTER_ALWAYS_RETAINED_UPSTREAM_CLASSES


def _is_delegate_master_retained_upstream_node(node, output_index=0):
    """Return True for lightweight upstream nodes safe to keep on the master."""
    if not isinstance(node, dict):
        return False
    class_type = node.get("class_type")
    if not isinstance(class_type, str):
        return False
    return (
        class_type in _DELEGATE_MASTER_RETAINED_UPSTREAM_CLASSES
        or class_type.startswith("Primitive")
        or _delegate_master_output_is_safe_scalar(class_type, output_index)
    )


def _collect_delegate_master_retained_upstream_branch(
    prompt_obj,
    node_id,
    output_index,
    memo,
    visiting,
):
    """Return safe retained branch nodes, or None when the branch is not safe."""
    node_id = str(node_id)
    cache_key = (node_id, output_index)
    if cache_key in memo:
        cached = memo[cache_key]
        return None if cached is None else set(cached)
    if cache_key in visiting:
        memo[cache_key] = None
        return None

    node = prompt_obj.get(node_id)
    if not _is_delegate_master_retained_upstream_node(node, output_index):
        memo[cache_key] = None
        return None

    visiting.add(cache_key)
    retained = {node_id}
    inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
    class_type = node.get("class_type") if isinstance(node, dict) else None
    for input_name, value in inputs.items():
        if not (isinstance(value, list) and len(value) == 2):
            continue
        if not _delegate_master_input_is_safe_scalar(class_type, input_name):
            visiting.remove(cache_key)
            memo[cache_key] = None
            return None
        source_id = str(value[0])
        branch = _collect_delegate_master_retained_upstream_branch(
            prompt_obj,
            source_id,
            value[1],
            memo,
            visiting,
        )
        if branch is None:
            visiting.remove(cache_key)
            memo[cache_key] = None
            return None
        retained.update(branch)

    visiting.remove(cache_key)
    memo[cache_key] = frozenset(retained)
    return retained


def _find_delegate_master_retained_upstream_nodes(prompt_obj, start_ids):
    """Return lightweight upstream nodes needed by kept delegate-master nodes."""
    connected = set()
    memo = {}
    for node_id in start_ids:
        node = prompt_obj.get(str(node_id)) or {}
        inputs = node.get("inputs", {})
        class_type = node.get("class_type") if isinstance(node, dict) else None
        for input_name, value in inputs.items():
            if not (isinstance(value, list) and len(value) == 2):
                continue
            source_node = prompt_obj.get(str(value[0]))
            if _is_delegate_master_always_retained_upstream_node(source_node):
                connected.add(str(value[0]))
                continue
            if not _delegate_master_input_is_safe_scalar(class_type, input_name):
                continue
            branch = _collect_delegate_master_retained_upstream_branch(
                prompt_obj,
                value[0],
                value[1],
                memo,
                set(),
            )
            if branch is not None:
                connected.update(branch)
    return connected


def prune_prompt_for_worker(prompt_obj):
    """Prune worker prompt to distributed nodes and their upstream dependencies."""
    collector_ids = find_nodes_by_class(prompt_obj, "DistributedCollector")
    upscale_ids = find_nodes_by_class(prompt_obj, "UltimateSDUpscaleDistributed")
    distributed_ids = collector_ids + upscale_ids
    if not distributed_ids:
        return prompt_obj

    connected = _find_upstream_nodes(prompt_obj, distributed_ids)
    pruned_prompt = {}
    for node_id in connected:
        node = prompt_obj.get(node_id)
        if node is not None:
            pruned_prompt[node_id] = json.loads(json.dumps(node))

    # Generate IDs from the original prompt so we never reuse IDs from pruned downstream nodes.
    next_id = _create_numeric_id_generator(prompt_obj)
    for dist_id in distributed_ids:
        if dist_id not in pruned_prompt:
            continue
        downstream = _find_downstream_nodes(prompt_obj, [dist_id])
        has_removed_downstream = any(node_id != dist_id for node_id in downstream)
        if has_removed_downstream:
            preview_id = next_id()
            pruned_prompt[preview_id] = {
                "inputs": {
                    "images": [dist_id, 0],
                },
                "class_type": "PreviewImage",
                "_meta": {
                    "title": "Preview Image (auto-added)",
                },
            }

    return pruned_prompt


def prepare_delegate_master_prompt(prompt_obj, collector_ids):
    """Prune master prompt so it only executes post-collector nodes in delegate mode."""
    downstream = _find_downstream_nodes(prompt_obj, collector_ids)
    nodes_to_keep = set(collector_ids)
    nodes_to_keep.update(downstream)
    nodes_to_keep.update(
        _find_delegate_master_retained_upstream_nodes(prompt_obj, nodes_to_keep)
    )

    pruned_prompt = {}
    for node_id in nodes_to_keep:
        node = prompt_obj.get(node_id)
        if node is not None:
            pruned_prompt[node_id] = json.loads(json.dumps(node))

    pruned_ids = set(pruned_prompt.keys())
    for node_id, node in pruned_prompt.items():
        inputs = node.get("inputs")
        if not inputs:
            continue
        for input_name, input_value in list(inputs.items()):
            if isinstance(input_value, list) and len(input_value) == 2:
                source_id = str(input_value[0])
                if source_id not in pruned_ids:
                    inputs.pop(input_name, None)
                    debug_log(
                        f"Removed upstream reference '{input_name}' from node {node_id} for delegate-only master prompt."
                    )

    # Generate IDs from the original prompt to avoid ID collisions with pruned nodes.
    next_id = _create_numeric_id_generator(prompt_obj)
    for collector_id in collector_ids:
        collector_entry = pruned_prompt.get(collector_id)
        if not collector_entry:
            continue
        placeholder_id = next_id()
        pruned_prompt[placeholder_id] = {
            "class_type": "DistributedEmptyImage",
            "inputs": {
                "height": 64,
                "width": 64,
                "channels": 3,
            },
            "_meta": {
                "title": "Distributed Empty Image (auto-added)",
            },
        }
        collector_entry.setdefault("inputs", {})["images"] = [placeholder_id, 0]
        debug_log(
            f"Inserted placeholder node {placeholder_id} for collector {collector_id} in delegate-only master prompt."
        )

    return pruned_prompt


def generate_job_id_map(prompt_index, prefix):
    """Create stable per-node job IDs for distributed nodes."""
    job_map = {}
    distributed_nodes = prompt_index.nodes_for_class("DistributedCollector") + prompt_index.nodes_for_class(
        "UltimateSDUpscaleDistributed"
    )
    for node_id in distributed_nodes:
        job_map[node_id] = f"{prefix}_{node_id}"
    return job_map


def _override_seed_nodes(prompt_copy, prompt_index, is_master, participant_id, worker_index_map):
    """Configure DistributedSeed nodes for master or worker role."""
    for node_id in prompt_index.nodes_for_class("DistributedSeed"):
        node = prompt_copy.get(node_id)
        if not isinstance(node, dict):
            continue
        inputs = node.setdefault("inputs", {})
        inputs["is_worker"] = not is_master
        if is_master:
            inputs["worker_id"] = ""
        else:
            inputs["worker_id"] = f"worker_{worker_index_map.get(participant_id, 0)}"


def _override_collector_nodes(
    prompt_copy,
    prompt_index,
    is_master,
    participant_id,
    job_id_map,
    master_url,
    enabled_json,
    delegate_master,
):
    """Configure DistributedCollector nodes for master or worker role."""
    for node_id in prompt_index.nodes_for_class("DistributedCollector"):
        node = prompt_copy.get(node_id)
        if not isinstance(node, dict):
            continue

        if prompt_index.has_upstream(node_id, "UltimateSDUpscaleDistributed"):
            node.setdefault("inputs", {})["pass_through"] = True
            continue

        inputs = node.setdefault("inputs", {})
        inputs["multi_job_id"] = job_id_map.get(node_id, node_id)
        inputs["is_worker"] = not is_master
        inputs["enabled_worker_ids"] = enabled_json
        if is_master:
            inputs["delegate_only"] = bool(delegate_master)
            inputs.pop("master_url", None)
            inputs.pop("worker_id", None)
        else:
            inputs["master_url"] = master_url
            inputs["worker_id"] = participant_id
            inputs["delegate_only"] = False


def _override_upscale_nodes(
    prompt_copy,
    prompt_index,
    is_master,
    participant_id,
    job_id_map,
    master_url,
    enabled_json,
):
    """Configure UltimateSDUpscaleDistributed nodes for master or worker role."""
    for node_id in prompt_index.nodes_for_class("UltimateSDUpscaleDistributed"):
        node = prompt_copy.get(node_id)
        if not isinstance(node, dict):
            continue
        inputs = node.setdefault("inputs", {})
        inputs["multi_job_id"] = job_id_map.get(node_id, node_id)
        inputs["is_worker"] = not is_master
        inputs["enabled_worker_ids"] = enabled_json
        if is_master:
            inputs.pop("master_url", None)
            inputs.pop("worker_id", None)
        else:
            inputs["master_url"] = master_url
            inputs["worker_id"] = participant_id


def _override_value_nodes(prompt_copy, prompt_index, is_master, participant_id, worker_index_map):
    """Configure DistributedValue nodes for master or worker role."""
    for node_id in prompt_index.nodes_for_class("DistributedValue"):
        node = prompt_copy.get(node_id)
        if not isinstance(node, dict):
            continue
        inputs = node.setdefault("inputs", {})
        inputs["is_worker"] = not is_master
        if is_master:
            inputs["worker_id"] = ""
        else:
            inputs["worker_id"] = f"worker_{worker_index_map.get(participant_id, 0)}"


def apply_participant_overrides(
    prompt_copy,
    participant_id,
    enabled_worker_ids,
    job_id_map,
    master_url,
    delegate_master,
    prompt_index,
):
    """Return a prompt copy with hidden inputs configured for master/worker."""
    is_master = participant_id == "master"
    worker_index_map = {wid: idx for idx, wid in enumerate(enabled_worker_ids)}
    enabled_json = json.dumps(enabled_worker_ids)

    _override_seed_nodes(prompt_copy, prompt_index, is_master, participant_id, worker_index_map)
    _override_value_nodes(prompt_copy, prompt_index, is_master, participant_id, worker_index_map)
    _override_collector_nodes(
        prompt_copy,
        prompt_index,
        is_master,
        participant_id,
        job_id_map,
        master_url,
        enabled_json,
        delegate_master,
    )
    _override_upscale_nodes(
        prompt_copy,
        prompt_index,
        is_master,
        participant_id,
        job_id_map,
        master_url,
        enabled_json,
    )

    return prompt_copy
