import json
from collections import deque
from typing import Any

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

    def has_upstream(self, start_node_id: str, target_class: str) -> bool:
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


def _find_downstream_of_output_slot(prompt_obj, node_id, slot_index):
    """Return all nodes reachable from a specific output slot on one node."""
    source_id = str(node_id)
    try:
        slot = int(slot_index)
    except (TypeError, ValueError):
        return set()

    direct_consumers = set()
    for candidate_id, candidate_node in _iter_prompt_nodes(prompt_obj):
        inputs = candidate_node.get("inputs", {})
        for value in inputs.values():
            if not (isinstance(value, list) and len(value) == 2):
                continue
            if str(value[0]) != source_id:
                continue
            try:
                input_slot = int(value[1])
            except (TypeError, ValueError):
                debug_log(
                    f"Prompt transform: skipping malformed slot reference for node {candidate_id}: {value}"
                )
                continue
            if input_slot == slot:
                direct_consumers.add(str(candidate_id))
                break

    if not direct_consumers:
        return set()
    return _find_downstream_nodes(prompt_obj, list(direct_consumers))


def _create_numeric_id_generator(prompt_obj):
    """Return a closure that yields new numeric string IDs."""
    max_id = 0
    for node_id in prompt_obj.keys():
        try:
            numeric = int(node_id)
        except (TypeError, ValueError):
            debug_log(f"Prompt transform: ignoring non-numeric node id while generating ids: {node_id}")
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


def _resolve_participants(enabled_worker_ids, delegate_master):
    worker_ids = [str(worker_id) for worker_id in (enabled_worker_ids or [])]
    if delegate_master:
        return worker_ids
    return ["master"] + worker_ids


def _remove_dangling_input_refs(prompt_obj):
    """Drop input links that point to nodes no longer present in prompt_obj."""
    existing_ids = set(prompt_obj.keys())
    for _node_id, node in _iter_prompt_nodes(prompt_obj):
        inputs = node.get("inputs", {})
        for input_name, input_value in list(inputs.items()):
            if isinstance(input_value, list) and len(input_value) == 2:
                source_id = str(input_value[0])
                if source_id not in existing_ids:
                    inputs.pop(input_name, None)


def _has_terminal_output_nodes(prompt_obj):
    """Return True when the prompt already has at least one terminal output node."""
    for _node_id, node in _iter_prompt_nodes(prompt_obj):
        if node.get("class_type") in {"PreviewImage", "SaveImage"}:
            return True
    return False


def _add_preview_node(prompt_obj, next_id_fn, source_node_id, slot_index, title_suffix):
    """Attach a PreviewImage output node to the given source output slot."""
    preview_id = next_id_fn()
    prompt_obj[preview_id] = {
        "inputs": {
            "images": [str(source_node_id), int(slot_index)],
        },
        "class_type": "PreviewImage",
        "_meta": {
            "title": f"Preview Image ({title_suffix})",
        },
    }
    return preview_id


def _ensure_worker_output_node(prompt_obj):
    """Guarantee worker prompts retain at least one output node after pruning.

    Branch pruning can remove all explicit output nodes for worker prompts, which
    causes ComfyUI validation to reject the prompt (`prompt_no_outputs`).
    """
    if _has_terminal_output_nodes(prompt_obj):
        return prompt_obj

    next_id = _create_numeric_id_generator(prompt_obj)

    # Prefer a DistributedBranchCollector output tied to the assigned branch for this worker.
    for node_id, node in _iter_prompt_nodes(prompt_obj):
        if node.get("class_type") != "DistributedBranchCollector":
            continue
        inputs = node.get("inputs", {})
        try:
            assigned_branch = int(inputs.get("assigned_branch", -1))
        except (TypeError, ValueError):
            assigned_branch = -1
        if assigned_branch >= 0:
            _add_preview_node(prompt_obj, next_id, node_id, assigned_branch, "auto-added worker output")
            return prompt_obj

    # Otherwise, anchor to the assigned DistributedBranch slot if available.
    for node_id, node in _iter_prompt_nodes(prompt_obj):
        if node.get("class_type") != "DistributedBranch":
            continue
        inputs = node.get("inputs", {})
        try:
            assigned_branch = int(inputs.get("assigned_branch", -1))
        except (TypeError, ValueError):
            assigned_branch = -1
        if assigned_branch >= 0:
            _add_preview_node(prompt_obj, next_id, node_id, assigned_branch, "auto-added worker output")
            return prompt_obj

    # Idle participants (assigned_branch=-1) still need a terminal node to pass
    # validation; keep this cheap by emitting a tiny synthetic image preview.
    empty_id = next_id()
    prompt_obj[empty_id] = {
        "class_type": "DistributedEmptyImage",
        "inputs": {
            "height": 64,
            "width": 64,
            "channels": 3,
        },
        "_meta": {
            "title": "Distributed Empty Image (auto-added worker output)",
        },
    }
    _add_preview_node(prompt_obj, next_id, empty_id, 0, "auto-added worker output")
    return prompt_obj


def _prune_worker_downstream_of_branch_collectors(prompt_obj):
    """Drop worker-side nodes downstream of branch collectors."""
    collector_ids = find_nodes_by_class(prompt_obj, "DistributedBranchCollector")
    if not collector_ids:
        return prompt_obj

    downstream = _find_downstream_nodes(prompt_obj, collector_ids)
    for node_id in downstream:
        if node_id in collector_ids:
            continue
        prompt_obj.pop(node_id, None)

    _remove_dangling_input_refs(prompt_obj)
    return prompt_obj


def prune_prompt_for_branch_worker(
    prompt_obj: dict[str, Any],
    branch_node_id: str,
    assigned_branch: int | list[int] | tuple[int, ...] | set[int],
    num_branches: int,
) -> dict[str, Any]:
    """Prune non-assigned branch paths while keeping shared downstream nodes."""
    branch_id = str(branch_node_id)

    assigned_slots = set()
    if isinstance(assigned_branch, (list, tuple, set)):
        for value in assigned_branch:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                debug_log(f"Prompt transform: invalid assigned branch slot ignored: {value}")
                continue
            if idx >= 0:
                assigned_slots.add(idx)
    else:
        try:
            idx = int(assigned_branch)
        except (TypeError, ValueError):
            idx = -1
        if idx >= 0:
            assigned_slots.add(idx)

    try:
        total_slots = int(num_branches)
    except (TypeError, ValueError):
        total_slots = 2
    total_slots = max(2, min(total_slots, 10))

    downstream_by_slot = {}
    for slot_idx in range(total_slots):
        downstream = _find_downstream_of_output_slot(prompt_obj, branch_id, slot_idx)
        downstream.discard(branch_id)
        downstream_by_slot[slot_idx] = downstream

    keep = {branch_id}
    for slot_idx in assigned_slots:
        keep.update(downstream_by_slot.get(slot_idx, set()))

    remove = set()
    for slot_idx in range(total_slots):
        if slot_idx in assigned_slots:
            continue
        remove.update(downstream_by_slot.get(slot_idx, set()))

    remove -= keep
    for node_id in remove:
        prompt_obj.pop(node_id, None)

    _remove_dangling_input_refs(prompt_obj)
    return prompt_obj


def prune_prompt_for_worker(prompt_obj: dict[str, Any]) -> dict[str, Any]:
    """Prune worker prompt to distributed nodes and their upstream dependencies."""
    collector_ids = find_nodes_by_class(prompt_obj, "DistributedCollector")
    list_collector_ids = find_nodes_by_class(prompt_obj, "DistributedListCollector")
    branch_collector_ids = find_nodes_by_class(prompt_obj, "DistributedBranchCollector")
    upscale_ids = find_nodes_by_class(prompt_obj, "UltimateSDUpscaleDistributed")
    branch_ids = find_nodes_by_class(prompt_obj, "DistributedBranch")
    distributed_ids = collector_ids + list_collector_ids + branch_collector_ids + upscale_ids + branch_ids
    if not distributed_ids:
        return prompt_obj

    connected = _find_upstream_nodes(prompt_obj, distributed_ids)
    if branch_ids:
        connected.update(_find_downstream_nodes(prompt_obj, branch_ids))
    if branch_collector_ids:
        downstream_of_collectors = _find_downstream_nodes(prompt_obj, branch_collector_ids)
        connected -= (downstream_of_collectors - set(branch_collector_ids))

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
        has_removed_downstream = any(node_id != dist_id and node_id not in connected for node_id in downstream)
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


def prepare_delegate_master_prompt(
    prompt_obj: dict[str, Any],
    collector_ids: list[str],
) -> dict[str, Any]:
    """Prune master prompt so it only executes post-collector nodes in delegate mode."""
    downstream = _find_downstream_nodes(prompt_obj, collector_ids)
    nodes_to_keep = set(collector_ids)
    nodes_to_keep.update(downstream)

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
        collector_class = collector_entry.get("class_type")
        if collector_class == "DistributedBranchCollector":
            # Branch collector does not require an image input in delegate-only mode.
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


def generate_job_id_map(prompt_index: PromptIndex, prefix: str) -> dict[str, str]:
    """Create stable per-node job IDs for distributed nodes."""
    job_map = {}
    distributed_nodes = (
        prompt_index.nodes_for_class("DistributedCollector")
        + prompt_index.nodes_for_class("DistributedListSplitter")
        + prompt_index.nodes_for_class("DistributedListCollector")
        + prompt_index.nodes_for_class("DistributedBranchCollector")
        + prompt_index.nodes_for_class("UltimateSDUpscaleDistributed")
        + prompt_index.nodes_for_class("DistributedBranch")
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


def _override_list_splitter_nodes(
    prompt_copy,
    prompt_index,
    is_master,
    participant_id,
    enabled_worker_ids,
    job_id_map,
    master_url,
    delegate_master,
):
    """Configure participant chunk mapping for DistributedListSplitter nodes."""
    participants = _resolve_participants(enabled_worker_ids, delegate_master)
    total_participants = max(1, len(participants))
    participant_id = str(participant_id)

    if participant_id in participants:
        participant_index = participants.index(participant_id)
    else:
        participant_index = 0

    for node_id in prompt_index.nodes_for_class("DistributedListSplitter"):
        node = prompt_copy.get(node_id)
        if not isinstance(node, dict):
            continue
        inputs = node.setdefault("inputs", {})
        inputs["participant_index"] = participant_index
        inputs["total_participants"] = total_participants
        inputs["multi_job_id"] = job_id_map.get(node_id, node_id)
        inputs["is_worker"] = not is_master
        if is_master:
            inputs.pop("master_url", None)
            inputs["worker_id"] = "master"
        else:
            inputs["master_url"] = master_url
            inputs["worker_id"] = participant_id


def _override_list_collector_nodes(
    prompt_copy,
    prompt_index,
    is_master,
    participant_id,
    job_id_map,
    master_url,
    enabled_json,
    delegate_master,
):
    """Configure DistributedListCollector nodes for master/worker role."""
    for node_id in prompt_index.nodes_for_class("DistributedListCollector"):
        node = prompt_copy.get(node_id)
        if not isinstance(node, dict):
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
            inputs["worker_id"] = str(participant_id)
            inputs["delegate_only"] = False


def _override_branch_nodes(
    prompt_copy,
    prompt_index,
    is_master,
    participant_id,
    enabled_worker_ids,
    job_id_map,
    delegate_master,
):
    """Assign branch slots per participant and prune non-assigned branch paths."""
    participants = _resolve_participants(enabled_worker_ids, delegate_master)
    participant_id = str(participant_id)
    participant_count = len(participants)
    participant_pos = participants.index(participant_id) if participant_id in participants else None

    for node_id in prompt_index.nodes_for_class("DistributedBranch"):
        node = prompt_copy.get(node_id)
        if not isinstance(node, dict):
            continue

        inputs = node.setdefault("inputs", {})
        try:
            num_branches = int(inputs.get("num_branches", 2))
        except (TypeError, ValueError):
            num_branches = 2
        num_branches = max(2, min(num_branches, 10))

        assigned_slots = []
        if participant_pos is not None and participant_count > 0:
            assigned_slots = [slot for slot in range(num_branches) if slot % participant_count == participant_pos]

        if len(assigned_slots) == 1:
            assigned_branch = assigned_slots[0]
        else:
            assigned_branch = -1

        inputs["is_worker"] = not is_master
        inputs["worker_id"] = "" if is_master else participant_id
        inputs["assigned_branch"] = assigned_branch
        inputs["multi_job_id"] = job_id_map.get(node_id, node_id)

        if is_master and delegate_master:
            continue

        prompt_copy = prune_prompt_for_branch_worker(
            prompt_copy,
            node_id,
            assigned_slots,
            num_branches,
        )

    return prompt_copy


def _find_upstream_branch_node(prompt_copy, start_node_id):
    """Locate the nearest upstream DistributedBranch node for the provided node."""
    visited = set()
    stack = [str(start_node_id)]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = prompt_copy.get(node_id)
        if not isinstance(node, dict):
            continue

        inputs = node.get("inputs", {})
        for input_value in inputs.values():
            if not (isinstance(input_value, list) and len(input_value) == 2):
                continue
            source_id = str(input_value[0])
            source_node = prompt_copy.get(source_id)
            if not isinstance(source_node, dict):
                continue
            if source_node.get("class_type") == "DistributedBranch":
                return source_id
            stack.append(source_id)
    return None


def _override_branch_collector_nodes(
    prompt_copy,
    prompt_index,
    is_master,
    participant_id,
    job_id_map,
    master_url,
    enabled_json,
    delegate_master,
):
    """Configure DistributedBranchCollector nodes for branch convergence."""
    for node_id in prompt_index.nodes_for_class("DistributedBranchCollector"):
        node = prompt_copy.get(node_id)
        if not isinstance(node, dict):
            continue

        upstream_branch_id = _find_upstream_branch_node(prompt_copy, node_id)
        assigned_branch = -1
        if upstream_branch_id is not None:
            branch_node = prompt_copy.get(upstream_branch_id, {})
            branch_inputs = branch_node.get("inputs", {}) if isinstance(branch_node, dict) else {}
            try:
                assigned_branch = int(branch_inputs.get("assigned_branch", -1))
            except (TypeError, ValueError):
                assigned_branch = -1

        inputs = node.setdefault("inputs", {})
        # Group all branch collectors under the same DistributedBranch into one queue.
        job_source_id = upstream_branch_id or node_id
        inputs["multi_job_id"] = job_id_map.get(job_source_id, job_source_id)
        inputs["is_worker"] = not is_master
        inputs["enabled_worker_ids"] = enabled_json
        inputs["assigned_branch"] = assigned_branch
        if is_master:
            inputs["delegate_only"] = bool(delegate_master)
            inputs.pop("master_url", None)
            inputs.pop("worker_id", None)
        else:
            inputs["master_url"] = master_url
            inputs["worker_id"] = str(participant_id)
            inputs["delegate_only"] = False


def apply_participant_overrides(
    prompt_copy: dict[str, Any],
    participant_id: str,
    enabled_worker_ids: list[str],
    job_id_map: dict[str, str],
    master_url: str,
    delegate_master: bool,
    prompt_index: PromptIndex,
) -> dict[str, Any]:
    """Return a prompt copy with hidden inputs configured for master/worker."""
    is_master = participant_id == "master"
    worker_index_map = {wid: idx for idx, wid in enumerate(enabled_worker_ids)}
    enabled_json = json.dumps(enabled_worker_ids)

    _override_seed_nodes(prompt_copy, prompt_index, is_master, participant_id, worker_index_map)
    _override_value_nodes(prompt_copy, prompt_index, is_master, participant_id, worker_index_map)
    _override_list_splitter_nodes(
        prompt_copy,
        prompt_index,
        is_master,
        participant_id,
        enabled_worker_ids,
        job_id_map,
        master_url,
        delegate_master,
    )
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
    _override_list_collector_nodes(
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
    prompt_copy = _override_branch_nodes(
        prompt_copy,
        prompt_index,
        is_master,
        participant_id,
        enabled_worker_ids,
        job_id_map,
        delegate_master,
    )
    _override_branch_collector_nodes(
        prompt_copy,
        prompt_index,
        is_master,
        participant_id,
        job_id_map,
        master_url,
        enabled_json,
        delegate_master,
    )

    if not is_master:
        prompt_copy = _prune_worker_downstream_of_branch_collectors(prompt_copy)
        prompt_copy = _ensure_worker_output_node(prompt_copy)

    return prompt_copy
