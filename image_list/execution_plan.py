"""Prepare image-list execution without changing the saved/API prompt or registry."""
from __future__ import annotations

import copy
import sys

from .runtime import (
    END_CLASS,
    _ancestors_for_inputs,
    _downstream_until,
    _image_list_pairs,
    _is_output_node,
    _prompt_links,
    validate_image_list_regions,
)


def _reachable(roots: set[str], edges: dict[str, set[str]]) -> set[str]:
    visited = set()
    pending = list(roots)
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        pending.extend(edges[node_id] - visited)
    return visited


def prepare_image_list_execution(prompt: dict) -> tuple[dict, list[str]]:
    """Return an execution copy and its outer-master output targets.

    Disposable per-item side branches are removed before boundary validation.
    Required output-node utilities remain normal dependencies, but must not be
    independent outer roots. Callers must use these targets (and End-only targets
    for item execution); this does not suppress side effects inside required nodes.
    """
    pairs = _image_list_pairs(prompt)
    prepared = copy.deepcopy(prompt)
    upstream = {node_id: set() for node_id in prepared}
    downstream = {node_id: set() for node_id in prepared}
    for node_id, node in prepared.items():
        for _, source, _ in _prompt_links(node, prepared):
            upstream[node_id].add(source)
            downstream[source].add(node_id)

    # Never erase another region or its dependencies to make a bad graph pass.
    boundaries = {node_id for pair in pairs for node_id in pair}
    required = _reachable(boundaries, upstream)
    discarded = set()
    for start_id, end_id in pairs:
        region_nodes = _ancestors_for_inputs(prepared, end_id, start_id, ("images", "masks", "texts"))
        outside = _downstream_until(prepared, start_id, end_id) - region_nodes
        # Preserve bypass joins and their whole chain into post-End computation.
        # They must fail boundary validation, not disappear with debug previews.
        post_end = _reachable({end_id}, downstream)
        protected = required | _reachable(post_end, upstream)
        discarded.update(outside - protected)
    for node_id in discarded:
        del prepared[node_id]

    regions = validate_image_list_regions(prepared, explicit_targets=True)
    per_item = set().union(*(region["branch_ids"] for region in regions))
    registry = getattr(sys.modules.get("nodes"), "NODE_CLASS_MAPPINGS", {})
    targets = [
        node_id for node_id, node in prepared.items()
        if node_id not in per_item and isinstance(node, dict)
        and (node.get("class_type") == END_CLASS or _is_output_node(node, registry))
    ]
    return prepared, targets
