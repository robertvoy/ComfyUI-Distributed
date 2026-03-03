import copy
from typing import Any


def clone_control_chain(control: Any, clone_hint: bool = True) -> Any:
    """Shallow copy the ControlNet chain, optionally cloning hints but sharing models."""
    if control is None:
        return None
    new_control = copy.copy(control)
    if clone_hint and hasattr(control, 'cond_hint_original'):
        hint = getattr(control, 'cond_hint_original', None)
        new_control.cond_hint_original = hint.clone() if hint is not None else None
    if hasattr(control, 'previous_controlnet'):
        new_control.previous_controlnet = clone_control_chain(control.previous_controlnet, clone_hint)
    return new_control


def clone_conditioning(
    cond_list: list[tuple[Any, dict[str, Any]]] | list[list[Any]],
    clone_hints: bool = True,
) -> list[list[Any]]:
    """Clone conditioning without duplicating ControlNet models."""
    new_cond = []
    for emb, cond_dict in cond_list:
        new_emb = emb.clone() if emb is not None else None
        new_dict = cond_dict.copy()
        if 'control' in new_dict:
            new_dict['control'] = clone_control_chain(new_dict['control'], clone_hints)
        if 'mask' in new_dict and new_dict['mask'] is not None:
            new_dict['mask'] = new_dict['mask'].clone()
        if 'pooled_output' in new_dict and new_dict['pooled_output'] is not None:
            new_dict['pooled_output'] = new_dict['pooled_output'].clone()
        if 'area' in new_dict:
            new_dict['area'] = new_dict['area'][:]
        new_cond.append([new_emb, new_dict])
    return new_cond
