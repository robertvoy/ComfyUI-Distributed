from typing import Any


def get_prompt_server_instance() -> Any:
    import server as _server

    return _server.PromptServer.instance


def throw_if_processing_interrupted() -> None:
    try:
        import comfy.model_management as model_management

        model_management.throw_exception_if_processing_interrupted()
    except Exception:
        return
