"""Validation and private storage for Runpod serverless worker settings.

Secret storage is supported only on POSIX masters. The module remains
importable elsewhere so ordinary worker configuration and redaction continue
to work, but secret-store operations fail explicitly without POSIX locking.
"""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .secret_fields import is_secret_field_name

try:
    import fcntl
except ImportError:  # Windows and other platforms without POSIX file locking.
    fcntl = None


class ServerlessConfigError(ValueError):
    """A serverless worker configuration is invalid."""


class SecretStoreError(RuntimeError):
    """The secret store could not be accessed safely."""


_ENDPOINT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,200}", re.ASCII)
_UNSUPPORTED_SECRET_STORE_MESSAGE = (
    "Serverless credential storage is currently supported on POSIX masters only"
)
# Images per run is also this worker's effective parallelism: every image job
# is submitted at once (the runtime's fixed PER_WORKER_JOB_LIMIT equals this),
# and the Runpod endpoint's max workers decides how many actually run together.
MAX_JOBS_PER_RUN = 16
_NUMERIC_FIELDS = {
    "jobs_per_run": (1, MAX_JOBS_PER_RUN, 1),
    "timeout_seconds": (30, 86400, 1800),
}
# Older configs saved a separate concurrency setting. It is accepted so they
# still load, then dropped: PER_WORKER_JOB_LIMIT replaces it.
_LEGACY_IGNORED_FIELDS = ("concurrency",)
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def validate_serverless_worker(worker: dict) -> dict:
    """Return a validated copy of a Runpod serverless worker record.

    Provider credentials are intentionally not required here; they are stored
    separately by :class:`SecretStore`.
    """
    if not isinstance(worker, dict):
        raise ServerlessConfigError("worker must be an object")

    try:
        result = copy.deepcopy(worker)
    except Exception:
        raise ServerlessConfigError("worker must be copyable") from None

    worker_id = result.get("id")
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise ServerlessConfigError("id must be a non-empty string")

    endpoint_id = result.get("endpoint_id")
    if not isinstance(endpoint_id, str) or not _ENDPOINT_ID_RE.fullmatch(endpoint_id):
        raise ServerlessConfigError(
            "Endpoint ID must be just the Runpod endpoint ID (letters, numbers, - or _), not a URL"
        )

    if result.get("type", "serverless") != "serverless":
        raise ServerlessConfigError("type must be serverless")
    if result.get("provider", "runpod") != "runpod":
        raise ServerlessConfigError("provider is not supported")
    if "enabled" in result and type(result["enabled"]) is not bool:
        raise ServerlessConfigError("enabled must be a boolean")

    for field, (minimum, maximum, default) in _NUMERIC_FIELDS.items():
        value = result.get(field, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ServerlessConfigError(f"{field} must be an integer from {minimum} to {maximum}")
        result[field] = value
    for field in _LEGACY_IGNORED_FIELDS:
        result.pop(field, None)

    result["type"] = "serverless"
    result["provider"] = "runpod"
    return result


def _thread_lock_for(path: str) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[path] = lock
        return lock


class SecretStore:
    """Atomically persist worker-id to API-key mappings in a private JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def get(self, worker_id: str) -> str | None:
        self._require_supported()
        worker_id = self._validate_worker_id(worker_id)
        try:
            with self._locked(create_parent=False) as (parent_fd, filename):
                data = self._read_mapping(parent_fd, filename)
                return data.get(worker_id)
        except FileNotFoundError:
            return None
        except SecretStoreError:
            raise
        except Exception:
            raise SecretStoreError("Unable to access secret store") from None

    def set(self, worker_id: str, key: str) -> None:
        self._require_supported()
        worker_id = self._validate_worker_id(worker_id)
        if (
            not isinstance(key, str)
            or not key
            or len(key) > 4096
            or "\r" in key
            or "\n" in key
        ):
            raise SecretStoreError("secret must be 1 to 4096 characters and contain no CR or LF")
        try:
            with self._locked(create_parent=True) as (parent_fd, filename):
                data = self._read_mapping(parent_fd, filename)
                data[worker_id] = key
                self._write_mapping(parent_fd, filename, data)
        except SecretStoreError:
            raise
        except Exception:
            raise SecretStoreError("Unable to access secret store") from None

    def delete(self, worker_id: str) -> None:
        self._require_supported()
        worker_id = self._validate_worker_id(worker_id)
        try:
            with self._locked(create_parent=False) as (parent_fd, filename):
                data = self._read_mapping(parent_fd, filename)
                if worker_id in data:
                    del data[worker_id]
                    self._write_mapping(parent_fd, filename, data)
        except FileNotFoundError:
            return
        except SecretStoreError:
            raise
        except Exception:
            raise SecretStoreError("Unable to access secret store") from None

    def has(self, worker_id: str) -> bool:
        return self.get(worker_id) is not None

    @staticmethod
    def _require_supported() -> None:
        # Never substitute a weaker Windows store: all operations retain the
        # dir_fd/O_NOFOLLOW and flock protections implemented below.
        if os.name != "posix" or fcntl is None:
            raise SecretStoreError(_UNSUPPORTED_SECRET_STORE_MESSAGE)

    @staticmethod
    def _validate_worker_id(worker_id: str) -> str:
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise SecretStoreError("worker_id must be a non-empty string")
        return worker_id

    def _absolute_path(self) -> Path:
        path = self.path
        if ".." in path.parts:
            raise SecretStoreError("unsafe secret store path")
        if not path.is_absolute():
            path = Path.cwd() / path
        if len(path.parts) < 2 or path.name in ("", ".", ".."):
            raise SecretStoreError("unsafe secret store path")
        return path

    @contextmanager
    def _locked(self, create_parent: bool) -> Iterator[tuple[int, str]]:
        lock_api = fcntl
        if os.name != "posix" or lock_api is None:
            raise SecretStoreError(_UNSUPPORTED_SECRET_STORE_MESSAGE)
        path = self._absolute_path()
        filename = path.name
        lock_name = f".{filename}.lock"
        thread_lock = _thread_lock_for(str(path))

        with thread_lock:
            parent_fd = self._open_parent(path, create=create_parent)
            lock_fd = None
            try:
                flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
                lock_fd = os.open(lock_name, flags, 0o600, dir_fd=parent_fd)
                lock_stat = os.fstat(lock_fd)
                if not stat.S_ISREG(lock_stat.st_mode):
                    raise SecretStoreError("unsafe secret store lock")
                os.fchmod(lock_fd, 0o600)
                lock_api.flock(lock_fd, lock_api.LOCK_EX)
                self._check_target(parent_fd, filename)
                yield parent_fd, filename
            except SecretStoreError:
                raise
            except Exception:
                raise SecretStoreError("Unable to access secret store") from None
            finally:
                if lock_fd is not None:
                    try:
                        lock_api.flock(lock_fd, lock_api.LOCK_UN)
                    except OSError:
                        pass
                    os.close(lock_fd)
                os.close(parent_fd)

    @staticmethod
    def _open_parent(path: Path, create: bool) -> int:
        parts = path.parts
        current_fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for part in parts[1:-1]:
                created = False
                try:
                    next_fd = os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=current_fd,
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(part, 0o700, dir_fd=current_fd)
                        created = True
                    except FileExistsError:
                        pass
                    next_fd = os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=current_fd,
                    )
                if created:
                    os.fchmod(next_fd, 0o700)
                os.close(current_fd)
                current_fd = next_fd
            return current_fd
        except Exception:
            os.close(current_fd)
            raise

    @staticmethod
    def _check_target(parent_fd: int, filename: str) -> None:
        try:
            target_stat = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(target_stat.st_mode):
            raise SecretStoreError("unsafe secret store file")

    @classmethod
    def _read_mapping(cls, parent_fd: int, filename: str) -> dict[str, str]:
        cls._check_target(parent_fd, filename)
        try:
            fd = os.open(
                filename,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return {}
        except Exception:
            raise SecretStoreError("Unable to access secret store") from None

        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise SecretStoreError("unsafe secret store file")
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "rb") as stream:
                raw = stream.read()
            try:
                mapping = json.loads(raw.decode("utf-8"), object_pairs_hook=cls._unique_object)
            except Exception:
                raise SecretStoreError("secret store is corrupt") from None
            if not isinstance(mapping, dict) or any(
                not isinstance(worker_id, str)
                or not worker_id.strip()
                or not isinstance(key, str)
                or not key
                for worker_id, key in mapping.items()
            ):
                raise SecretStoreError("secret store is corrupt")
            return mapping
        except SecretStoreError:
            raise
        except Exception:
            raise SecretStoreError("Unable to access secret store") from None
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    @staticmethod
    def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    @staticmethod
    def _write_mapping(parent_fd: int, filename: str, mapping: dict[str, str]) -> None:
        payload = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        temp_name = f".{filename}.{secrets.token_hex(12)}.tmp"
        temp_fd = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
            temp_fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
            os.fchmod(temp_fd, 0o600)
            with os.fdopen(temp_fd, "wb") as stream:
                temp_fd = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            SecretStore._check_target(parent_fd, filename)
            os.replace(temp_name, filename, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        except SecretStoreError:
            raise
        except Exception:
            raise SecretStoreError("Unable to access secret store") from None
        finally:
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except OSError:
                pass


class ServerlessSecretChanges:
    """Apply serverless secret changes and restore them unless committed.

    Use this around the associated config transaction so a failed config save
    compensates any secret-store writes or deletions made along the way.
    """

    def __init__(self):
        self._originals: dict[tuple[int, str], tuple[SecretStore, str, str | None]] = {}
        self._mutations: list[tuple[int, str]] = []
        self._committed = False

    def __enter__(self) -> "ServerlessSecretChanges":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        if not self._committed:
            self.rollback()
        return False

    def capture(self, store: SecretStore, worker_id: str) -> str | None:
        """Capture and return the original key once per store/worker pair."""
        mutation_id = (id(store), worker_id)
        if mutation_id not in self._originals:
            self._originals[mutation_id] = (store, worker_id, store.get(worker_id))
        return self._originals[mutation_id][2]

    def set(self, store: SecretStore, worker_id: str, key: str) -> None:
        self._prepare_mutation(store, worker_id)
        store.set(worker_id, key)

    def delete(self, store: SecretStore, worker_id: str) -> None:
        self._prepare_mutation(store, worker_id)
        store.delete(worker_id)

    def commit(self) -> None:
        """Keep the secret changes after the corresponding config save."""
        self._committed = True

    def rollback(self) -> None:
        """Best-effort restoration without masking the original route error."""
        for mutation_id in reversed(self._mutations):
            store, worker_id, prior_secret = self._originals[mutation_id]
            try:
                if prior_secret is None:
                    store.delete(worker_id)
                else:
                    store.set(worker_id, prior_secret)
            except Exception:
                pass
        self._mutations.clear()

    def _prepare_mutation(self, store: SecretStore, worker_id: str) -> None:
        self.capture(store, worker_id)
        mutation_id = (id(store), worker_id)
        if mutation_id not in self._mutations:
            # Record before applying: a store operation may fail after replacing
            # its file, so the exception path must still attempt compensation.
            self._mutations.append(mutation_id)


def _redact_credentials(value):
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if is_secret_field_name(key):
                continue
            redacted[key] = _redact_credentials(item)
        return redacted
    if isinstance(value, list):
        return [_redact_credentials(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_credentials(item) for item in value)
    return value


def public_config(config, store: SecretStore):
    """Return a deep-copied public configuration with credentials removed."""
    result = _redact_credentials(copy.deepcopy(config))
    if not isinstance(result, dict):
        return result

    workers = result.get("workers")
    if isinstance(workers, list):
        for worker in workers:
            if isinstance(worker, dict) and worker.get("type") == "serverless":
                worker_id = worker.get("id")
                configured = (
                    store.has(worker_id)
                    if isinstance(worker_id, str) and worker_id.strip()
                    else False
                )
                worker["api_key_configured"] = configured
    return result
