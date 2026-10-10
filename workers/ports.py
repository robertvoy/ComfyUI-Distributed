"""Host-local worker port selection and launch preflight (no ComfyUI imports)."""
import errno
import socket
from urllib.parse import urlsplit


def _port_number(value):
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise ValueError("Worker port must be an integer between 1 and 65535") from None
    if isinstance(value, (bool, float)) or not 1 <= port <= 65535:
        raise ValueError("Worker port must be an integer between 1 and 65535")
    return port


def _local_config(worker):
    local_hosts = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "::")
    host = (worker.get("host") or "localhost").strip().lower()
    if worker.get("type") == "local" or host in local_hosts:
        return True
    try:
        host = urlsplit(host if "://" in host else "//" + host).hostname
    except ValueError:
        return False
    return host in local_hosts


def _reserved_ports(workers, exclude_id=None):
    reserved = {}
    for worker in workers:
        if exclude_id is not None and str(worker.get("id")) == str(exclude_id):
            continue
        if not _local_config(worker):
            continue
        try:
            port = _port_number(worker.get("port"))
        except ValueError:
            continue
        reserved[port] = worker.get("name") or worker.get("id") or "another local worker"
    return reserved


def is_port_available(port):
    """Probe bindability without SO_REUSEADDR; include IPv6-only listeners."""
    addresses = [(socket.AF_INET, "0.0.0.0")]
    if socket.has_ipv6:
        addresses.append((socket.AF_INET6, "::"))
    for family, address in addresses:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.bind((address, port))
        except OSError as exc:
            if family == socket.AF_INET6 and exc.errno in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                continue
            return False
    return True


def allocate_worker_ports(master_port, workers, count):
    """Select ports above the actual master, skipping reservations/listeners.

    This is a snapshot, not a reservation. Launch must validate again because
    another process can claim a suggested port before the user starts a worker.
    """
    master_port = _port_number(master_port)
    reserved = _reserved_ports(workers)
    selected = []
    if count == 0:
        return selected
    for port in range(master_port + 1, 65536):
        if port not in reserved and is_port_available(port):
            selected.append(port)
            if len(selected) == count:
                return selected
    raise ValueError(f"Not enough available worker ports above master port {master_port}; need {count}")


def validate_worker_port(worker, master_port, workers):
    """Reject conflicts without silently changing a configured worker port."""
    port = _port_number(worker.get("port"))
    if port == _port_number(master_port):
        raise ValueError(f"Worker port {port} conflicts with the master port {master_port}; choose another worker port")
    reserved = _reserved_ports(workers, exclude_id=worker.get("id"))
    if port in reserved:
        raise ValueError(f"Worker port {port} is assigned to local worker {reserved[port]}; choose another worker port")
    if not is_port_available(port):
        raise ValueError(f"Worker port {port} is already in use on this host; choose another worker port")
