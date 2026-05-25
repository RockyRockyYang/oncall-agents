import json

import psutil
from fastmcp import FastMCP

mcp = FastMCP("monitor")

# Mock time-series data for service-aware tools.
# payment-service: CPU/memory are nominal, but DB connections are maxed out.
_CPU_METRICS = {
    "payment-service": {
        "series": [32, 35, 38, 33, 36, 34, 37, 35, 33, 36],  # 10-min samples
        "avg": 35.0,
        "max": 38.0,
        "p95": 37.5,
    }
}

_MEMORY_METRICS = {
    "payment-service": {
        "series": [60, 62, 63, 61, 62, 64, 62, 61, 63, 62],  # percent samples
        "avg": 62.0,
        "max": 64.0,
        "p95": 63.5,
    }
}

_DB_CONNECTIONS = {
    "payment-service": {"active": 10, "max": 10, "waiting": 47}
}


@mcp.tool()
def get_cpu_usage() -> str:
    """Get the current CPU usage percentage."""
    percent = psutil.cpu_percent(interval=1)
    return f"Current CPU usage: {percent}%"


@mcp.tool()
def get_memory_usage() -> str:
    """Get the current memory usage."""
    mem = psutil.virtual_memory()
    return (
        f"Used: {mem.used / (1024 ** 3):.2f} GB, "
        f"Available: {mem.available / (1024 ** 3):.2f} GB"
        f"percent: {mem.percent}%"
    )


@mcp.tool()
def list_top_processes(limit: int = 5) -> str:
    """List the top CPU-consuming processes."""
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
        try:
            info = proc.info
            if info["cpu_percent"] is not None:
                processes.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    processes.sort(key=lambda x: x["cpu_percent"] or 0, reverse=True)
    top_processes = processes[:limit]
    lines = [
        f"PID {p['pid']:>6}  CPU {p['cpu_percent']:>5.1f}%  {p['name']}"
        for p in top_processes
    ]
    return "\n".join(lines) if lines else "No processes found."


@mcp.tool()
def query_cpu_metrics(service_name: str, start_time: str = "", end_time: str = "") -> str:
    """Get time-series CPU usage for a named service, with avg/max/p95 summary."""
    data = _CPU_METRICS.get(service_name)
    if not data:
        return json.dumps({"service": service_name, "error": "no data found"})
    return json.dumps({"service": service_name, **data})


@mcp.tool()
def query_memory_metrics(service_name: str, start_time: str = "", end_time: str = "") -> str:
    """Get time-series memory usage for a named service, with avg/max/p95 summary."""
    data = _MEMORY_METRICS.get(service_name)
    if not data:
        return json.dumps({"service": service_name, "error": "no data found"})
    return json.dumps({"service": service_name, **data})


@mcp.tool()
def query_db_connections(service_name: str) -> str:
    """Get current active DB connections vs max for a named service."""
    data = _DB_CONNECTIONS.get(service_name)
    if not data:
        return json.dumps({"service": service_name, "error": "no data found"})
    result = {
        "service": service_name,
        "active": data["active"],
        "max": data["max"],
        "waiting": data["waiting"],
        "utilization": f"{data['active'] / data['max'] * 100:.0f}%",
    }
    return json.dumps(result)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8004)
