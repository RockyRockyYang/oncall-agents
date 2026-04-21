import psutil
from fastmcp import FastMCP

mcp = FastMCP("monitor")


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
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    processes.sort(key=lambda x: x["cpu_percent"], reverse=True)
    top_processes = processes[:limit]
    lines = [
        f"PID {p['pid']:>6}  CPU {p['cpu_percent']:>5.1f}%  {p['name']}"
        for p in top_processes
    ]
    return "\n".join(lines) if lines else "No processes found."


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8004)
