import json
from datetime import datetime, timedelta
from fastmcp import FastMCP

mcp = FastMCP("logs")

# ... mock data ...

_now = datetime.now()

_LOGS = {
    "payment-service": [
        {
            "level": "ERROR",
            "message": "connection pool exhausted: max=10 current=10",
            "trace_id": "t001",
            "timestamp": (_now - timedelta(minutes=2)).isoformat(),
        },
        {
            "level": "ERROR",
            "message": "connection pool exhausted: max=10 current=10",
            "trace_id": "t002",
            "timestamp": (_now - timedelta(minutes=3)).isoformat(),
        },
        {
            "level": "ERROR",
            "message": "connection pool exhausted: max=10 current=10",
            "trace_id": "t003",
            "timestamp": (_now - timedelta(minutes=5)).isoformat(),
        },
        {
            "level": "ERROR",
            "message": "PostgreSQL connection timeout after 30s",
            "trace_id": "t004",
            "timestamp": (_now - timedelta(minutes=6)).isoformat(),
        },
        {
            "level": "ERROR",
            "message": "connection pool exhausted: max=10 current=10",
            "trace_id": "t005",
            "timestamp": (_now - timedelta(minutes=8)).isoformat(),
        },
        {
            "level": "INFO",
            "message": "Request processed successfully",
            "trace_id": "t010",
            "timestamp": (_now - timedelta(minutes=1)).isoformat(),
        },
    ]
}

_DEPLOYMENTS = {
    "payment-service": [
        {
            "deployed_at": (_now - timedelta(minutes=5)).isoformat(),
            "version": "v2.3.1",
            "change": "scale replicas 3 → 5",
            "author": "ci-bot",
        }
    ]
}


@mcp.tool()
def search_logs(service: str, query: str = "", limit: int = 20) -> str:
    """Search raw log entries for a service. Optionally filter by keyword."""
    logs = _LOGS.get(service, [])
    if query:
        logs = [l for l in logs if query.lower() in l["message"].lower()]
    return json.dumps(logs[:limit], indent=2)


@mcp.tool()
def get_error_summary(service: str, window_minutes: int = 15) -> str:
    """Get error counts grouped by type. Returns error rate and dominant error types."""
    logs = _LOGS.get(service, [])
    errors = [l for l in logs if l["level"] == "ERROR"]

    counts: dict[str, int] = {}
    for e in errors:
        key = e["message"][:50]
        counts[key] = counts.get(key, 0) + 1

    summary = {
        "service": service,
        "window_minutes": window_minutes,
        "total_errors": len(errors),
        "error_rate_per_min": round(len(errors) / window_minutes, 2),
        "top_errors": sorted(counts.items(), key=lambda x: -x[1]),
    }
    return json.dumps(summary, indent=2)


@mcp.tool()
def get_service_deployments(service: str, hours: int = 2) -> str:
    """Get recent deployment events for a service."""
    return json.dumps(_DEPLOYMENTS.get(service, []), indent=2)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8003)
