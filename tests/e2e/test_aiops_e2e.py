"""
End-to-end test for the AIOps investigation workflow.

Requires all services to be running:
    docker-compose up -d
    uv run python mcp_servers/logs_server.py &
    uv run python mcp_servers/monitor_server.py &
    uvicorn app.main:app --port 9900

Run:
    uv run pytest tests/e2e/test_aiops_e2e.py -v -s
"""

import json
from pathlib import Path

import httpx
import pytest

BASE_URL = "http://localhost:9900"
ALERT = "payment-service: HTTP 5xx error rate exceeded 10% for 15 minutes"


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data:
                events.append(json.loads(data))
    return events


@pytest.fixture(scope="module")
def ingest_runbook():
    """Ingest docs/high_error_rate.md before running AIOps tests."""
    runbook_path = Path("docs/high_error_rate.md")
    text = runbook_path.read_text()
    resp = httpx.post(
        f"{BASE_URL}/ingest",
        json={"source": "high_error_rate", "text": text},
        timeout=30,
    )
    assert resp.status_code == 200, f"Ingest failed: {resp.text}"
    data = resp.json()
    print(f"\nIngested {data['chunks_inserted']} chunks from {data['source']}")
    return data


def test_services_healthy():
    """All backend services should be reachable before running e2e tests."""
    resp = httpx.get(f"{BASE_URL}/docs", timeout=10)
    assert resp.status_code == 200, "FastAPI not reachable at port 9900"

    mcp_resp_monitor = httpx.get("http://localhost:8004/health", timeout=5)
    assert mcp_resp_monitor.status_code in (200, 404), "Monitor MCP server not reachable"

    mcp_resp_logs = httpx.get("http://localhost:8003/health", timeout=5)
    assert mcp_resp_logs.status_code in (200, 404), "Logs MCP server not reachable"


def test_aiops_investigate_event_sequence(ingest_runbook):
    """
    Full investigation: verify SSE event sequence is plan → step_complete(s) → report → complete.
    """
    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/aiops/investigate",
            json={"alert": ALERT, "session_id": "e2e-full-flow"},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            raw = response.read().decode()

    events = _parse_sse(raw)
    types = [e["type"] for e in events]

    print(f"\nReceived {len(events)} SSE events: {types}")
    for e in events:
        print(f"  [{e['type']}] {e.get('message', e.get('report', '')[:80])}")

    assert types[0] == "plan", f"First event should be 'plan', got: {types[0]}"
    assert "step_complete" in types, "Should have at least one step_complete event"
    assert "report" in types, "Should have a final report event"
    assert types[-1] == "complete", f"Last event should be 'complete', got: {types[-1]}"


def test_aiops_investigate_plan_has_steps(ingest_runbook):
    """The plan event should contain a non-empty list of investigation steps."""
    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/aiops/investigate",
            json={"alert": ALERT, "session_id": "e2e-plan-check"},
        ) as response:
            raw = response.read().decode()

    events = _parse_sse(raw)
    plan_event = next((e for e in events if e["type"] == "plan"), None)

    assert plan_event is not None, "No plan event received"
    assert isinstance(plan_event.get("plan"), list), "plan event should have a 'plan' list"
    assert len(plan_event["plan"]) > 0, "Plan should not be empty"

    print(f"\nPlan steps ({len(plan_event['plan'])}):")
    for i, step in enumerate(plan_event["plan"], 1):
        print(f"  {i}. {step}")


def test_aiops_investigate_report_is_markdown(ingest_runbook):
    """The final report should be non-empty Markdown."""
    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/aiops/investigate",
            json={"alert": ALERT, "session_id": "e2e-report-check"},
        ) as response:
            raw = response.read().decode()

    events = _parse_sse(raw)
    report_event = next((e for e in events if e["type"] == "report"), None)

    assert report_event is not None, "No report event received"
    report = report_event.get("report", "")
    assert len(report) > 100, "Report is too short"
    assert "#" in report, "Report should contain Markdown headings"

    print(f"\nReport preview ({len(report)} chars):\n{report[:400]}")


def test_aiops_missing_alert_returns_422():
    """Request without required 'alert' field should return 422."""
    resp = httpx.post(
        f"{BASE_URL}/aiops/investigate",
        json={"session_id": "no-alert"},
        timeout=10,
    )
    assert resp.status_code == 422
