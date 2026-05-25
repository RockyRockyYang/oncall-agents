# Runbook: High Memory Usage

## Alert Trigger Conditions

- **Condition:** Memory usage exceeds 85% for more than 5 minutes
- **Severity:** P2 — investigate promptly; OOM kill risk if left unresolved
- **Commonly affected services:** payment-service, order-service, user-service
- **Alert source:** Container memory monitor or service-level metrics

---

## Investigation Steps

Follow these steps in order. Use the specified tool at each step and record the result before proceeding.

1. **Record the current time**
   - Tool: `get_current_time`
   - Parameters: none
   - Purpose: Anchor the investigation timeline for log queries and metric windows.

2. **Check memory metrics for the affected service**
   - Tool: `query_memory_metrics`
   - Parameters: `service_name` = name of the affected service from the alert
   - What to look for: avg > 85% is abnormal; p95 near 100% means OOM kill is imminent. Check the trend in `series` — a slow steady rise suggests a memory leak; a sudden spike suggests a traffic burst or large payload.

3. **Check CPU metrics to distinguish GC pressure from a leak**
   - Tool: `query_cpu_metrics`
   - Parameters: `service_name` = name of the affected service from the alert
   - What to look for: if CPU is also high alongside memory, the service is likely spending time in garbage collection. If CPU is normal but memory keeps rising, a memory leak is more likely.

4. **Search error logs for OOM or GC-related errors**
   - Tool: `get_error_summary`
   - Parameters: `service` = affected service name, `window_minutes` = 15
   - What to look for: errors containing `OutOfMemoryError`, `OOMKilled`, `GC overhead limit exceeded`, or `heap space`.

5. **Search raw logs for memory-related events**
   - Tool: `search_logs`
   - Parameters: `service` = affected service name, `query` = `memory` or top error keyword from step 4
   - What to look for: frequency and timing of OOM events; whether they correlate with specific request types or batch jobs.

6. **Check recent deployments**
   - Tool: `get_service_deployments`
   - Parameters: `service` = affected service name, `hours` = 2
   - What to look for: a deployment shortly before memory started rising is a strong signal for a regression (memory leak introduced in new code).

---

## Common Root Causes

### 1. Memory Leak
**Signals:**
- `query_memory_metrics`: `series` shows a slow, steady upward trend over time; memory does not drop after GC
- `query_cpu_metrics`: CPU may spike periodically (GC activity) but returns to normal
- `get_error_summary`: may show `GC overhead limit exceeded` after prolonged leak

**Why it happens:** Objects are allocated but never released — common causes include unbounded caches, event listeners not unregistered, or connection objects not closed properly.

### 2. Traffic Burst / Object Surge
**Signals:**
- `query_memory_metrics`: `series` shows a sudden spike, not a gradual rise
- `query_cpu_metrics`: CPU also spikes at the same time as memory
- `get_error_summary`: errors align with a specific time window matching the traffic surge

**Why it happens:** A sudden increase in requests creates more objects than the GC can collect fast enough. Memory recovers once traffic normalizes.

### 3. Large Payload Processing
**Signals:**
- `query_memory_metrics`: memory spikes at specific intervals (suggesting batch jobs or scheduled tasks)
- `search_logs`: log entries mention file uploads, data exports, or report generation around the spike time
- Memory returns to normal after the task completes

**Why it happens:** Loading large files or datasets into memory at once instead of streaming them.

### 4. Deploy Regression
**Signals:**
- `get_service_deployments`: a deployment exists shortly before memory started rising
- `query_memory_metrics`: `series` shows the trend change starting around the deploy timestamp
- `get_error_summary`: new error types appear after the deploy

**Why it happens:** New code introduced a memory leak, an oversized cache, or inefficient data structure.

---

## Resolution Procedures

### Memory Leak
1. Restart affected instances immediately to recover memory and restore service.
2. Increase memory limits temporarily to prevent OOM kill while investigating.
3. Capture a heap dump before restarting if possible, for offline analysis.
4. Identify the leaking code path and deploy a fix.

### Traffic Burst
1. Scale out — add more instances to distribute memory pressure.
2. Enable rate limiting to protect the service from being overwhelmed.
3. Verify memory returns to baseline once traffic normalizes.

### Large Payload Processing
1. Switch to streaming processing to avoid loading entire datasets into memory.
2. Move batch jobs to off-peak hours or dedicated worker instances.
3. Set request size limits for file uploads.

### Deploy Regression
1. Immediately roll back to the previous version.
2. Verify memory trend starts declining within 5 minutes of rollback.
3. Profile memory usage of the new version in a staging environment before re-deploying.

---

## Verification

After applying a fix, confirm recovery using the following checks:

- `query_memory_metrics` — avg memory should drop below 70%; `series` should show a downward trend
- `get_error_summary` — no new OOM or GC-related errors in the past 5 minutes
- `query_cpu_metrics` — CPU should return to normal (no sustained GC spikes)
- Monitor for at least 15 minutes to confirm the trend does not resume
