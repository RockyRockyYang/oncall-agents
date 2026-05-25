# Runbook: High HTTP 5xx Error Rate

## Alert Trigger Conditions

- **Condition:** HTTP 5xx error rate exceeds 10% over a 15-minute window
- **Severity:** P1 — requires immediate investigation
- **Commonly affected services:** payment-service, order-service, user-service
- **Alert source:** API gateway or service-level error rate monitor

---

## Investigation Steps

Follow these steps in order. Use the specified tool at each step and record the result before proceeding.

1. **Record the current time**
   - Tool: `get_current_time`
   - Parameters: none
   - Purpose: Anchor the investigation timeline. All subsequent tool calls should reference this timestamp.

2. **Check CPU metrics for the affected service**
   - Tool: `query_cpu_metrics`
   - Parameters: `service_name` = name of the affected service from the alert
   - What to look for: avg < 70% is normal; p95 > 90% sustained indicates CPU saturation as root cause.

3. **Check memory metrics for the affected service**
   - Tool: `query_memory_metrics`
   - Parameters: `service_name` = name of the affected service from the alert
   - What to look for: avg < 80% is normal; values near 100% indicate OOM risk as root cause.

4. **Check database connection utilization**
   - Tool: `query_db_connections`
   - Parameters: `service_name` = name of the affected service from the alert
   - What to look for: if `active == max` and `waiting > 0`, the connection pool is exhausted — high-confidence root cause. Proceed directly to DB connection pool resolution; skip remaining steps.

5. **Search error logs for dominant error types**
   - Tool: `get_error_summary`
   - Parameters: `service` = affected service name, `window_minutes` = 15
   - What to look for: the top error message by count. Keywords like `connection pool exhausted`, `connection timeout`, or `too many connections` confirm the DB hypothesis.

6. **Search raw logs for additional context**
   - Tool: `search_logs`
   - Parameters: `service` = affected service name, `query` = top error keyword from step 5
   - What to look for: frequency, trace_ids, and whether errors cluster around a specific time.

7. **Check recent deployments (if root cause is still unclear)**
   - Tool: `get_service_deployments`
   - Parameters: `service` = affected service name, `hours` = 2
   - What to look for: any deployment within 30 minutes of the alert onset is a strong regression signal.

---

## Common Root Causes

### 1. DB Connection Pool Exhausted
**Signals:**
- `query_db_connections`: `active == max`, `waiting > 0`, `utilization = 100%`
- `get_error_summary`: dominant error contains `connection pool exhausted`, `too many connections`, or `connection timeout`
- CPU and memory metrics are within normal range

**Why it happens:** A scale-out event (more replicas) increases total connections without increasing DB pool capacity. Each replica opens its own connection pool, so 5 replicas × pool_size=10 = 50 connections hitting a DB configured for 10.

### 2. Downstream Service Failure (Cascade)
**Signals:**
- CPU and memory metrics are within normal range
- `search_logs`: errors contain `connection refused`, `upstream timeout`, or the name of an internal dependency
- `get_service_deployments`: no recent deployment
- `query_db_connections`: connections are within normal range

**Why it happens:** A dependency (e.g., auth service, inventory service) is down or slow, causing the upstream service to queue requests until they time out.

### 3. Deploy Regression
**Signals:**
- `get_service_deployments`: a deployment exists within 30 minutes of the alert onset
- `get_error_summary`: error rate increased sharply after the deploy timestamp
- Errors may include `NullPointerException`, `configuration missing`, or application-specific panics

**Why it happens:** A code bug, missing migration, or misconfigured environment variable introduced in the new version.

### 4. OOM / GC Pressure
**Signals:**
- `query_memory_metrics`: avg memory near or above 90%, p95 near 100%
- `search_logs`: errors contain `out of memory`, `OOMKilled`, or extreme GC pause durations
- Latency spikes visible before the error rate increase

**Why it happens:** Memory leak, large payload processing without streaming, or insufficient container memory limits.

---

## Resolution Procedures

### DB Connection Pool Exhausted
1. Immediately reduce `DB_POOL_SIZE` env var per replica (e.g., from 10 to 3) and trigger a rolling restart.
2. Alternatively, deploy a connection pooler (PgBouncer) in front of the database to multiplex connections.
3. Long-term: add `query_db_connections` to the deployment checklist — alert if utilization exceeds 80% post-deploy.

### Downstream Service Failure
1. Identify the failing dependency from error logs.
2. Check the dependency's own error rate and health endpoint.
3. If the dependency cannot recover quickly, enable the circuit breaker or serve a degraded response (e.g., cached data).

### Deploy Regression
1. Immediately roll back to the previous version using the deployment system.
2. Verify error rate drops within 2–3 minutes of rollback.
3. File a post-mortem and re-examine the deployment in a staging environment.

### OOM / GC Pressure
1. Increase container memory limits as a temporary mitigation.
2. Trigger a rolling restart to clear leaked memory.
3. Profile heap usage in staging to locate the memory leak source.

---

## Verification

After applying a fix, confirm recovery using the following checks:

- `get_error_summary` — total errors should drop significantly; error rate per minute should return to baseline (< 0.1/min under normal load)
- `query_db_connections` — `active` should be well below `max`; `waiting` should be 0
- `search_logs` — no new occurrences of the dominant error type in the past 2 minutes
- Monitor HTTP 5xx rate in the API gateway dashboard — should fall below 1% within 5 minutes of the fix
