# Runbook: Slow Response Time

## Alert Trigger Conditions

- **Condition:** P99 response time exceeds 3 seconds for more than 5 minutes
- **Severity:** P2 — user-facing latency degradation; may escalate to P1 if request backlog builds
- **Commonly affected services:** payment-service, order-service, user-service
- **Alert source:** API gateway latency monitor or service-level P99 metrics

---

## Investigation Steps

Follow these steps in order. Use the specified tool at each step and record the result before proceeding.

1. **Record the current time**
   - Tool: `get_current_time`
   - Parameters: none
   - Purpose: Anchor the investigation timeline for all subsequent queries.

2. **Check CPU metrics for the affected service**
   - Tool: `query_cpu_metrics`
   - Parameters: `service_name` = name of the affected service from the alert
   - What to look for: p95 > 80% sustained suggests CPU saturation is throttling request processing. avg > 70% combined with high latency points to CPU as a bottleneck.

3. **Check memory metrics for the affected service**
   - Tool: `query_memory_metrics`
   - Parameters: `service_name` = name of the affected service from the alert
   - What to look for: avg > 85% may indicate GC pressure causing stop-the-world pauses, which directly causes latency spikes.

4. **Check database connection utilization**
   - Tool: `query_db_connections`
   - Parameters: `service_name` = name of the affected service from the alert
   - What to look for: if `active == max` and `waiting > 0`, requests are queuing for a DB connection — this is a direct cause of slow response. Proceed to DB connection pool resolution.

5. **Search error logs for timeout-related errors**
   - Tool: `get_error_summary`
   - Parameters: `service` = affected service name, `window_minutes` = 15
   - What to look for: errors containing `timeout`, `slow query`, `connection refused`, or downstream service names indicate where latency is originating.

6. **Search raw logs for slow request details**
   - Tool: `search_logs`
   - Parameters: `service` = affected service name, `query` = `timeout` or top error keyword from step 5
   - What to look for: which endpoints or operations are slow; whether slowness is tied to specific request patterns or a specific downstream dependency.

7. **Check recent deployments**
   - Tool: `get_service_deployments`
   - Parameters: `service` = affected service name, `hours` = 2
   - What to look for: a deployment within 30 minutes of the latency increase is a strong regression signal — new code may have introduced an inefficient query or synchronous blocking call.

---

## Common Root Causes

### 1. DB Connection Pool Exhausted
**Signals:**
- `query_db_connections`: `active == max`, `waiting > 0`, `utilization = 100%`
- `get_error_summary`: dominant errors contain `connection pool exhausted` or `connection timeout`
- CPU and memory are within normal range

**Why it happens:** Requests queue waiting for a DB connection. Each queued request adds to P99 latency even if the DB query itself is fast.

### 2. CPU Saturation
**Signals:**
- `query_cpu_metrics`: avg > 70%, p95 > 90% sustained
- `get_error_summary`: errors may include `request timeout` as a secondary effect
- Latency increases correlate with CPU spikes in the `series` data

**Why it happens:** Insufficient CPU capacity to process incoming requests at the current rate. Each request waits longer in the queue before being scheduled.

### 3. GC Pressure / Memory Saturation
**Signals:**
- `query_memory_metrics`: avg > 85%, p95 near 100%
- `query_cpu_metrics`: periodic CPU spikes (GC runs) correlating with latency spikes
- `search_logs`: entries mention GC pause durations or heap pressure

**Why it happens:** Frequent or long GC pauses cause stop-the-world events, during which no requests are processed — all in-flight requests experience the pause as added latency.

### 4. Downstream Service Slow / Timeout
**Signals:**
- CPU, memory, and DB connections are all within normal range
- `get_error_summary`: errors mention a specific internal service name or contain `upstream timeout`
- `search_logs`: log entries show repeated slow calls to the same downstream endpoint

**Why it happens:** A dependency (auth service, inventory service, external API) is responding slowly, and the upstream service blocks waiting for its response.

### 5. Deploy Regression
**Signals:**
- `get_service_deployments`: a deployment exists within 30 minutes of the latency onset
- `query_cpu_metrics` or `query_memory_metrics`: resource usage increased after the deploy
- `get_error_summary`: new error types or increased error volume after the deploy

**Why it happens:** New code introduced an N+1 query, a missing cache, a synchronous blocking call, or an unoptimized algorithm.

---

## Resolution Procedures

### DB Connection Pool Exhausted
1. Reduce `DB_POOL_SIZE` per replica and trigger a rolling restart to free waiting connections immediately.
2. Deploy PgBouncer as a connection pooler to multiplex connections across replicas.
3. Long-term: set a utilization alert at 80% to catch this before it reaches 100%.

### CPU Saturation
1. Scale out — add more instances to distribute load.
2. Enable rate limiting to shed excess traffic while scaling.
3. Profile CPU usage to identify hot code paths; optimize after service is stable.

### GC Pressure
1. Increase memory limits to give GC more headroom.
2. Trigger a rolling restart to clear accumulated heap objects.
3. Investigate memory usage patterns to identify the source of object accumulation.

### Downstream Service Slow
1. Check the downstream service's own health and error rate.
2. Enable circuit breaker or timeout to stop blocked requests from accumulating.
3. If the downstream cannot recover quickly, serve a degraded response using cached data.

### Deploy Regression
1. Immediately roll back to the previous version.
2. Verify P99 latency starts recovering within 2–3 minutes of rollback.
3. Profile the new version in staging to identify the slow code path before re-deploying.

---

## Verification

After applying a fix, confirm recovery using the following checks:

- `query_cpu_metrics` — avg should return below 70%
- `query_memory_metrics` — avg should return below 80%
- `query_db_connections` — `waiting` should be 0; `active` well below `max`
- `get_error_summary` — timeout-related errors should drop to zero or near-zero
- Monitor P99 response time for at least 10 minutes to confirm sustained recovery
