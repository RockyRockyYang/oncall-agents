# High CPU Usage Runbook

## Symptoms
- CPU usage above 80% for more than 5 minutes
- System load average exceeds number of CPU cores
- Application response times degrading

## Common Causes
1. Runaway process consuming excessive CPU
2. Insufficient capacity for current traffic load
3. Inefficient database queries causing CPU spikes
4. Memory pressure causing excessive garbage collection

## Diagnosis Steps
1. Run `top` or `htop` to identify the offending process
2. Check application logs for error spikes around the same time
3. Review recent deployments that may have introduced the issue
4. Check if traffic volume increased abnormally

## Resolution
- If a runaway process: restart the service
- If traffic spike: scale horizontally or enable rate limiting
- If bad deployment: rollback to previous version
