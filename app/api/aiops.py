import json

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.services.aiops_service import aiops_service

router = APIRouter()


class AIOpsRequest(BaseModel):
    alert: str
    session_id: str = "default"


@router.post("/aiops/investigate")
async def investigate(request: AIOpsRequest) -> EventSourceResponse:
    """
    触发一次 AIOps 告警调查，流式返回调查过程和最终报告。

    SSE 事件类型：
    - type=plan          计划生成完毕
    - type=step_complete 一个步骤执行完毕
    - type=status        中间状态（replanner 决定 continue）
    - type=report        最终事故报告（Markdown）
    - type=complete      整个流程结束
    - type=error         发生错误

    curl example:
    curl -X POST http://localhost:9900/aiops/investigate \\
      -H "Content-Type: application/json" \\
      -d '{"alert": "payment-service P99 > 3s for 10 minutes", "session_id": "test-1"}' \\
      --no-buffer
    """
    logger.info(
        f"AIOps investigate | session={request.session_id} alert={request.alert!r}"
    )

    async def event_stream():
        async for event in aiops_service.execute(request.alert, request.session_id):
            yield {"event": "message", "data": json.dumps(event)}
            if event.get("type") in ("complete", "error"):
                break

    return EventSourceResponse(event_stream())
