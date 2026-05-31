"""
START → planner → executor → replanner ─┬─ (有 response) → END
                     ↑                   │
                     └───────────────────┘  (plan 不为空，继续循环)
"""

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from loguru import logger

from .executor import executor
from .planner import planner
from .replanner import replanner
from .state import PlanExecuteState

NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"


class AIOpsAgent:
    def __init__(self) -> None:
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        logger.info("AIOpsAgent initialized")

    def _build_graph(self):
        """构建 Plan-Execute-Replan 工作流图。"""
        workflow = StateGraph(PlanExecuteState)

        workflow.add_node(NODE_PLANNER, planner)
        workflow.add_node(NODE_EXECUTOR, executor)
        workflow.add_node(NODE_REPLANNER, replanner)

        workflow.set_entry_point(NODE_PLANNER)
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)

        def should_continue(state: PlanExecuteState) -> str:
            if state.get("response"):
                return END
            if state.get("plan"):
                return NODE_EXECUTOR
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER,
            should_continue,
            {NODE_EXECUTOR: NODE_EXECUTOR, END: END},
        )

        return workflow.compile(checkpointer=self.checkpointer)

    def astream(self, alert: str, session_id: str):
        """
        返回 LangGraph 原始 astream，供 service 层消费。
        AIOps 用的是自定义的 PlanExecuteState，有 4 个字段（input、plan、past_steps、response），每次调查都是全新开始，所以需要完整初始化
        """

        initial_state: PlanExecuteState = {
            "input": alert,
            "plan": [],
            "past_steps": [],
            "response": "",
        }

        config = RunnableConfig(configurable={"thread_id": session_id})

        return (
            self.graph.astream(
                input=initial_state,
                config=config,
                stream_mode="updates",
            ),
            config,
        )

    def get_state(self, config: RunnableConfig):
        return self.graph.get_state(config)


aiops_agent = AIOpsAgent()
