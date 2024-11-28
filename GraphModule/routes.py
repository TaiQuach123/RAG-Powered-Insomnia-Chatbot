from GraphModule.pydantic_models import State
from langchain_core.messages import SystemMessage

def query_analyzer_route(state: State):
    if state["query_analysis"].route == "retrieve":
        return "rewriter"
    return "responder"

def rewriter_route(state: State):
    if state["rewriter_response"].need_clarification:
        return "ask_human"
    return "llm_compiler"

def compiler_route(state: State):
    messages = state["llm_compiler_messages"]
    if isinstance(messages[-1], SystemMessage):
        return "plan_and_schedule"
    else:
        return "responder"