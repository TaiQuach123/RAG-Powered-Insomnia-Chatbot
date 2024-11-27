from GraphModule.pydantic_models import State

def query_analyzer_route(state: State):
    if state["query_analysis"].route == "retrieve":
        return "rewriter"
    return "responder"

def rewriter_route(state: State):
    if state["rewriter_response"].need_clarification:
        return "ask_human"
    return "retrieve"
