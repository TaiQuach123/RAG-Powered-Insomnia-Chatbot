from GraphModule.pydantic_models import State, QueryAnalysis, RewriterResponse
from GraphModule.chains import query_analyzer_chain, rewriter_chain, responder_chain
from RAGModule.utils import format_chunks
from RAGModule.retrieve import retrieve_relevant_chunks
from langchain_core.messages import AIMessage, HumanMessage


def query_analyzer(state: State):
    query_analyzer_response: QueryAnalysis = query_analyzer_chain.invoke({"messages": state["messages"]})
    print("=== Query Analyzer Response: ", query_analyzer_response)
    return {"query_analysis": query_analyzer_response}

def rewriter(state: State):
    rewriter_response: RewriterResponse = rewriter_chain.invoke({"messages": state["messages"]})
    print("=== Rewriter Response: ", rewriter_response)
    if rewriter_response.need_clarification:
        return {"rewriter_response": rewriter_response, "messages": [AIMessage(rewriter_response.response)]}
    return {"rewriter_response": rewriter_response, "llm_compiler_messages": [HumanMessage(rewriter_response.response)]}

def generate_response(state: State):
    response =responder_chain.invoke(input={"messages": state['messages'], "context": state.get('documents', '')})
    print("=== Response: ", response)
    return {"messages": [response]}

def ask_human(state: State):
    pass
