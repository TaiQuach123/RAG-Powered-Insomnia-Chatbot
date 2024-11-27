from GraphModule.pydantic_models import State, QueryAnalysis, RewriterResponse
from GraphModule.chains import query_analyzer_chain, rewriter_chain, responder_chain
from RAGModule.utils import format_chunks
from RAGModule.retrieve import retrieve_relevant_chunks
from langchain_core.messages import AIMessage


def query_analyzer(state: State):
    query_analyzer_response: QueryAnalysis = query_analyzer_chain.invoke({"messages": state["messages"]})
    return {"query_analysis": query_analyzer_response}

def rewriter(state: State):
    rewriter_response: RewriterResponse = rewriter_chain.invoke({"messages": state["messages"]})
    if rewriter_response.need_clarification:
        return {"rewriter_response": rewriter_response, "messages": [AIMessage(rewriter_response.response)]}
    return {"rewriter_response": rewriter_response}

def generate_response(state: State):
    response =responder_chain.invoke(input={"messages": state['messages'], "context": format_chunks(state['documents'])})
    return {"messages": [response]}

def ask_human(state: State):
    pass

def retrieve_from_vectorstore(state: State):
    relevant_chunks = retrieve_relevant_chunks(state['rewriter_response'].response)
    return {"documents": relevant_chunks}