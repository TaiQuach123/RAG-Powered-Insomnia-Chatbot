from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator
from pydantic import BaseModel, Field


class RewriterResponse(BaseModel):
    """Response from the rewriter - a LLM that rewrites the user's query based on provided context (chat history)"""
    response: str = Field(description="The rewritten query or a request for clarification (if the user's query is unclear)")
    need_clarification: bool = Field(description="Indicates if the rewriter requires clarification from the user to effectively rewrite the user's query.")

class QueryAnalysis(BaseModel):
    """Analyzes a query and context to determine the next action for a LLM."""
    route: str = Field(description="Action route based on query analysis. 'answer' if the query and context are sufficient for a direct response, 'retrieve' if additional information is needed from a vector store.")

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    rewriter_response: RewriterResponse
    query_analysis: QueryAnalysis
    documents: str
    llm_compiler_messages: Annotated[List[BaseMessage], operator.add]