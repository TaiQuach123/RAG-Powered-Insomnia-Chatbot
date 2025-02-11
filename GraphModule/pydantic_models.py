from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator
from pydantic import BaseModel, Field


class RewriterResponse(BaseModel):
    """Indicates whether the user's query is clear enough for retrieving information from a vector store or if further clarification is necessary."""
    need_clarification: bool = Field(description=("Determines if the user's query is unclear and requires clarification. "
                                                  "For instance, if the user asks 'I need help with insomnia' or 'tell me about ...', "
                                                  " the request lacks specificity, and we don't know what the user is asking for exactly.\n" 
                                                  "- True if the query is unclear and requires clarification, False if the query is clear and can proceed with rewriting."))
    response: str = Field(description=("If need_clarification is 'True', this field contains a polite request for the user to clarify their query. "
                                       "If need_clarification is 'False', this field contains the rewritten query that is ready for retrieving information from a vector store."))

class QueryAnalysis(BaseModel):
    """Analyzes the user's latest query along with the provided context (if available) to determine the next appropriate action."""
    route: str = Field(description=("The determined action to take based on the analysis of the query and context.\n"
                                    "- 'answer' if the context and query provide enough information for a direct response.\n"
                                    "- 'retrieve' if additional information is required and needs to be fetched from a vector store."))

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    rewriter_response: RewriterResponse
    query_analysis: QueryAnalysis
    documents: str
    llm_compiler_messages: Annotated[List[BaseMessage], operator.add]