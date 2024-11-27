from typing import Union
from pydantic import BaseModel, Field

class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response to the user's question."""
    thought: str = Field(
        description="The chain of thought reasoning, including logical steps and connections between the query and provided context."
    )
    should_replan: bool = Field(description="A boolean flag indicating whether a replan is necessary.")
    final_response: str = Field(description=("The final response to the user's query. "
                                             "If should_replan is True, this should be an analysis of the previous attempts and recommendations on what needs to be fixed. "
                                             "If should_replan is False, this should be a comprehensive answer to the user's query, addressing all aspects of the query effectively based on provided context."))