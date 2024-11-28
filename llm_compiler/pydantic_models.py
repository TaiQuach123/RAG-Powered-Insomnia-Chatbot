from typing import Union
from pydantic import BaseModel, Field

class JoinOutputs(BaseModel):
    """A structured output model for an LLM's response generation process based on a user's query and relevant information retrieved from a vector store.
    This model helps determine whether a comprehensive response can be provided or if a replan is necessary."""
    thought: str = Field(
        description=(
            "The reasoning process based on the user's question and the provided context from function messages. "
            "Includes logical steps and connections to determine whether the context is sufficient to generate a comprehensive response."
        )
    )
    should_replan: bool = Field(description=(
            "A boolean flag indicating if a replan is necessary. "
            "True if the provided context is insufficient to answer the user's query effectively, requiring further attempts or additional information."
        )
    )
    replan_analysis: str = Field(description=(
            "If should_replan is True, this should contain an analysis of the previous attempt, identifying gaps or missing information, and recommendations on how to proceed. "
            "If should_replan is False, leave this field empty."
        )
    )