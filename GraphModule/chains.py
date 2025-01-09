from langchain_groq import ChatGroq
from GraphModule.prompts import query_analyzer_prompt, rewriter_prompt, responder_prompt
from GraphModule.pydantic_models import QueryAnalysis, RewriterResponse
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

query_analyzer_chain = query_analyzer_prompt | llm.with_structured_output(QueryAnalysis)
rewriter_chain = rewriter_prompt | llm.with_structured_output(RewriterResponse)
responder_chain = responder_prompt | llm