from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from GraphModule.prompts import query_analyzer_prompt, rewriter_prompt, responder_prompt
from GraphModule.pydantic_models import QueryAnalysis, RewriterResponse
from dotenv import load_dotenv
load_dotenv()

llm = ChatOllama(model="llama3.2", temperature=0)
#llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

query_analyzer_chain = query_analyzer_prompt | llm.with_structured_output(QueryAnalysis)
rewriter_chain = rewriter_prompt | llm.with_structured_output(RewriterResponse)
responder_chain = responder_prompt | llm


# I need some help with insomnia.
# How drinking coffee in the evening can affect my health
# Tell me more about disrupting sleep
# I also watch TikTok before sleep. Will it cause any harm?