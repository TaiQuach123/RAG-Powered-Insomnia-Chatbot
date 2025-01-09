import functools

from qdrant_client import QdrantClient

from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import torch
from transformers import AutoModel
from FlagEmbedding import BGEM3FlagModel

from RAGModule.retrieve import retrieve_relevant_chunks

from GraphModule.prompts import query_analyzer_prompt, rewriter_prompt, responder_prompt
from GraphModule.pydantic_models import State
from GraphModule.nodes import *
from GraphModule.routes import *

from llm_compiler.utils import *
from llm_compiler.runnables import *
from llm_compiler.prompts import llm_compiler_prompt, joiner_prompt
from llm_compiler.pydantic_models import JoinOutputs

import streamlit as st

torch.set_grad_enabled(False)

from dotenv import load_dotenv
load_dotenv()


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


if 'client' not in st.session_state:
    st.session_state.client = QdrantClient("http://localhost:6333")
if 'jina_embeddings' not in st.session_state:
    st.session_state.jina_embeddings = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(device)
if 'bge_embeddings' not in st.session_state:
    st.session_state.bge_embeddings = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
if 'llm' not in st.session_state:
    st.session_state.llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


retrieve_relevant_chunks = functools.partial(retrieve_relevant_chunks, jina_embedding = st.session_state.jina_embeddings, bge_embedding = st.session_state.bge_embeddings, client = st.session_state.client)

@tool
def retrieve_chunks(query: str) -> str:
    """retrieve_chunks(query="the search query") - This tool retrieves relevant information from a vector store containing insomnia-related data based on the given query"""
    relevant_chunks = retrieve_relevant_chunks(query=query)
    relevant_chunks = format_chunks(relevant_chunks[:3])
    return relevant_chunks

tools = [retrieve_chunks]

query_analyzer_chain = query_analyzer_prompt | st.session_state.llm.with_structured_output(QueryAnalysis)
rewriter_chain = rewriter_prompt | st.session_state.llm.with_structured_output(RewriterResponse)
responder_chain = responder_prompt | st.session_state.llm

planner = create_planner(st.session_state.llm, tools, llm_compiler_prompt)
runnable = joiner_prompt | st.session_state.llm.with_structured_output(JoinOutputs)
joiner_chain = select_recent_messages | runnable

@as_runnable
def plan_and_schedule(state: State):
    messages = state["llm_compiler_messages"]
    tasks = planner.stream(messages)
    # Begin executing the planner immediately
    try:
        tasks = itertools.chain([next(tasks)], tasks)
    except StopIteration:
        # Handle the case where tasks is empty.
        tasks = iter([])
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": messages,
            "tasks": tasks,
        }
    )
    return {"llm_compiler_messages": scheduled_tasks}


def joiner(state: State):
    join_outputs: JoinOutputs = joiner_chain.invoke({"llm_compiler_messages": state["llm_compiler_messages"]})
    join_outputs.should_replan = False ### Temporary not using Replan

    if join_outputs.should_replan:
        return {"llm_compiler_messages":[AIMessage(content = f"Thought: {join_outputs.thought}")] +  [SystemMessage(content = join_outputs.replan_analysis)]}
    else:
        messages = state["llm_compiler_messages"]
        documents = ''
        for msg in messages[::-1]:
            if isinstance(msg, HumanMessage):
                break
            if msg.name == 'join':
                continue
            documents += msg.content + '\n\n'
        
        return {"documents": documents}
    


graph_builder = StateGraph(State)

graph_builder.add_node("query_analyzer", query_analyzer)
graph_builder.add_node("rewriter", rewriter)
graph_builder.add_node("plan_and_schedule", plan_and_schedule)
graph_builder.add_node("joiner", joiner)
graph_builder.add_node("ask_human", ask_human)
graph_builder.add_node("responder", generate_response)

graph_builder.add_edge(START, "query_analyzer")
graph_builder.add_conditional_edges("query_analyzer", query_analyzer_route, {"rewriter": "rewriter", "responder": "responder"})
graph_builder.add_conditional_edges("rewriter", rewriter_route, {"ask_human": "ask_human", "llm_compiler": "plan_and_schedule"})
graph_builder.add_conditional_edges("joiner", compiler_route, {"plan_and_schedule": "plan_and_schedule", "responder": "responder"})
graph_builder.add_edge("ask_human", "rewriter")
graph_builder.add_edge("plan_and_schedule", "joiner")
graph_builder.add_edge("responder", END)

if 'graph' not in st.session_state:
    st.session_state.graph = graph_builder.compile(checkpointer=MemorySaver(), interrupt_before=["ask_human"])

config = {"configurable": {"thread_id": 0}}
is_interrupt = False

st.set_page_config(page_title="Insomnia Chatbot", page_icon="🤖")
st.title("Insomnia Chatbot")


for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)


query = st.chat_input("Enter Your Query")
if query:
    st.session_state.chat_history.append(HumanMessage(content = query))
    with st.chat_message("Human"):
        st.markdown(query)
    

    try:
        if st.session_state.graph.get_state(config).next[0] == 'ask_human':
            is_interrupt = True
            st.session_state.graph.update_state(config, {"messages": [HumanMessage(content = query)]}, as_node="ask_human")
        else:
            is_interrupt = False
    
    except:
        pass

    for event in st.session_state.graph.stream({"messages": [HumanMessage(content = query)]} if not is_interrupt else None, config=config, stream_mode="values"):
        pass
    
    ai_response = st.session_state.graph.get_state(config).values['messages'][-1].content
    st.session_state.chat_history.append(AIMessage(content = ai_response))
    with st.chat_message("AI"):
        st.markdown(st.session_state.graph.get_state(config).values['messages'][-1].content)

    with st.expander("Reference Chunks"):
        print(st.session_state.graph.get_state(config).values)
        documents = st.session_state.graph.get_state(config).values.get('documents', '')
        st.write(documents)