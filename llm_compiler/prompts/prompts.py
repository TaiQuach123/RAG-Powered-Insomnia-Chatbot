from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

with open("./llm_compiler/prompts/llm_compiler_prompt.txt") as f:
    llm_compiler_sys_prompt = f.read()

with open("./llm_compiler/prompts/joiner_prompt.txt") as f:
    joiner_sys_prompt = f.read()

llm_compiler_prompt = ChatPromptTemplate([
    ("system", llm_compiler_sys_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

joiner_prompt = ChatPromptTemplate([
    ("system", joiner_sys_prompt),
    MessagesPlaceholder(variable_name="llm_compiler_messages")
])