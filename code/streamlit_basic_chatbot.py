import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import MessagesPlaceholder

# Initialize the conversation chain
llm = OllamaLLM(model="gemma3:4b")  # or another suitable model

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain = prompt | llm

# Store chat histories for different session IDs
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

def get_message_history(session_id):
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = ChatMessageHistory()
    return st.session_state.chat_histories[session_id]

conversation_with_memory = RunnableWithMessageHistory(
    chain,
    get_message_history,  # Function to get history for a session
    input_messages_key="input",
    history_messages_key="history"
)

st.title("LangChain Chatbot")
session_id = st.text_input("Session ID:", "default")

# Display chat messages from history
if session_id in st.session_state.chat_histories:
    history = st.session_state.chat_histories[session_id].messages
    for message in history:
        role = "Assistant" if message.type == "ai" else "You"
        with st.chat_message(role.lower()):
            st.write(message.content)

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation_with_memory.invoke(
                {"input": user_input}, 
                config={"configurable": {"session_id": session_id}}
            )
            st.write(response)
