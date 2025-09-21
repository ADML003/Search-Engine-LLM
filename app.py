import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="LangChain Knowledge Agent",
    page_icon="📚",
    layout="wide"
)

# Clear any cached resources to avoid old model references
if st.sidebar.button("🔄 Clear Cache"):
    st.cache_resource.clear()
    st.rerun()

@st.cache_resource
def initialize_tools():
    """Initialize and cache tools for better performance"""
    # Arxiv tool
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    
    # Wikipedia tool
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
    
    return [arxiv, wiki]

def create_agent(api_key, model_choice):
    """Create and return the agent executor"""
    try:
        llm = ChatGroq(
            groq_api_key=api_key, 
            model_name=model_choice,
            streaming=True,
            temperature=0.1
        )
        
        tools = initialize_tools()
        
        # Get the React prompt from hub
        react_prompt = hub.pull("hwchase17/react")
        
        # Create agent using modern approach
        agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return agent_executor
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None

# Main app
st.title("📚 LangChain - Knowledge Assistant")
st.markdown("""
This application uses LangChain agents to search Wikipedia and Arxiv to answer your questions.
The agent can access academic papers and encyclopedia articles to provide comprehensive answers.
""")

# Sidebar for settings
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:", 
    type="password", 
    help="Get your API key from https://console.groq.com/keys"
)

# Model selection with current supported models
model_options = {
    "Llama 3.3 70B (Recommended)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Faster)": "llama-3.1-8b-instant",
    "Qwen 3 32B": "qwen/qwen3-32b"
}

selected_model = st.sidebar.selectbox(
    "Choose Model:",
    options=list(model_options.keys()),
    index=0
)

model_name = model_options[selected_model]

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a knowledge assistant who can search Wikipedia and Arxiv. How can I help you?"}
    ]
    st.rerun()

st.sidebar.markdown("### Available Knowledge Sources")
st.sidebar.markdown("""
- 📚 **Wikipedia**: General encyclopedia articles  
- 📖 **Arxiv**: Academic papers and research
""")

st.sidebar.markdown("### Perfect for")
st.sidebar.markdown("""
- General knowledge questions
- Academic research
- Scientific concepts
- Historical information
- Technical explanations
""")

st.sidebar.markdown(f"### Current Model")
st.sidebar.info(f"Using: {model_name}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a knowledge assistant who can search Wikipedia and Arxiv. How can I help you learn something new?"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input(placeholder="Ask me about any topic..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Process with agent
    if not api_key:
        with st.chat_message("assistant"):
            st.warning("Please enter your Groq API key in the sidebar to continue.")
    else:
        with st.chat_message("assistant"):
            try:
                # Create agent with selected model
                agent_executor = create_agent(api_key, model_name)
                
                if agent_executor:
                    # Create callback handler
                    st_cb = StreamlitCallbackHandler(
                        st.container(), 
                        expand_new_thoughts=False,
                        collapse_completed_thoughts=True
                    )
                    
                    # Execute agent with spinner
                    with st.spinner("Searching knowledge bases..."):
                        response = agent_executor.invoke(
                            {"input": prompt},
                            {"callbacks": [st_cb]}
                        )
                    
                    # Extract and display response
                    if "output" in response:
                        answer = response["output"]
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer
                        })
                        st.write(answer)
                    else:
                        st.error("No output received from agent")
                        
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                
                # Suggest trying a different model if the current one fails
                if "decommissioned" in str(e).lower() or "deprecated" in str(e).lower():
                    st.warning("The selected model may be deprecated. Try selecting a different model from the sidebar.")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Built with")
st.sidebar.markdown("""
- 🦜 LangChain
- 🚀 Streamlit  
- ⚡ Groq Models
- 📚 Wikipedia & Arxiv
""")

st.sidebar.success("✅ All tools operational")
