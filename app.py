import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, SearxSearchWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, SearxSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="LangChain Search Agent",
    page_icon="🔎",
    layout="wide"
)

@st.cache_resource
def initialize_tools():
    """Initialize and cache tools for better performance"""
    try:
        # Arxiv tool
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        
        # Wikipedia tool
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
        
        # SearxNG search tool - CORRECTED VERSION
        search_wrapper = SearxSearchWrapper(searx_host="https://searx.be")
        search = SearxSearchRun(wrapper=search_wrapper)  # Use 'wrapper', not 'api_wrapper'
        
        return [search, arxiv, wiki]
        
    except Exception as e:
        st.error(f"Error initializing tools: {str(e)}")
        # Fallback to just Wikipedia and Arxiv if SearxNG fails
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
        
        return [arxiv, wiki]

def create_agent(api_key):
    """Create and return the agent executor"""
    try:
        llm = ChatGroq(
            groq_api_key=api_key, 
            model_name="llama3-8b-8192", 
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
st.title("🔎 LangChain - Chat with Search")
st.markdown("""
This application uses LangChain agents to search the web (via SearxNG), Wikipedia, and Arxiv to answer your questions.
The agent can access multiple information sources to provide comprehensive answers.
""")

# Sidebar for settings
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:", 
    type="password", 
    help="Get your API key from https://console.groq.com/keys"
)

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]
    st.rerun()

st.sidebar.markdown("### Available Tools")
st.sidebar.markdown("""
- 🌐 **Web Search**: SearxNG search (free & privacy-focused)
- 📚 **Wikipedia**: Encyclopedia articles  
- 📖 **Arxiv**: Academic papers
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web using SearxNG. How can I help you?"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input(placeholder="What would you like to know?"):
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
                # Create agent
                agent_executor = create_agent(api_key)
                
                if agent_executor:
                    # Create callback handler
                    st_cb = StreamlitCallbackHandler(
                        st.container(), 
                        expand_new_thoughts=False,
                        collapse_completed_thoughts=True
                    )
                    
                    # Execute agent with spinner
                    with st.spinner("Searching and thinking..."):
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
- ⚡ Groq (Llama3)
- 🔍 SearxNG (Free Search)
""")
