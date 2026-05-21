"""
Streamlit UI for Research Assistant
Clean, modern interface matching the design mockup
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Add project directory to path (works from any location)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import your existing system
try:
    from main import execute_routed_query, memory_manager, chat_history
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Make sure app_ui.py is in the same directory as main.py, tools.py, and memory_manager.py")
    st.stop()

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    /* Modern Dark Theme - Main App */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 100%);
    }
    
    /* Sidebar - Clean Dark */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f1f1f 0%, #111111 100%);
        padding: 1.5rem 1rem;
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Sidebar text - Force white */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #a3a3a3 !important;
    }
    
    /* User messages - Light Dark (RIGHT) */
    .user-message {
        background: linear-gradient(135deg, #2a2a2a 0%, #333333 100%);
        padding: 14px 18px;
        border-radius: 20px;
        margin: 10px 0;
        max-width: 70%;
        float: right;
        clear: both;
        color: #e5e5e5;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }
    
    /* Assistant messages - Dark (LEFT) */
    .assistant-message {
        background: linear-gradient(135deg, #3a3a3a 0%, #2a2a2a 100%);
        padding: 14px 18px;
        border-radius: 20px;
        margin: 10px 0;
        max-width: 70%;
        float: left;
        clear: both;
        color: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
    }
    
    .message-container::after {
        content: "";
        display: table;
        clear: both;
    }
    
    /* Input and buttons */
    .stTextInput input {
        border: 2px solid #555555;
        border-radius: 10px;
        padding: 10px;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #3a3a3a 0%, #1f1f1f 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #4a4a4a 0%, #2a2a2a 100%);
        transform: translateY(-1px);
    }
    
    /* Progress indicators */
    .stProgress > div > div {
        background-color: #555555;
    }
    
    /* Info/Success boxes */
    .stInfo {
        background-color: #2a2a2a;
        color: #d4d4d4;
        border-left: 4px solid #555555;
    }
    
    .stSuccess {
        background-color: #1a2e1a;
        color: #86efac;
        border-left: 4px solid #4ade80;
    }
    
    /* Hide branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Title */
    .main-title {
        text-align: center;
        color: #f5f5f5;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #a3a3a3;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============ SESSION STATE INITIALIZATION ============
if "messages" not in st.session_state:
    st.session_state.messages = []

if "action_log" not in st.session_state:
    st.session_state.action_log = []

if "show_traces" not in st.session_state:
    st.session_state.show_traces = False

if "model_name" not in st.session_state:
    st.session_state.model_name = "Claude 3.5 Sonnet"

# ============ SIDEBAR ============
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.write("---")
    
    # Model selection
    st.subheader("Model")
    model_option = st.selectbox(
        "Choose your model",
        ["Claude 3.5 Sonnet", "Claude 3.5 Haiku", "Claude 3 Opus"]
    )
    st.session_state.model_name = model_option
    
    st.write("---")
    
    # Show traces
    st.subheader("Debug")
    show_traces = st.checkbox("Show Traces", value=st.session_state.show_traces)
    st.session_state.show_traces = show_traces
    
    st.write("---")
    
    # Memory stats
    st.subheader("📊 Memory Stats")
    try:
        stats = memory_manager.get_stats()
        st.write(f"**Papers:** {stats.get('papers', 0)}")
        st.write(f"**Conversations:** {stats.get('conversations', 0)}")
    except:
        st.write("Stats unavailable")
    
    st.write("---")
    
    # Clear button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.action_log = []
        st.rerun()
    
    st.write("")
    st.write("")
    st.caption("Research Assistant v1.0")

# ============ MAIN CHAT AREA ============
st.markdown('<div class="main-title">🤖 Research Assistant - Defense Test</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive AI Assistant for Academic Research</div>', unsafe_allow_html=True)

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(
                f'<div class="message-container"><div class="user-message">{content}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="message-container"><div class="assistant-message">{content}</div></div>',
                unsafe_allow_html=True
            )
    
    # Show traces if enabled
    if st.session_state.show_traces and st.session_state.action_log:
        with st.expander("🔍 Recent Traces", expanded=False):
            for trace in st.session_state.action_log[-5:]:
                st.code(trace, language=None)

# ============ INPUT AREA ============
st.markdown("---")

# Create input form at bottom
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "message",
            placeholder="Type your message...",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_button = st.form_submit_button("Send", use_container_width=True)

# ============ PROCESS INPUT ============
if submit_button and user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Log action
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.action_log.append(f"[{timestamp}] User: {user_input[:50]}...")
    
    # Get response from your existing system with progress indicators
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Show appropriate status message based on query type
        if "literature review" in user_input.lower() or "synthesis" in user_input.lower() or "generate" in user_input.lower():
            status_placeholder.info("📚 Generating literature review... This may take 30-60 seconds.")
            progress_bar.progress(10)
        elif "find" in user_input.lower() and "connection" in user_input.lower():
            status_placeholder.info("🔍 Analyzing paper connections... This may take 10-20 seconds.")
            progress_bar.progress(15)
        elif "semantic" in user_input.lower() or "bridge" in user_input.lower():
            status_placeholder.info("🧬 Performing semantic analysis... This may take 15-30 seconds.")
            progress_bar.progress(15)
        elif "export" in user_input.lower() or "bibtex" in user_input.lower():
            status_placeholder.info("📎 Preparing BibTeX export...")
            progress_bar.progress(20)
        elif "search" in user_input.lower() or "find" in user_input.lower():
            status_placeholder.info("🔍 Searching for papers...")
            progress_bar.progress(25)
        else:
            status_placeholder.info("🤔 Processing your request...")
            progress_bar.progress(30)
        
        progress_bar.progress(50)  # Halfway through
        response = execute_routed_query(user_input)
        progress_bar.progress(100)  # Complete
        
        status_placeholder.success("✅ Response generated!")
        import time
        time.sleep(0.5)  # Show success briefly
        status_placeholder.empty()  # Clear status message
        progress_bar.empty()  # Clear progress bar
        
        # Extract clean output from various response formats
        if isinstance(response, dict):
            # Try to get output field
            assistant_reply = response.get('output', None)
            
            # If output is a list of dicts (LangChain format)
            if isinstance(assistant_reply, list):
                # Extract text from list of response objects
                text_parts = []
                for item in assistant_reply:
                    if isinstance(item, dict):
                        if 'text' in item:
                            text_parts.append(item['text'])
                        elif 'content' in item:
                            text_parts.append(item['content'])
                assistant_reply = '\n\n'.join(text_parts) if text_parts else str(assistant_reply)
            
            # If still no good output, stringify the whole response
            if not assistant_reply or assistant_reply == '':
                assistant_reply = str(response)
        else:
            assistant_reply = str(response)
        
        # Truncate very long responses for UI display
        if len(assistant_reply) > 10000:
            assistant_reply = assistant_reply[:10000] + "\n\n... (Response truncated for display. Full output in terminal.)"
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_reply
        })
        
        # Log action
        st.session_state.action_log.append(f"[{timestamp}] Assistant: Response generated ({len(assistant_reply)} chars)")
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg
        })
        st.session_state.action_log.append(f"[{timestamp}] Error: {str(e)}")
        status_placeholder.empty()
        progress_bar.empty()
    
    # Rerun to update chat display
    st.rerun()

# ============ WELCOME MESSAGE ============
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 3rem;'>
        <p>👋 Welcome! I'm your research assistant.</p>
        <p>I can help you with:</p>
        <ul style='list-style: none; padding: 0;'>
            <li>🔍 <strong>Search Papers:</strong> ArXiv, OpenAlex, or your local database</li>
            <li>🧬 <strong>Analyze Connections:</strong> Find semantic links between papers</li>
            <li>🌉 <strong>Discover Bridges:</strong> Identify cross-domain research opportunities</li>
            <li>📝 <strong>Generate Reviews:</strong> Citation-enforced literature synthesis</li>
            <li>📎 <strong>Export Citations:</strong> BibTeX format for LaTeX/reference managers</li>
            <li>💾 <strong>Save Results:</strong> Export findings to files</li>
        </ul>
        <p><strong>Try asking:</strong></p>
        <p style='font-size: 0.9rem; color: #888;'>
            "Find new papers about transformers"<br>
            "What papers do I have about BERT?"<br>
            "Find semantic connections in my papers"<br>
            "Generate a literature review on attention mechanisms"<br>
            "Export papers to BibTeX"
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============ FOOTER INFO ============
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.caption(f"📊 Papers in DB: {memory_manager.get_stats().get('papers', 0)}")
with col2:
    st.caption(f"🧠 Model: {st.session_state.model_name}")