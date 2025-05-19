import streamlit as st
import os
import re
import base64
from typing import Optional
from dotenv import load_dotenv

# Import all agents
try:
    from agents.abacus_agent import get_abacus_agent
    ABACUS_AVAILABLE = True
except ImportError:
    ABACUS_AVAILABLE = False

try:
    from agents.finance_agent import get_finance_agent
    FINANCE_AVAILABLE = True
except ImportError:
    FINANCE_AVAILABLE = False

try:
    from agents.vedic_maths_agent import get_vedic_agent
    VEDIC_AVAILABLE = True
except ImportError:
    VEDIC_AVAILABLE = False

load_dotenv()

# Initialize session state
if 'selected_agent' not in st.session_state:
    st.session_state.selected_agent = 'Finance'

# Initialize agents (only available ones)
@st.cache_resource
def initialize_agents():
    """Initialize available agents and cache them"""
    agents = {}
    
    if FINANCE_AVAILABLE:
        try:
            agents['Finance'] = get_finance_agent()
        except Exception as e:
            st.error(f"Failed to initialize Finance agent: {e}")
    
    if ABACUS_AVAILABLE:
        try:
            agents['Abacus'] = get_abacus_agent()
        except Exception as e:
            st.error(f"Failed to initialize Abacus agent: {e}")
    
    if VEDIC_AVAILABLE:
        try:
            agents['Vedic Maths'] = get_vedic_agent()
        except Exception as e:
            st.error(f"Failed to initialize Vedic Maths agent: {e}")
    
    return agents

def get_agent_examples(agent_type: str) -> list:
    """Return example queries based on selected agent"""
    examples = {
        'Finance': [
            "Explain the basics of stock market investing",
            "How do I calculate compound interest?",
            "What is the difference between debt and equity?",
            "Teach me about financial planning for beginners"
        ],
        'Abacus': [
            "Show me how to add 123 + 456 on an abacus",
            "How do I subtract 78 from 125 on an abacus?",
            "Teach me multiplication on an abacus",
            "What are the basic parts of an Indian abacus?"
        ],
        'Vedic Maths': [
            "Show me the Vedic method for multiplication",
            "How to use the Ekadhikena Purvena sutra?",
            "Teach me fast division using Vedic methods",
            "What are the 16 Vedic math sutras?"
        ]
    }
    return examples.get(agent_type, [])

def display_image_from_base64(base64_string):
    """Display an image from a base64 string in Streamlit"""
    try:
        # If the string already starts with 'data:image', use it directly
        if not base64_string.startswith('data:image'):
            base64_string = f"data:image/png;base64,{base64_string}"
        
        # Create an HTML img tag
        html = f'<img src="{base64_string}" style="max-width:100%;"/>'
        st.markdown(html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to display image: {e}")

def process_response(response):
    """Process the response to extract any base64 images and format the text
    
    Args:
        response: The text response from the agent
        
    Returns:
        Processed text with images displayed if present
    """
    # Check if there are any base64 images embedded in markdown format
    img_pattern = r'!\[.*?\]\((data:image\/[^;]+;base64,[^)]+)\)'
    matches = re.findall(img_pattern, response)
    
    if matches:
        # Split the response by image tags
        parts = re.split(img_pattern, response)
        
        # Display each text part followed by its corresponding image
        for i, part in enumerate(parts):
            if part.strip():
                st.markdown(part)
            
            # Display image after each part (except after the last part)
            if i < len(matches):
                display_image_from_base64(matches[i])
        return ""
    
    # Check for base64 strings without markdown formatting
    base64_pattern = r'(data:image\/[^;]+;base64,[A-Za-z0-9+/=]+)'
    base64_matches = re.findall(base64_pattern, response)
    
    if base64_matches:
        # Remove the base64 strings from the response
        clean_text = re.sub(base64_pattern, '', response)
        st.markdown(clean_text)
        
        # Display the images
        for img_data in base64_matches:
            display_image_from_base64(img_data)
        return ""
    
    # Look for plain base64 strings
    plain_base64_pattern = r'([A-Za-z0-9+/=]{50,})'
    plain_matches = re.findall(plain_base64_pattern, response)
    
    if plain_matches:
        for potential_base64 in plain_matches:
            try:
                # Try to decode it to check if it's valid base64
                base64.b64decode(potential_base64)
                # If it decodes successfully, it might be an image
                display_image_from_base64(potential_base64)
                # Remove it from the response
                response = response.replace(potential_base64, "*[Image displayed above]*")
            except:
                # Not valid base64, continue
                pass
    
    return response

def main():
    st.set_page_config(
        page_title="Educational AI Tutor",
        page_icon="🎓",
        layout="centered"
    )
    
    st.title("🎓 Educational AI Tutor")
    st.markdown("Choose your learning domain and get personalized tutoring!")
    
    # Initialize agents
    agents = initialize_agents()
    
    if not agents:
        st.error("No agents available. Please check your agent implementations.")
        return
    
    # Agent selection
    available_agents = list(agents.keys())
    
    # Create columns for agent selection
    st.markdown("### Choose Your Tutor:")
    agent_cols = st.columns(len(available_agents))
    
    for i, agent_name in enumerate(available_agents):
        with agent_cols[i]:
            if st.button(f"📚 {agent_name}", key=f"agent_{agent_name}"):
                st.session_state.selected_agent = agent_name
                st.rerun()
    
    # Show selected agent
    selected_agent = st.session_state.selected_agent
    if selected_agent not in available_agents:
        selected_agent = available_agents[0]
        st.session_state.selected_agent = selected_agent
    
    st.markdown(f"**Current Tutor:** {selected_agent}")
    
    # Display example queries
    st.markdown("### 💡 Example Questions")
    example_queries = get_agent_examples(selected_agent)
    
    # Create buttons for examples
    for i, query in enumerate(example_queries):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"• {query}")
        with col2:
            if st.button("Try", key=f"try_{i}"):
                st.session_state.current_query = query
                st.rerun()
    
    # Input area
    st.markdown("### 💬 Ask Your Question")
    
    # Get query from session state or use empty string
    default_query = st.session_state.get('current_query', '')
    
    # Text input
    user_query = st.text_area(
        "Enter your question:",
        value=default_query,
        height=100,
        key="query_input"
    )
    
    # Submit button
    if st.button("🚀 Get Answer", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question first!")
            return
        
        # Save the query to session state to preserve it across reloads
        if 'current_query' in st.session_state:
            st.session_state.current_query = user_query
            
        # Process the query with selected agent
        with st.spinner(f"Getting response from {selected_agent} tutor..."):
            try:
                agent = agents[selected_agent]
                response, trace = agent.handle_query(user_query.strip())
                
                # Display the response
                st.markdown("### 📝 Response")
                
                # Process the response for any images and display formatted text
                processed_response = process_response(response)
                if processed_response:  # If there's any leftover text to display
                    st.markdown(processed_response)
                
                # Show debug info if in development mode
                if os.environ.get("ENVIRONMENT") == "development":
                    with st.expander("🔍 Debug Info"):
                        st.json(trace)
                        
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")
                st.info("Please try again with a different question.")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip**: Be specific with your questions for better responses!")

if __name__ == "__main__":
    main()