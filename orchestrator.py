import streamlit as st
import base64
import re
import json
from typing import Tuple, Optional, Dict, Any
import time
import os
from dotenv import load_dotenv


from agents.abacus_agent import get_abacus_agent


load_dotenv()

# Initialize the abacus agent
abacus_agent = get_abacus_agent()

def extract_base64_image(response: str) -> Optional[str]:
    """
    Extract base64 image data from agent response.
    Returns the base64 string or None if not found.
    """
    
    if not response:
        return None
        
    
    if len(response) > 100 and response.strip().startswith("data:image") or response.strip().startswith("iVBOR"):
        return response.strip()
    
 
    base64_pattern = r'data:image\/[^;]+;base64,([A-Za-z0-9+/=]+)'
    match = re.search(base64_pattern, response)
    if match:
        return match.group(1)
    
   
    base64_only_pattern = r'([A-Za-z0-9+/=]{100,})'
    match = re.search(base64_only_pattern, response)
    if match:
        return match.group(1)
    
    return None

def display_abacus_response(response: str, trace: Any = None):
    """
    Display the abacus agent response with proper formatting for text and images
    """
    # Extract any base64 image
    base64_image = extract_base64_image(response)
    
    # Clean up the response text by removing any base64 data to make it readable
    if base64_image and len(base64_image) > 100:
        clean_text = re.sub(r'data:image\/[^;]+;base64,[A-Za-z0-9+/=]+', 
                           '[Abacus Visualization]', response)
        clean_text = re.sub(r'[A-Za-z0-9+/=]{100,}', '[Abacus Visualization]', clean_text)
    else:
        clean_text = response
    
 
    st.markdown("### Abacus Explanation")
    st.write(clean_text)
   
    if base64_image:
        st.markdown("### Abacus Visualization")
        
    
        if base64_image.startswith('data:image'):
            st.image(base64_image)
      
        else:
            try:
                
                st.image(f"data:image/png;base64,{base64_image}")
            except Exception:
                
                st.error("Could not display image visualization")

def main():
    st.set_page_config(
        page_title="Abacus Tutor",
        page_icon="🧮",
        layout="wide"
    )
    
    st.title("🧮 Interactive Abacus Tutor")
    
    # Sidebar for options
    st.sidebar.header("Options")
    visualization_enabled = st.sidebar.checkbox("Enable Visualizations", value=True)
    
    # Teaching mode selection
    teaching_mode = st.sidebar.radio(
        "Teaching Mode",
        ["Basic Operations", "Advanced Techniques", "Practice Problems"]
    )
    
    # Main content area
    st.markdown("""
    Welcome to the Abacus Tutor! Ask questions about abacus calculations or follow along with guided examples.
    The tutor will show you step-by-step how to use an abacus for various calculations.
    """)
    
    # Example queries
    st.markdown("### Example Queries")
    example_queries = [
        "Show me how to add 123 + 456 on an abacus",
        "How do I subtract 78 from 125 on an abacus?",
        "Teach me multiplication on an abacus",
        "What are the basic parts of an Indian abacus?"
    ]
    
    # Create columns for example buttons
    cols = st.columns(2)
    example_buttons = {}
    
    for i, query in enumerate(example_queries):
        col_idx = i % 2
        with cols[col_idx]:
            example_buttons[query] = st.button(query)
    
    # Input area
    user_query = st.text_input("Enter your question:", key="user_query")
    
    # Process example button clicks
    for query, clicked in example_buttons.items():
        if clicked:
            user_query = query
            st.session_state.user_query = query
    
    # Process user query when submitted
    if st.button("Submit") or user_query:
        if not user_query:
            st.warning("Please enter a question or select an example.")
            return
            
        with st.spinner("Generating response..."):
            # Call the abacus agent with the query
            response, trace = abacus_agent.handle_query(user_query)
            
            # Display the response
            display_abacus_response(response, trace)
            
            # Optional: Display debugging info in the sidebar if in development
            if os.environ.get("ENVIRONMENT") == "development":
                with st.sidebar.expander("Debug Info"):
                    st.json(trace)
    
    # Add a section for interactive abacus practice if in that mode
    if teaching_mode == "Practice Problems":
        st.markdown("---")
        st.header("Practice Zone")
        
        # Generate some practice problems
        st.markdown("### Try these practice problems:")
        
        practice_cols = st.columns(3)
        with practice_cols[0]:
            if st.button("Addition Practice"):
                with st.spinner("Generating practice problem..."):
                    response, _ = abacus_agent.handle_query("Generate an addition practice problem")
                    st.session_state.practice_problem = response
        
        with practice_cols[1]:
            if st.button("Subtraction Practice"):
                with st.spinner("Generating practice problem..."):
                    response, _ = abacus_agent.handle_query("Generate a subtraction practice problem")
                    st.session_state.practice_problem = response
        
        with practice_cols[2]:
            if st.button("Mixed Practice"):
                with st.spinner("Generating practice problem..."):
                    response, _ = abacus_agent.handle_query("Generate a mixed practice problem")
                    st.session_state.practice_problem = response
        
        # Display the current practice problem if available
        if hasattr(st.session_state, 'practice_problem'):
            st.markdown(st.session_state.practice_problem)
            
            # Answer submission
            user_answer = st.text_input("Your answer:")
            if st.button("Check Answer"):
                with st.spinner("Checking answer..."):
                    response, _ = abacus_agent.handle_query(f"Check if {user_answer} is the correct answer to the practice problem")
                    display_abacus_response(response)

if __name__ == "__main__":
    main()