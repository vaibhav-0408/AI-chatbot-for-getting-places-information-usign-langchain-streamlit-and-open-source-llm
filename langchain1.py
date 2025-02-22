import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def initialize_llm():
    """Initialize the Groq LLM with custom settings"""
    return ChatGroq(
        groq_api_key=groq_api_key,
        model="llama3-70b-8192",
        temperature=0.7,
        max_tokens=1000
    )

def create_prompt_template():
    """Create a structured prompt template for place information"""
    template = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable guide about places in India. For each place, provide detailed information about:
        1. Historical significance
        2. Cultural importance
        3. Notable attractions
        4. Best time to visit
        5. How to reach
        6. Local cuisine
        
        Format the information in a clear, readable way using markdown headings and paragraphs.
        Be specific and highlight unique aspects of each place."""),
        ("human", "Tell me about {place}")
    ])
    return template

def get_place_info(place: str, llm) -> str:
    """Get information about a specific place using the LLM"""
    prompt = create_prompt_template()
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"place": place})
        return response
    except Exception as e:
        return f"Error getting information: {str(e)}"

def display_info_with_style():
    """Add custom CSS styling to the Streamlit app"""
    st.markdown("""
        <style>
        .place-header {
            color: #1E88E5;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .info-section {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Indian Places Guide",
        page_icon="🏛️",
        layout="wide"
    )
    
    # Add custom styling
    display_info_with_style()
    
    # Header
    st.markdown("<h1 style='text-align: center;'>Discover Places in India </h1>", unsafe_allow_html=True)
    
    # Sidebar with additional information
    with st.sidebar:
        st.header("About")
        st.write("""
        This app provides detailed information about places in India.
        Enter any city, historical site, or tourist destination to learn more.
        """)
        
        st.header("Features")
        st.write("""
        - Historical information
        - Cultural significance
        - Tourist attractions
        - Travel tips
        - Local cuisine
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        place_input = st.text_input(
            "Enter a place name",
            placeholder="e.g., Taj Mahal, Varanasi, Hampi...",
            key="place_input"
        )
    
    with col2:
        search_button = st.button("🔍 Get Information", use_container_width=True)
    
    # Initialize LLM
    llm = initialize_llm()
    
    # Process the query
    if search_button and place_input:
        with st.spinner(f"Gathering information about {place_input}..."):
            # Get information
            info = get_place_info(place_input, llm)
            
            # Add to session state
            st.session_state.messages.append({"place": place_input, "info": info})
            
            # Display information
            st.markdown(f"### Information about {place_input}")
            st.markdown(info)
    
    # Display history
    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### Recent Searches")
        
        for msg in reversed(st.session_state.messages[-5:]):  # Show last 5 searches
            with st.expander(f"🏛️ {msg['place']}"):
                st.markdown(msg['info'])

if __name__ == "__main__":
    main()