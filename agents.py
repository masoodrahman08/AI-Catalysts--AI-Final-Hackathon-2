# Initialize the zero-drift Gemini engine model for the swarm workers
def get_agent_llm():
    # Check st.secrets first (for Streamlit Cloud deployment)
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        # Fallback to local OS environment variable (for local machine testing)
        api_key = os.environ.get("GEMINI_API_KEY", "")
        
    if not api_key:
        st.error("🔒 Security Defect: GEMINI_API_KEY not found in secrets vault or local environment variables.")
        return None
        
    # Force temperature=0.0 to lock token probability drift variations
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=api_key,
        temperature=0.0
    )
