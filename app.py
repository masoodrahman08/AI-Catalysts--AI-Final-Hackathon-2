import streamlit as st
import collections
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from crewai import Crew, Process

# Import decoupled agent dependencies
from agents import get_agent_llm, create_triage_agent, create_compiler_agent, create_router_agent, create_automation_agent
from tasks import define_workflow_tasks

st.set_page_config(page_title="AI SOP-to-Action Process Agent", page_icon="🤖", layout="wide")

# --- EXECUTIVE COBALT BLUE THEME framework ---
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; max-width: 95%; }
    .header-banner { background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%); padding: 26px; border-radius: 12px; margin-bottom: 28px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .header-banner h1 { color: #FFFFFF !important; font-weight: 800; font-size: 26px !important; margin:0; text-align: center; }
    .header-banner p { color: #93C5FD !important; margin: 4px 0 12px 0; font-size: 14px; text-align: center; }
    .roster-grid { background: rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 10px; font-size: 13px; color: #F3F4F6; border-left: 4px solid #3B82F6; text-align: center; }
    .premium-card { background-color: #FFFFFF; border: 1px solid #E5E7EB; border-left: 6px solid #2563EB; border-radius: 8px; padding: 22px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); color: #000000 !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-banner">
        <h1>🤖 AI SOP-to-Action Business Process Assistant</h1>
        <p>Engineered by Team: <b>AI-Catalysts</b> | Powered by CrewAI, Gemini 2.5, & Local TF-IDF</p>
        <div class="roster-grid">
            👑 <b>Project Leader:</b> Hafiz Masood Ur Rehman &nbsp;|&nbsp; 👥 <b>Engineers:</b> Fatima Ijaz • Muhammad Aslam • Shakeel Ahmed • Sami Ur Rahman • Muhammad Haroon Jan
        </div>
    </div>
""", unsafe_allow_html=True)

# Initialize System Cache Arrays
if "chunks" not in st.session_state: st.session_state.chunks = []
if "sources" not in st.session_state: st.session_state.sources = []
if "vectorizer" not in st.session_state: st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state: st.session_state.tfidf_matrix = None

# Sidebar Telemetry Display
st.sidebar.markdown("### ⚙️ Core Infrastructure")
st.sidebar.success("⚡ Swarm Framework: CrewAI Live")
st.sidebar.info("📌 Local Matrix Index: ACTIVE")

# Split Column Panels
col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("📄 Ingestion Panel")
    uploaded_files = st.file_uploader("Upload corporate SOP manuals (PDF):", type=["pdf"], accept_multiple_files=True)
    
    if st.button("⚙️ Process & Index Documents") and uploaded_files:
        all_chunks, all_sources = [], []
        with st.spinner("Extracting strings and building sparse local coordinate index..."):
            for uploaded_file in uploaded_files:
                reader = PdfReader(uploaded_file)
                file_text = ""
                for page in reader.pages:
                    file_text += (page.extract_text() or "") + "\n"
                
                # Split strings into clean 500-word tokens
                words = file_text.split()
                chunks = [" ".join(words[i:i+500]) for i in range(0, len(words), 500)]
                for idx, chunk in enumerate(chunks):
                    if chunk.strip():
                        all_chunks.append(chunk)
                        all_sources.append(f"{uploaded_file.name} (Segment {idx+1})")
            
            if all_chunks:
                st.session_state.chunks = all_chunks
                st.session_state.sources = all_sources
                vectorizer = TfidfVectorizer(stop_words='english')
                st.session_state.tfidf_matrix = vectorizer.fit_transform(all_chunks)
                st.session_state.vectorizer = vectorizer
                st.success(f"Indexed {len(all_chunks)} corporate knowledge nodes in volatile memory cache.")
                st.rerun()

with col2:
    st.header("💬 Agent Action Controller")
    incident_description = st.text_area("Log a plain-language operational event exception:", placeholder="e.g., I have received damaged steel coil shipments at Loading Bay 4. What should I do?")
    trigger_workflow = st.button("🚀 Trigger Agent Execution Workflow")
    
    if st.session_state.tfidf_matrix is not None:
        st.info(f"📂 Deployed Framework Status: {len(st.session_state.chunks)} matrix chunks locked in session memory.")
    else:
        st.warning("📥 Deployed Framework Status: Awaiting PDF reference index mapping profiles.")

    if trigger_workflow and incident_description:
        if st.session_state.tfidf_matrix is None:
            st.error("Please process compliance standard manuals on the left first.")
        else:
            with st.spinner("Executing Local Coordinate Search..."):
                query_vec = st.session_state.vectorizer.transform([incident_description])
                similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
                top_indices = np.argsort(similarities)[-3:][::-1]
                
                context_str = ""
                matched_citations = []
                for idx in top_indices:
                    if similarities[idx] > 0.05:
                        context_str += f"Source Segment: {st.session_state.sources[idx]}\nText Content: {st.session_state.chunks[idx]}\n\n"
                        matched_citations.append(st.session_state.sources[idx])
                
                if not context_str.strip():
                    st.error("Anti-Hallucination Block: No relevant compliance text matches found. Halting agent execution.")
                else:
                    st.success("Grounding reference matrix extracted. Initializing Multi-Agent task loop execution tracker...")
                    
                    # Instantiate CrewAI workflow modules
                    shared_llm = get_agent_llm()
                    if shared_llm:
                        triage_agent = create_triage_agent(shared_llm)
                        compiler_agent = create_compiler_agent(shared_llm)
                        router_agent = create_router_agent(shared_llm)
                        automation_agent = create_automation_agent(shared_llm)
                        
                        agent_tasks = define_workflow_tasks(
                            triage_agent, compiler_agent, router_agent, automation_agent, 
                            incident_description, context_str
                        )
                        
                        # Execute sequential multi-agent swarm flow
                        operations_crew = Crew(
                            agents=[triage_agent, compiler_agent, router_agent, automation_agent],
                            tasks=agent_tasks,
                            process=Process.sequential
                        )
                        
                        crew_output = operations_crew.kickoff()
                        
                        st.markdown("---")
                        st.markdown("### 📋 Executive Action Map Output")
                        st.markdown(f'<div class="premium-card">{crew_output}</div>', unsafe_allow_html=True)
                        
                        st.write("")
                        st.markdown("#### 📌 Mathematical Compliance Auditing Citations")
                        for source in matched_citations:
                            st.caption(f"✔ **Verified Grounding Reference Node:** `{source}`")
