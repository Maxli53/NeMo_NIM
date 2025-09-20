import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import config
from src.agents.expert import ExpertAgent
from src.agents.consensus import ConsensusAgent
from src.core.vector_db import FAISSVectorDB, VectorDBManager
from src.core.moderator import DiscussionModerator
from src.core.session import DiscussionSession, SessionManager
from anthropic import AsyncAnthropic


st.set_page_config(
    page_title="AI Expert Discussion",
    page_icon="🤖",
    layout="wide"
)

if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionManager()
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = VectorDBManager()


st.title("🤖 AI Expert Multi-Agent Discussion System")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")

    task = st.text_area(
        "Discussion Task",
        value="Design a novel, scientifically grounded multi-modal search engine.",
        height=100
    )

    max_rounds = st.slider("Maximum Rounds", 1, 20, 10)

    st.subheader("Consensus Thresholds")
    novelty_threshold = st.slider("Novelty Threshold", 1.0, 10.0, 7.0, 0.5)
    feasibility_threshold = st.slider("Feasibility Threshold", 1.0, 10.0, 6.0, 0.5)

    st.subheader("Knowledge Base")
    use_knowledge = st.checkbox("Use Knowledge Base", value=False)

    if use_knowledge:
        st.info("Upload PDFs to populate knowledge bases")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=['pdf'],
            accept_multiple_files=True
        )

    st.markdown("---")
    start_button = st.button("🚀 Start Discussion", type="primary", use_container_width=True)
    export_button = st.button("💾 Export Session", use_container_width=True)


main_col, stats_col = st.columns([2, 1])

with main_col:
    st.header("💬 Discussion Thread")
    conversation_container = st.container()
    with conversation_container:
        conversation_box = st.empty()

with stats_col:
    st.header("📊 Statistics")

    metrics_container = st.container()
    with metrics_container:
        col1, col2 = st.columns(2)
        with col1:
            novelty_metric = st.empty()
        with col2:
            feasibility_metric = st.empty()

    st.subheader("📈 Voting Trends")
    votes_chart = st.empty()

    st.subheader("📚 Citations")
    citations_container = st.container()
    with citations_container:
        citations_box = st.empty()


timeline_container = st.container()
with timeline_container:
    st.header("⏱️ Timeline")
    timeline_box = st.empty()


status_container = st.container()
with status_container:
    status_box = st.empty()


async def run_discussion(task: str, max_rounds: int, use_knowledge: bool):
    client = AsyncAnthropic(api_key=config.api_key) if config.api_key else None

    agents = []
    for agent_config in config.agents_config:
        vector_db = None
        if use_knowledge:
            vector_db = st.session_state.db_manager.get_or_create_db(agent_config.domain)

        agent = ExpertAgent(
            name=agent_config.name,
            domain=agent_config.domain,
            vector_db=vector_db,
            temperature=agent_config.temperature,
            client=client
        )
        agents.append(agent)

    consensus_agent = ConsensusAgent(
        client=client,
        novelty_threshold=novelty_threshold,
        feasibility_threshold=feasibility_threshold
    )

    session = st.session_state.session_manager.create_session(
        task=task,
        agents=agents,
        consensus_agent=consensus_agent,
        max_rounds=max_rounds
    )
    st.session_state.current_session = session

    moderator = DiscussionModerator(max_rounds=max_rounds)
    queue = asyncio.Queue()

    results = await moderator.moderate_discussion(
        agents, consensus_agent, task, queue, session.session_log
    )

    session.mark_complete()
    return results


def update_display(session: DiscussionSession):
    if not session or not session.session_log:
        return

    conversation_html = "<div style='max-height: 500px; overflow-y: auto;'>"
    agent_colors = {
        "PhysicsExpert": "#1f77b4",
        "BiologyExpert": "#2ca02c",
        "AIResearcher": "#d62728",
        "ChemistryExpert": "#ff7f0e",
        "ConsensusAgent": "#9467bd"
    }

    for entry in session.session_log:
        agent = entry['agent']
        message = entry['message']
        color = agent_colors.get(agent, "#000000")
        conversation_html += f"""
        <div style='margin-bottom: 15px; padding: 10px; border-left: 3px solid {color};'>
            <b style='color: {color};'>{agent}</b><br>
            {message}
        </div>
        """

    conversation_html += "</div>"
    conversation_box.markdown(conversation_html, unsafe_allow_html=True)

    stats = session.get_statistics()
    novelty_metric.metric("Novelty Score", f"{stats['avg_novelty']:.1f}/10")
    feasibility_metric.metric("Feasibility Score", f"{stats['avg_feasibility']:.1f}/10")

    novelty_data = []
    feasibility_data = []
    for i, entry in enumerate(session.session_log):
        if entry.get('novelty_score'):
            novelty_data.append({"Round": i+1, "Score": entry['novelty_score'], "Type": "Novelty"})
        if entry.get('feasibility_score'):
            feasibility_data.append({"Round": i+1, "Score": entry['feasibility_score'], "Type": "Feasibility"})

    if novelty_data or feasibility_data:
        df_votes = pd.DataFrame(novelty_data + feasibility_data)
        if not df_votes.empty:
            fig = px.line(df_votes, x="Round", y="Score", color="Type",
                         title="Voting Trends", markers=True)
            fig.add_hline(y=novelty_threshold, line_dash="dash",
                         annotation_text="Novelty Threshold", line_color="blue")
            fig.add_hline(y=feasibility_threshold, line_dash="dash",
                         annotation_text="Feasibility Threshold", line_color="green")
            votes_chart.plotly_chart(fig, use_container_width=True)

    citations_text = []
    for entry in session.session_log:
        if entry.get('citations'):
            citations_text.append(f"**{entry['agent']}:**")
            for citation in entry['citations']:
                citations_text.append(f"- {citation.get('source', 'Unknown source')}")
            citations_text.append("")

    if citations_text:
        citations_box.markdown("\n".join(citations_text))
    else:
        citations_box.info("No citations available (conceptual reasoning mode)")

    timeline_data = []
    for i, entry in enumerate(session.session_log):
        timeline_data.append({
            "Message": i+1,
            "Agent": entry['agent'],
            "Time": entry.get('timestamp', '')
        })

    if timeline_data:
        df_timeline = pd.DataFrame(timeline_data)
        timeline_box.dataframe(df_timeline, use_container_width=True)

    consensus_reached = any(
        e.get('consensus_reached', False)
        for e in session.session_log
        if e['agent'] == 'ConsensusAgent'
    )

    if consensus_reached:
        status_box.success("✅ Consensus Reached!")
    else:
        status_box.info("🔄 Discussion in progress...")


if start_button:
    with st.spinner("Running discussion..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(
            run_discussion(task, max_rounds, use_knowledge)
        )
        loop.close()

    if st.session_state.current_session:
        update_display(st.session_state.current_session)
        st.success("Discussion completed!")


if export_button and st.session_state.current_session:
    filepath = st.session_state.current_session.export_session()
    st.success(f"Session exported to: {filepath}")
    with open(filepath, 'r') as f:
        st.download_button(
            label="Download Session Data",
            data=f.read(),
            file_name=filepath,
            mime="application/json"
        )


if st.session_state.current_session:
    with st.expander("Session Details"):
        st.json(st.session_state.current_session.get_statistics())