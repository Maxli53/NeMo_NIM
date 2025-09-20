"""
MCP (Model Context Protocol) Server for Claude Desktop Integration
Provides REST API and WebSocket support for multi-agent discussion
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import json
import logging
from datetime import datetime
import uvicorn

from src.config import config
from src.agents.expert import ExpertAgent
from src.agents.consensus import ConsensusAgent
from src.core.moderator import DiscussionModerator
from src.core.session import DiscussionSession, SessionManager
from src.core.vector_db import vector_db_manager
from src.utils.pdf_processor import PDFProcessor
from src.utils.embeddings import embedding_manager

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Multi-Agent Discussion MCP Server",
    description="MCP server for Claude Desktop integration with RAG multi-agent discussion",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.mcp_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management
session_manager = SessionManager()
active_sessions = {}


# Pydantic models for requests/responses
class DiscussionRequest(BaseModel):
    task: str = Field(..., description="The discussion task/question")
    max_rounds: int = Field(default=5, description="Maximum discussion rounds")
    pdf_paths: Optional[List[str]] = Field(default=None, description="PDF paths for each agent")
    use_knowledge_base: bool = Field(default=True, description="Whether to use RAG")


class AgentResponse(BaseModel):
    agent: str
    message: str
    citations: List[Dict[str, str]]
    novelty_score: Optional[float]
    feasibility_score: Optional[float]
    timestamp: str


class SessionResponse(BaseModel):
    session_id: str
    task: str
    status: str
    messages: List[AgentResponse]
    consensus_reached: bool
    statistics: Dict[str, Any]


class ToolCall(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]


class ToolResult(BaseModel):
    content: str
    mime_type: str = "application/json"
    success: bool = True
    error: Optional[str] = None


# MCP Tool Registration
MCP_TOOLS = {
    "run_discussion": {
        "description": "Run a multi-agent expert discussion and return structured output",
        "parameters": {
            "task": "The discussion task",
            "max_rounds": "Maximum rounds (default 5)",
            "pdf_paths": "Optional PDF paths for agents"
        }
    },
    "get_session": {
        "description": "Get details of a specific discussion session",
        "parameters": {
            "session_id": "The session ID to retrieve"
        }
    },
    "list_sessions": {
        "description": "List all discussion sessions",
        "parameters": {}
    },
    "ping": {
        "description": "Health check for the MCP server",
        "parameters": {}
    }
}


@app.get("/")
async def root():
    """Root endpoint with server info"""
    return {
        "server": "Multi-Agent Discussion MCP",
        "version": "1.0.0",
        "status": "running",
        "tools": list(MCP_TOOLS.keys())
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.get_active_model(),
            "embedding_model": config.embedding_config.model,
            "agents": len(config.agents_config)
        }
    }


@app.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, request: Dict[str, Any]):
    """Execute an MCP tool"""
    if tool_name not in MCP_TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    try:
        if tool_name == "run_discussion":
            result = await run_discussion_internal(
                task=request.get("task"),
                max_rounds=request.get("max_rounds", 5),
                pdf_paths=request.get("pdf_paths"),
                use_knowledge_base=request.get("use_knowledge_base", True)
            )
        elif tool_name == "get_session":
            result = get_session_internal(request.get("session_id"))
        elif tool_name == "list_sessions":
            result = list_sessions_internal()
        elif tool_name == "ping":
            result = {"status": "ok", "timestamp": datetime.now().isoformat()}
        else:
            raise HTTPException(status_code=400, detail="Invalid tool")

        return ToolResult(
            content=json.dumps(result, indent=2),
            mime_type="application/json",
            success=True
        )

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return ToolResult(
            content="",
            success=False,
            error=str(e)
        )


@app.post(f"{config.mcp_config.api_prefix}/discussion")
async def run_discussion(request: DiscussionRequest):
    """Run a multi-agent discussion"""
    try:
        result = await run_discussion_internal(
            task=request.task,
            max_rounds=request.max_rounds,
            pdf_paths=request.pdf_paths,
            use_knowledge_base=request.use_knowledge_base
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Discussion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_discussion_internal(
    task: str,
    max_rounds: int = 5,
    pdf_paths: Optional[List[str]] = None,
    use_knowledge_base: bool = True
) -> Dict[str, Any]:
    """Internal function to run discussion"""

    # Initialize agents
    agents = []
    for i, agent_config in enumerate(config.agents_config):
        # Get or create vector DB for agent
        vector_db = None
        if use_knowledge_base:
            vector_db = vector_db_manager.get_or_create_db(agent_config.name)

            # Process PDF if provided
            if pdf_paths and i < len(pdf_paths):
                pdf_path = pdf_paths[i]
                await process_pdf_for_agent(pdf_path, vector_db)
            elif agent_config.pdf_path and os.path.exists(agent_config.pdf_path):
                await process_pdf_for_agent(agent_config.pdf_path, vector_db)

        agent = ExpertAgent(
            name=agent_config.name,
            domain=agent_config.domain,
            vector_db=vector_db,
            temperature=agent_config.temperature
        )
        agents.append(agent)

    # Create consensus agent
    consensus_agent = ConsensusAgent(
        novelty_threshold=config.consensus_config.novelty_threshold,
        feasibility_threshold=config.consensus_config.feasibility_threshold
    )

    # Create session
    session = session_manager.create_session(
        task=task,
        agents=agents,
        consensus_agent=consensus_agent,
        max_rounds=max_rounds
    )

    # Run discussion
    moderator = DiscussionModerator(max_rounds=max_rounds)
    queue = asyncio.Queue()
    session_log = []

    results = await moderator.moderate_discussion(
        agents, consensus_agent, task, queue, session_log
    )

    # Update session
    session.session_log = session_log
    session.mark_complete()

    # Format response
    response = {
        "session_id": session.session_id,
        "task": task,
        "consensus_reached": results["consensus_reached"],
        "total_rounds": results["total_rounds"],
        "total_messages": results["total_messages"],
        "avg_novelty": results["avg_novelty"],
        "avg_feasibility": results["avg_feasibility"],
        "messages": format_messages(session_log),
        "final_consensus": get_final_consensus(session_log)
    }

    return response


async def process_pdf_for_agent(pdf_path: str, vector_db):
    """Process PDF and add to vector database"""
    processor = PDFProcessor()
    chunks = processor.chunk_pdf(pdf_path)

    # Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    sources = [chunk["source"] for chunk in chunks]
    embeddings = embedding_manager.embed_batch(texts).embeddings

    # Add to vector DB
    vector_db.add_batch(embeddings, texts, sources)


def format_messages(session_log: List[Dict]) -> List[Dict]:
    """Format session messages for response"""
    formatted = []
    for entry in session_log:
        formatted.append({
            "agent": entry.get("agent"),
            "message": entry.get("message"),
            "citations": entry.get("citations", []),
            "novelty_score": entry.get("novelty_score"),
            "feasibility_score": entry.get("feasibility_score"),
            "timestamp": entry.get("timestamp", datetime.now().isoformat())
        })
    return formatted


def get_final_consensus(session_log: List[Dict]) -> Optional[str]:
    """Extract final consensus message if reached"""
    for entry in reversed(session_log):
        if entry.get("agent") == "ConsensusAgent" and entry.get("consensus_reached"):
            return entry.get("message")
    return None


def get_session_internal(session_id: str) -> Dict[str, Any]:
    """Get session details"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "task": session.task,
        "start_time": session.start_time.isoformat(),
        "end_time": session.end_time.isoformat() if session.end_time else None,
        "statistics": session.get_statistics(),
        "messages": format_messages(session.session_log)
    }


def list_sessions_internal() -> List[Dict[str, Any]]:
    """List all sessions"""
    return session_manager.list_sessions()


@app.websocket("/ws/discussion")
async def websocket_discussion(websocket: WebSocket):
    """WebSocket endpoint for real-time discussion streaming"""
    await websocket.accept()
    session_id = None

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "start_discussion":
                # Start new discussion
                task = data.get("task")
                max_rounds = data.get("max_rounds", 5)

                # Create session ID
                session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Send initial acknowledgment
                await websocket.send_json({
                    "type": "session_started",
                    "session_id": session_id,
                    "task": task
                })

                # Run discussion with streaming
                await stream_discussion(websocket, task, max_rounds, session_id)

            elif data.get("action") == "stop_discussion":
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })


async def stream_discussion(
    websocket: WebSocket,
    task: str,
    max_rounds: int,
    session_id: str
):
    """Stream discussion messages in real-time"""
    # This would implement real-time streaming of agent messages
    # For now, sending periodic updates
    for i in range(max_rounds):
        await asyncio.sleep(1)  # Simulate processing
        await websocket.send_json({
            "type": "agent_message",
            "round": i + 1,
            "agent": f"Agent{i}",
            "message": f"Processing round {i+1}...",
            "timestamp": datetime.now().isoformat()
        })

    await websocket.send_json({
        "type": "discussion_complete",
        "session_id": session_id,
        "consensus_reached": True
    })


def start_server():
    """Start the MCP server"""
    logger.info(f"Starting MCP server on {config.mcp_config.host}:{config.mcp_config.port}")
    uvicorn.run(
        app,
        host=config.mcp_config.host,
        port=config.mcp_config.port,
        log_level="info"
    )


if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    start_server()