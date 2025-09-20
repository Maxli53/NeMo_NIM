#!/usr/bin/env python3
"""
Professional Multi-Agent Discussion System
Main entry point with multiple modes: CLI, UI, MCP, API
Supports GPT-OSS 20B MoE, RAG, and Claude Desktop integration
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse
import logging
from typing import Optional, List, Dict, Any
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import config, ModelProvider
from src.agents.expert import ExpertAgent
from src.agents.consensus import ConsensusAgent
from src.core.vector_db import vector_db_manager
from src.core.moderator import DiscussionModerator
from src.core.session import DiscussionSession, SessionManager
from src.core.model_manager import model_manager
from src.utils.embeddings import embedding_manager
from src.utils.pdf_processor import PDFProcessor

# Initialize rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=config.logging_config.level,
    format=config.logging_config.format,
    handlers=[
        RichHandler(console=console, rich_tracebacks=True),
        logging.FileHandler(config.logging_config.file)
    ]
)
logger = logging.getLogger(__name__)


class MultiAgentSystem:
    """Main system orchestrator"""

    def __init__(self):
        self.session_manager = SessionManager()
        self.model_initialized = False
        self.embeddings_initialized = False

    async def initialize(self):
        """Initialize models and embeddings"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Initialize model manager
            if not self.model_initialized:
                task = progress.add_task("Loading GPT-OSS 20B MoE model...", total=None)
                try:
                    if config.is_local_mode():
                        # Model manager initializes on creation
                        model_info = model_manager.get_model_info()
                        console.print(f"[green]✓ Model loaded: {model_info['model_name']}[/green]")
                        if 'moe_experts' in model_info:
                            console.print(f"  MoE: {model_info['moe_experts']} experts, "
                                        f"{model_info['moe_active_experts']} active")
                    self.model_initialized = True
                except Exception as e:
                    console.print(f"[red]✗ Model initialization failed: {e}[/red]")
                    raise
                progress.remove_task(task)

            # Initialize embeddings
            if not self.embeddings_initialized:
                task = progress.add_task("Loading embedding model...", total=None)
                try:
                    embed_info = embedding_manager.get_model_info()
                    console.print(f"[green]✓ Embeddings loaded: {embed_info['model_name']} "
                                f"(dim: {embed_info['dimension']})[/green]")
                    self.embeddings_initialized = True
                except Exception as e:
                    console.print(f"[red]✗ Embedding initialization failed: {e}[/red]")
                    raise
                progress.remove_task(task)

    async def create_agents(
        self,
        use_knowledge_base: bool = True,
        pdf_paths: Optional[List[str]] = None
    ) -> tuple:
        """Create expert agents and consensus agent"""
        agents = []

        for i, agent_config in enumerate(config.agents_config):
            # Get or create vector DB for agent
            vector_db = None
            if use_knowledge_base:
                vector_db = vector_db_manager.get_or_create_db(agent_config.name)

                # Process PDF if provided
                if pdf_paths and i < len(pdf_paths):
                    await self.process_pdf_for_agent(pdf_paths[i], vector_db, agent_config.name)
                elif agent_config.pdf_path and os.path.exists(agent_config.pdf_path):
                    await self.process_pdf_for_agent(agent_config.pdf_path, vector_db, agent_config.name)

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

        return agents, consensus_agent

    async def process_pdf_for_agent(
        self,
        pdf_path: str,
        vector_db,
        agent_name: str
    ):
        """Process PDF and add to vector database"""
        console.print(f"[yellow]Processing PDF for {agent_name}: {pdf_path}[/yellow]")

        processor = PDFProcessor()
        chunks = processor.chunk_pdf(pdf_path)

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        sources = [chunk["source"] for chunk in chunks]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Embedding {len(texts)} chunks...", total=None)
            result = embedding_manager.embed_batch(texts, show_progress=True)
            progress.remove_task(task)

        # Add to vector DB
        vector_db.add_batch(result.embeddings, texts, sources)
        console.print(f"[green]✓ Added {len(texts)} chunks to {agent_name}'s knowledge base[/green]")

    async def run_discussion(
        self,
        task: str,
        max_rounds: int = 10,
        use_knowledge_base: bool = True,
        pdf_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run a multi-agent discussion"""
        console.rule(f"[bold blue]Starting Discussion[/bold blue]")
        console.print(f"[cyan]Task:[/cyan] {task}")
        console.print(f"[cyan]Max Rounds:[/cyan] {max_rounds}")
        console.print(f"[cyan]Knowledge Base:[/cyan] {'Enabled' if use_knowledge_base else 'Disabled'}")
        console.print()

        # Initialize system if needed
        await self.initialize()

        # Create agents
        agents, consensus_agent = await self.create_agents(use_knowledge_base, pdf_paths)

        # Create session
        session = self.session_manager.create_session(
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

        # Display results
        self.display_results(results, session)

        # Save session
        export_path = session.export_session()
        console.print(f"\n[green]✓ Session saved to: {export_path}[/green]")

        return results

    def display_results(self, results: Dict[str, Any], session: DiscussionSession):
        """Display discussion results in a formatted table"""
        console.rule("[bold green]Discussion Results[/bold green]")

        # Create results table
        table = Table(title="Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Consensus Reached", "✅ Yes" if results['consensus_reached'] else "❌ No")
        table.add_row("Total Rounds", str(results['total_rounds']))
        table.add_row("Total Messages", str(results['total_messages']))
        table.add_row("Avg Novelty", f"{results['avg_novelty']:.1f}/10")
        table.add_row("Avg Feasibility", f"{results['avg_feasibility']:.1f}/10")

        console.print(table)

        # Display final consensus if reached
        if results['consensus_reached']:
            console.rule("[bold yellow]Final Consensus[/bold yellow]")
            for entry in reversed(session.session_log):
                if entry.get('agent') == 'ConsensusAgent':
                    console.print(entry['message'])
                    break


def run_cli_mode(system: MultiAgentSystem):
    """Run in CLI mode"""
    console.print("[bold cyan]AI Multi-Agent Discussion System - CLI Mode[/bold cyan]")
    console.print()

    # Get task from user
    task = console.input("[bold yellow]Enter discussion task:[/bold yellow] ")
    if not task:
        task = "Design a novel approach combining quantum computing with biological neural networks."

    # Get configuration
    max_rounds = console.input("Max rounds (default 10): ")
    max_rounds = int(max_rounds) if max_rounds else 10

    use_kb = console.input("Use knowledge base? (y/n, default y): ")
    use_knowledge_base = use_kb.lower() != 'n'

    # Run discussion
    asyncio.run(system.run_discussion(task, max_rounds, use_knowledge_base))


def run_ui_mode():
    """Run Streamlit UI"""
    import subprocess
    console.print("[bold cyan]Starting Streamlit Dashboard...[/bold cyan]")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py"])


def run_mcp_mode():
    """Run MCP server for Claude Desktop"""
    from src.mcp_server import start_server
    console.print("[bold cyan]Starting MCP Server for Claude Desktop...[/bold cyan]")
    console.print(f"Server will run on http://{config.mcp_config.host}:{config.mcp_config.port}")
    start_server()


def run_api_mode():
    """Run as REST API server"""
    console.print("[bold cyan]Starting API Server...[/bold cyan]")
    import uvicorn
    from src.mcp_server import app
    uvicorn.run(
        app,
        host=config.mcp_config.host,
        port=config.mcp_config.port,
        log_level="info"
    )


async def run_test_mode(system: MultiAgentSystem):
    """Run system test"""
    console.print("[bold cyan]Running System Test...[/bold cyan]")

    test_task = "What are the implications of quantum computing for biological systems?"
    results = await system.run_discussion(test_task, max_rounds=3, use_knowledge_base=False)

    # Verify results
    assert results['total_messages'] > 0
    assert 'session_log' in results
    console.print("[green]✅ Test completed successfully![/green]")
    console.print(f"Generated {results['total_messages']} messages")
    console.print(f"Consensus reached: {results['consensus_reached']}")


@click.command()
@click.option(
    '--mode',
    type=click.Choice(['cli', 'ui', 'mcp', 'api', 'test']),
    default='ui',
    help='Run mode'
)
@click.option('--task', help='Discussion task (CLI mode only)')
@click.option('--rounds', type=int, default=10, help='Maximum discussion rounds')
@click.option('--knowledge/--no-knowledge', default=True, help='Use knowledge base')
@click.option('--pdf', multiple=True, help='PDF paths for agents')
def main(mode: str, task: Optional[str], rounds: int, knowledge: bool, pdf: tuple):
    """
    Professional Multi-Agent Discussion System

    Supports multiple modes:
    - cli: Command-line interface
    - ui: Streamlit dashboard
    - mcp: MCP server for Claude Desktop
    - api: REST API server
    - test: System test
    """
    console.print(f"[bold green]Multi-Agent Discussion System v1.0[/bold green]")
    console.print(f"Mode: {mode.upper()}")
    console.print(f"Model: {config.get_active_model()}")
    console.print(f"Provider: {config.model_config.provider.value}")
    console.print()

    system = MultiAgentSystem()

    if mode == 'cli':
        if task:
            asyncio.run(system.run_discussion(task, rounds, knowledge, list(pdf)))
        else:
            run_cli_mode(system)
    elif mode == 'ui':
        run_ui_mode()
    elif mode == 'mcp':
        run_mcp_mode()
    elif mode == 'api':
        run_api_mode()
    elif mode == 'test':
        asyncio.run(run_test_mode(system))


if __name__ == "__main__":
    main()