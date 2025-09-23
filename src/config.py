import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class ModelProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"
    GPT_OSS = "gpt_oss"


class EmbeddingProvider(Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    INSTRUCTOR = "instructor"


@dataclass
class ModelConfig:
    provider: ModelProvider = ModelProvider.LOCAL
    model: str = os.getenv("MODEL_NAME", "gpt-oss-20b")  # From environment
    temperature: float = 0.7
    max_tokens: int = 2000
    device: str = "auto"  # auto, cuda, cpu
    quantization: Optional[str] = None  # 8bit, 4bit, None
    moe_experts: int = 128  # Total experts in MoE
    moe_active_experts: int = 32  # Active experts per token

    # API settings (if using API providers)
    api_model: str = "claude-3-sonnet-20240229"
    api_temperature: float = 0.7


@dataclass
class EmbeddingConfig:
    provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    model: str = "all-MiniLM-L6-v2"  # Local embedding model
    api_model: str = "text-embedding-3-small"  # If using OpenAI
    dimension: int = 384  # Dimension for local model
    batch_size: int = 32
    device: str = "cuda"  # cuda or cpu


@dataclass
class AgentConfig:
    name: str
    domain: str
    temperature: float = 0.7
    pdf_path: Optional[str] = None  # Path to agent's PDF
    top_k_retrieval: int = 5
    chunk_overlap: int = 50


@dataclass
class ConsensusConfig:
    novelty_threshold: float = 7.0
    feasibility_threshold: float = 6.0
    temperature: float = 0.2
    consensus_rounds: int = 3  # Rounds before forcing consensus
    voting_weights: Dict[str, float] = field(default_factory=lambda: {
        "novelty": 0.5,
        "feasibility": 0.5
    })


@dataclass
class MCPConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_prefix: str = "/api/v1"


@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    rerank: bool = False
    min_relevance_score: float = 0.5


@dataclass
class UIConfig:
    streamlit_port: int = 8501
    theme: str = "dark"
    auto_refresh: bool = True
    chart_height: int = 400
    enable_export: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/app.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    enable_session_logs: bool = True


@dataclass
class Config:
    # API Keys (optional for local mode)
    api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Model configurations
    model_config: ModelConfig = None
    embedding_config: EmbeddingConfig = None

    # Agent configurations
    agents_config: List[AgentConfig] = None
    consensus_config: ConsensusConfig = None

    # System configurations
    mcp_config: MCPConfig = None
    rag_config: RAGConfig = None
    ui_config: UIConfig = None
    logging_config: LoggingConfig = None

    # Paths
    data_dir: str = "data"
    pdf_dir: str = "data/pdfs"
    index_dir: str = "data/indices"
    session_dir: str = "sessions"

    # Runtime
    max_rounds: int = 10
    async_agents: bool = True
    cache_embeddings: bool = True

    def __init__(self):
        # Load API keys from environment
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Initialize configurations
        self.model_config = ModelConfig()
        self.embedding_config = EmbeddingConfig()
        self.consensus_config = ConsensusConfig()
        self.mcp_config = MCPConfig()
        self.rag_config = RAGConfig()
        self.ui_config = UIConfig()
        self.logging_config = LoggingConfig()

        # Initialize agents with domain-specific PDFs
        self.agents_config = [
            AgentConfig(
                name="PhysicsExpert",
                domain="Physics and Quantum Mechanics",
                pdf_path=os.path.join(self.pdf_dir, "physics.pdf")
            ),
            AgentConfig(
                name="BiologyExpert",
                domain="Biology and Life Sciences",
                pdf_path=os.path.join(self.pdf_dir, "biology.pdf")
            ),
            AgentConfig(
                name="AIResearcher",
                domain="Artificial Intelligence and Machine Learning",
                pdf_path=os.path.join(self.pdf_dir, "ai_ml.pdf")
            ),
            AgentConfig(
                name="ChemistryExpert",
                domain="Chemistry and Materials Science",
                pdf_path=os.path.join(self.pdf_dir, "chemistry.pdf")
            ),
        ]

        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.pdf_dir, self.index_dir, self.session_dir, "logs"]:
            os.makedirs(dir_path, exist_ok=True)

    def is_local_mode(self) -> bool:
        """Check if running in local mode (no API)"""
        return self.model_config.provider in [ModelProvider.LOCAL, ModelProvider.GPT_OSS]

    def get_active_model(self) -> str:
        """Get the active model name based on provider"""
        if self.is_local_mode():
            return self.model_config.model
        return self.model_config.api_model


# Global config instance
config = Config()