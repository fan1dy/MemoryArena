import os
import uuid
from typing import Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from mem0 import Memory


class Mem0LocalMemorySystem:
    """
    Local deployment of mem0 (https://github.com/mem0ai/mem0).
    Uses the open-source Memory class instead of the cloud MemoryClient.

    Uses Qdrant in local file-based mode by default — no Docker or server needed.
    Pass qdrant_path=None and qdrant_host/qdrant_port to use a running Qdrant server instead.

    For graph memory (enable_graph=True), run Neo4j:
        docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        enable_graph: bool = False,
        llm_model: str = "us/azure/openai/eccn-gpt-5-mini",
        llm_base_url: str = "https://inference-api.nvidia.com/v1",
        embedding_model: str = "us/azure/openai/eccn-text-embedding-3-small",
        embedding_base_url: str = "https://inference-api.nvidia.com/v1",
        embedding_dims: int = 1536,
        collection_name: str = "mem0",
        qdrant_base_path: str = "/tmp/mem0_qdrant",
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        neo4j_url: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        resolved_user_id = user_id if user_id is not None else str(uuid.uuid4())

        if qdrant_host is not None:
            qdrant_config = {
                "collection_name": collection_name,
                "host": qdrant_host,
                "port": qdrant_port or 6333,
                "embedding_model_dims": embedding_dims,
            }
        else:
            # Each user gets an isolated path so concurrent instances don't conflict.
            qdrant_config = {
                "collection_name": collection_name,
                "path": os.path.join(qdrant_base_path, resolved_user_id),
                "on_disk": True,
                "embedding_model_dims": embedding_dims,
            }

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "openai_base_url": llm_base_url,
                    "api_key": api_key,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedding_model,
                    "openai_base_url": embedding_base_url,
                    "api_key": api_key,
                    "embedding_dims": embedding_dims,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": qdrant_config,
            },
        }

        if enable_graph:
            url = neo4j_url or os.getenv("NEO4J_URL", "bolt://localhost:7687")
            username = neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j")
            password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
            config["graph_store"] = {
                "provider": "neo4j",
                "config": {
                    "url": url,
                    "username": username,
                    "password": password,
                },
            }

        self.client = Memory.from_config(config)
        self.user_id = resolved_user_id
        self.enable_graph = enable_graph

    def add_chunk(self, chunk: str):
        self.client.add(
            [
                {"role": "user", "content": chunk},
                {"role": "assistant", "content": "Thanks for the information! I will remember this."},
            ],
            user_id=self.user_id,
        )

    def wrap_user_prompt(self, prompt: str):
        results = self.client.search(prompt.lower(), filters={"user_id": self.user_id})
        memory_context_lines = ["<memory_context>"]
        entries = results if isinstance(results, list) else results.get("results", [])
        for result in entries:
            memory_text = result.get("memory")
            if memory_text:
                categories = result.get("categories") or []
                if categories:
                    memory_context_lines.append(f"{memory_text} (categories: {', '.join(categories)})")
                else:
                    memory_context_lines.append(memory_text)
        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User Prompt: {prompt}")
        return "\n".join(memory_context_lines)
