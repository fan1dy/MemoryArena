import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from letta_client import Letta, LLMConfig, EmbeddingConfig

_DEFAULT_BASE_URL = "http://localhost:8283"
_DEFAULT_MODEL = "us/azure/openai/eccn-gpt-5-mini"
_DEFAULT_EMBEDDING = "us/azure/openai/eccn-text-embedding-3-small"
_DEFAULT_LLM_ENDPOINT = "https://inference-api.nvidia.com/v1"
_DEFAULT_EMBEDDING_ENDPOINT = "https://inference-api.nvidia.com/v1"
_DEFAULT_EMBEDDING_DIMS = 1536


class LettaLocalMemorySystem:
    """
    Local deployment of Letta (https://github.com/letta-ai/letta).
    Connects to a self-hosted Letta server instead of the cloud API.

    Prerequisites:
        pip install letta
        letta server          # starts on http://localhost:8283 by default

    Environment variables (all optional):
        LETTA_BASE_URL           - Letta server URL (default: http://localhost:8283)
        LETTA_MODEL              - LLM model identifier
        LETTA_EMBEDDING          - embedding model identifier
        LETTA_LLM_ENDPOINT       - LLM inference base URL
        LETTA_EMBEDDING_ENDPOINT - embedding inference base URL
        NVIDIA_API_KEY / OPENAI_API_KEY - API key for inference endpoints
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        embedding: Optional[str] = None,
        llm_endpoint: Optional[str] = None,
        embedding_endpoint: Optional[str] = None,
        embedding_dims: int = _DEFAULT_EMBEDDING_DIMS,
    ):
        resolved_base_url = base_url or os.getenv("LETTA_BASE_URL", _DEFAULT_BASE_URL)
        resolved_model = model or os.getenv("LETTA_MODEL", _DEFAULT_MODEL)
        resolved_embedding = embedding or os.getenv("LETTA_EMBEDDING", _DEFAULT_EMBEDDING)
        resolved_llm_endpoint = llm_endpoint or os.getenv("LETTA_LLM_ENDPOINT", _DEFAULT_LLM_ENDPOINT)
        resolved_embedding_endpoint = embedding_endpoint or os.getenv("LETTA_EMBEDDING_ENDPOINT", _DEFAULT_EMBEDDING_ENDPOINT)
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"

        llm_config = LLMConfig(
            model=resolved_model,
            model_endpoint_type="openai",
            model_endpoint=resolved_llm_endpoint,
            context_window=16000,
        )
        embedding_config = EmbeddingConfig(
            embedding_model=resolved_embedding,
            embedding_endpoint_type="openai",
            embedding_endpoint=resolved_embedding_endpoint,
            embedding_dim=embedding_dims,
        )

        self.client = Letta(base_url=resolved_base_url)
        self.agent_state = self.client.agents.create(
            llm_config=llm_config,
            embedding_config=embedding_config,
            memory_blocks=[
                {"label": "human", "value": ""},
                {"label": "persona", "value": "I am a self-improving superintelligence."},
            ],
            tools=[],
        )

    def add_chunk(self, chunk: str, user_id: Optional[str] = None):
        response = self.client.agents.messages.create(
            agent_id=self.agent_state.id,
            input="Remember this:\n" + chunk,
        )

        parsed_messages = []
        if hasattr(response, "messages"):
            for message in response.messages:
                if getattr(message, "message_type", None) == "tool_call_message":
                    tool_calls = getattr(message, "tool_calls", None) or []
                    if not tool_calls and hasattr(message, "tool_call"):
                        tool_calls = [message.tool_call]
                    for tool_call in tool_calls:
                        parsed_messages.append({
                            "type": "tool_call",
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        })
                elif getattr(message, "message_type", None) == "assistant_message":
                    parsed_messages.append({
                        "type": "text",
                        "content": message.content,
                    })

        return parsed_messages if parsed_messages else None

    def wrap_user_prompt(self, prompt: str, user_id: Optional[str] = None):
        response = self.client.agents.messages.create(
            agent_id=self.agent_state.id,
            input=(
                "This is the user's prompt: " + prompt
                + "\n\nRetrieve the most relevant information from your memory and return it in text format."
            ),
        )

        memory_context_lines = ["<memory_context>"]
        for message in response.messages:
            try:
                memory_context_lines.append(message.content)
            except Exception as e:
                print("Error in message:", message)
                continue
        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User: {prompt}")
        return "\n".join(memory_context_lines)
