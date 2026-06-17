"""
Anthropic Client - routes through an OpenAI-compatible endpoint (NVIDIA NIM).
Accepts OpenAI-format messages and tools; no format conversion needed.
"""

import os
import json
from typing import List, Dict, Optional

from .base_client import BaseModelClient, ModelResponse, ToolCall

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")


class AnthropicClient(BaseModelClient):
    def __init__(self, model_name: str = "claude-sonnet-4-20250514", api_key: str = None, base_url: str = None):
        super().__init__(model_name)

        api_key = (
            api_key
            or os.environ.get("NVIDIA_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("NVIDIA_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY environment variable not set")

        base_url = (
            base_url
            or os.environ.get("ANTHROPIC_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://inference-api.nvidia.com/v1"
        )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 8192
    ) -> ModelResponse:
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools

            response = self.client.chat.completions.create(**kwargs)

            if hasattr(response, "usage") and response.usage:
                self.total_input_tokens += response.usage.prompt_tokens or 0
                self.total_output_tokens += response.usage.completion_tokens or 0

            msg = response.choices[0].message
            content = msg.content

            tool_calls = None
            if msg.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                    )
                    for tc in msg.tool_calls
                ]

            try:
                raw_response_dict = response.model_dump()
            except Exception:
                raw_response_dict = {"error": "Could not serialize response"}

            return ModelResponse(content=content, tool_calls=tool_calls, raw_response=raw_response_dict)

        except Exception as e:
            print(f"API error: {e}")
            raise

    def format_tool_result(self, tool_call_id: str, result: str, name: str = None) -> Dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }

    def format_assistant_tool_calls(self, tool_calls: List[ToolCall]) -> Dict:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ],
        }

    def get_usage_stats(self) -> Dict:
        input_cost_per_1m = 3.0
        output_cost_per_1m = 15.0
        total_cost = (
            self.total_input_tokens * input_cost_per_1m / 1_000_000
            + self.total_output_tokens * output_cost_per_1m / 1_000_000
        )
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": total_cost,
        }
