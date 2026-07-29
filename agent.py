import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, AuthenticationError, APIConnectionError, APIError

from tools import search_internet

load_dotenv()


class ConversationMemory:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def add_message(self, message: Dict[str, Any]):
        self.messages.append(message)

    def get(self) -> List[Dict[str, Any]]:
        return self.messages

    def trim(self, max_messages: int = 20):
        if len(self.messages) <= max_messages:
            return

        system_messages = [m for m in self.messages if m.get("role") == "system"]
        other_messages = [m for m in self.messages if m.get("role") != "system"]

        keep_count = max_messages - len(system_messages)
        self.messages = system_messages + other_messages[-keep_count:]


class RealtimeAgent:
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Create a .env file and add:\n"
                "OPENAI_API_KEY=your_api_key_here"
            )

        self.client = OpenAI(api_key=api_key)
        self.memory = ConversationMemory()

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_internet",
                    "description": (
                        "Search the internet for real-time/latest information. "
                        "Use this tool for recent facts, current events, prices, news, "
                        "or whenever verification is needed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to look up online."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        self.system_prompt = (
            "You are a helpful AI assistant with real-time internet access.\n"
            "Use web search when the user asks about latest, recent, current, live, "
            "news, prices, versions, events, or facts that may have changed.\n"
            "When you use search results, cite sources with title and link.\n"
            "If the user only greets you or asks a simple non-current question, "
            "you may answer normally without searching."
        )

        self.memory.add("system", self.system_prompt)

    def _call_llm(self):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.memory.get(),
            tools=self.tools,
            tool_choice="auto",
            temperature=0.3,
        )
        return response.choices[0].message

    def _execute_tool(self, tool_call) -> str:
        function_name = tool_call.function.name

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid tool arguments"})

        if function_name == "search_internet":
            query = arguments.get("query", "")
            results = search_internet(query, max_results=5)
            return json.dumps(results, ensure_ascii=False)

        return json.dumps({"error": f"Unknown tool: {function_name}"})

    def process(self, user_input: str) -> str:
        self.memory.add("user", user_input)
        self.memory.trim(max_messages=20)

        try:
            while True:
                message = self._call_llm()

                if message.tool_calls:
                    assistant_tool_message = {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in message.tool_calls
                        ],
                    }

                    self.memory.add_message(assistant_tool_message)

                    for tool_call in message.tool_calls:
                        tool_result = self._execute_tool(tool_call)

                        self.memory.add_message(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result,
                            }
                        )

                    continue

                final_answer = message.content or "I could not generate a response."
                self.memory.add("assistant", final_answer)
                return final_answer

        except RateLimitError:
            return (
                "OpenAI API quota error.\n\n"
                "Your API key has no available credits or billing is not enabled.\n"
                "Fix it here:\n"
                "https://platform.openai.com/settings/organization/billing/overview\n\n"
                "Note: ChatGPT Plus does not include API credits."
            )

        except AuthenticationError:
            return (
                "Authentication error.\n\n"
                "Your OPENAI_API_KEY is missing, invalid, or expired.\n"
                "Create a new key here:\n"
                "https://platform.openai.com/api-keys\n"
            )

        except APIConnectionError:
            return (
                "Network connection error.\n\n"
                "Please check your internet connection and try again."
            )

        except APIError as e:
            return f"OpenAI API error:\n{str(e)}"

        except Exception as e:
            return f"Unexpected error:\n{str(e)}"