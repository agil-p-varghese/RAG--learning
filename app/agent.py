from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langsmith import traceable
import os

from app.config import get_settings
from dotenv import load_dotenv
load_dotenv()

# == Agent State ==
class AgentState(TypedDict):
    """State for the production agent.

    Uses Annotated with add_messages reducer for message accumulation.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str


class ProductionAgent:
    """Production langgraph agent with:

    - Retry on Failure (fallback model)
    - Graceful error handling
    - LangGraph tracing
    """

    def __init__(self):
        settings = get_settings()

        # Primary Hugging Face Endpoint & Chat Wrapper
        primary_endpoint = HuggingFaceEndpoint(
            repo_id=getattr(
                settings, "primary_model", "Qwen/Qwen2.5-Coder-7B-Instruct"
            ),
            task="text-generation",
            max_new_tokens=512,
            temperature=0,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
        self.primary_llm = ChatHuggingFace(llm=primary_endpoint)

        # Fallback Hugging Face Endpoint & Chat Wrapper
        fallback_endpoint = HuggingFaceEndpoint(
            repo_id=getattr(
                settings, "fallback_model", "Qwen/Qwen2.5-Coder-7B-Instruct"
            ),
            task="text-generation",
            max_new_tokens=512,
            temperature=0,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
        self.fallback_llm = ChatHuggingFace(llm=fallback_endpoint)

        self.max_retries = getattr(settings, "max_retries", 1)
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph state machine"""

        def process_message(state: AgentState) -> dict:
            """Try to process the message with the primary model"""
            try:
                response = self.primary_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def try_fallback(state: AgentState) -> dict:
            """Fallback to secondary model"""
            try:
                response = self.fallback_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def handle_error(state: AgentState) -> dict:
            """Return a graceful error message"""
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I'm sorry, I'm having trouble processing your request right now. Please try again later."
                        )
                    )
                ],
                "model_used": "error_handler",
            }

        def route_after_process(state: AgentState) -> str:
            """Decide what to do after primary model attempt"""
            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            """Decide what to do after fallback model attempt"""
            if state.get("error") is None:
                return "done"
            else:
                return "error"

        # Build the graph
        graph = StateGraph(AgentState)
        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "process")
        
        # FIXED: Changed add_conditional_edge to add_conditional_edges
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )
        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"},
        )
        graph.add_edge("error", END)
        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self, message: str) -> dict:
        """Invoke the agent with a user message.

        Returns {"response": str, "model_used": str, "error": str | None}
        """
        result = self.graph.invoke({
            "messages": [HumanMessage(content=message)],
            "error": None,
            "retry_count": 0,
            "model_used": "",
        })

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }