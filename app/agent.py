""" some first lines are missing .may be it belong to some comments or import smts"""

from typing import Optional
from typing_extensions import TypedDict,Annotated
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from langsmith import traceable

from app.config import get_settings

#== Agent State ==
class AgentState(TypedDict):
    """
    State for the production agent.
    Uses Annotated with add_messages reducer for message accumulation.
    """
    messages:Annotated[list[BaseMessage],add_messages]
    error:Optional[str]
    retry_count:int
    model_used:str

class ProductionAgent:
    """Production langgraph agent with
    -Retry on Failure(fall back model)
    -Graceful error handling
    -Langgraph tracing
    """

    def __init__(self):
        settings=get_settings()
        self.primary_llm=ChatOpenAI(
            model=settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0,#we handle retry ourself
        )
        self.fallback_llm=ChatOpenAI(
            model=settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0,#we handle retry ourself
        )

        self.max_retries=settings.max_retries
        self.graph=self._build_graph()
    
    def _build_graph(self):
        """build the langgraph state machine"""
        def process_message(state: AgentState)->dict:
            """try to process the msg with primary model"""
            try:
                response=self.primary_llm.invoke(state["messages"])
                return {
                    "messages":[response],
                    "error":None,
                    "model_used":"primary",
                }
            except Exception as e:
                return {
                    "error":str(e),
                    "retry_count":state["retry_count"]+1,
                    "model_used":"",

                }
        
        def try_fallback(state:AgentState)->dict:
            """fallback to secondary model"""
            try:
                response=self.primary_llm.invoke(state["messages"])
                return {
                    "messages":[response],
                    "error":None,
                    "model_used":"fallback",
                }
            except Exception as e:
                return {
                    "error":str(e),
                    "retry_count":state["retry_count"]+1,
                    "model_used":"",
                }
        
        def handle_error(state:AgentState)->dict:
            """return a gracefull error msg"""
            return {
                "messages":[
                    AIMessage(content=(
                        "im sorry im having troublle in processing your request right now ====="
                    ))
                ],
                "model_used":"error_handler",
            }
        def route_after_process(state:AgentState)->str:
            """decide what to do after primary model attempt"""
            if state.get("error") is None:
                return "done"
            elif state["retry_count"]<self.max_retries:
                return "fallback"
            else:
                return "error"
        
        def route_after_fallback(state:AgentState)->str:
            """decide what to do after fallback model attempt"""
            if state.get("error") is None:
                return "done"
            else:
                return "error"
        
        # Build the graph
        graph=StateGraph(AgentState)
        graph.add_node("process",process_message)
        graph.add_node("fallback",try_fallback)
        graph.add_node("error",handle_error)

        graph.add_edge(START,"process")
        graph.add_conditional_edge(
            "process",
            route_after_process,
            {"done":END,"fallback":"fallback","error":"error"},
        )
        graph.add_conditional_edge(
            "fallback",
            route_after_fallback,
            {"done":END,"error":"error"},
        )
        graph.add_edge("error",END)
        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self,message:str)->dict:
        """invoke the agent with user message.
        returns {"reponse":str,"model_used":str,"error":str|None}
        """
        result=self.graph.invoke({
            "messages":[HumanMessage(content=message)],
            "error":None,
            "retry_count":0,
            "model_used":"",
        })

        return {
            "reponse":result["messages"][-1].content,
            "model_used":result.get("model_used","unknown"),
            "error":result.get("error"),
        }
            