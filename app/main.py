import time
import os
from contextlib import asynccontextmanager

from fastapi import FASTAPI,Request,HTTPException
from fastapi.responses import JSONRespnses
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langsmith import traceable
from dotenv import load_dotenv
from app.config import get_settings

from app.models import (
    ChatRequest,ChatResponse,healthResponse,MetricsResponse,ErrorResponse
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger,MetricsCollector,RequestTimer
from app.agent import ProductionAgent

load_dotenv()

cache: ResponseCache=None
metrics: MetricsCollector=None
agent: ProductionAgent=None
logger=get_logger()

#====LifeSpan (startup/shutdown)======

@asynccontextmanager
async def lifespan(app:FastApi):
    """initailizes all components on startup and clean up on shutdown.
    This is the Modern FastApi Pattern (replaces @app.on_event)."""

    global security,cache,metrics,agent
    settings=get_settings()

    logger.info("Starting Production API......",extra={"extra_data":{
        "environment":settings.app_env,
        "primary_model":settings.primary_model,
        "tracing_enabled":settings.langchain_tracing_v2,

    }})

    #Initialize Components
    security=SecurityPipeline()
    cache=ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics=MetricsCollector()
    agent=ProductionAgent()

    logger.info("All Components are initialized, Ready to serve Requests !!!!")

    yield  #app is running

    #shutdown
    logger.info("Shutting down...",extra={"extra_data":metrics.summary})

#=====Rate limiter setup=====
limiter=Limiter(key_func=get_remote_address)

#==== FastApi App ==============
app=FastAPI(
    title="Production LangGraph API",
    description="A production Ready Chat API with security,caching,observability",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter=limiter

#====Exception Handler========
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request:Request, exc: RateLimitExceeded):
    #there are about 10 lines of code below missing

#=====================================================
#===========ENDPOINTS=================================
#=====================================================

@app.post("/chat",response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")

async def chat(request:Request,body:ChatRequest):
    """
    Main chat endpoint.
    Flow::
    1.Secuirty Check(Injection + PII masking)
    2.cache lookup
    3.LangGraph agent Invoke(if cache miss)
    4.Output Validation
    5.Cache store
    6.return response
    """

    with RequestTimer() as timer:
        security_notes=[]
        #-----security check-------------
        is_allowed,cleaned_message,notes=secuirty.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning("Request blocked by security",extra={"extra_data":{
                "reason":notes,
                "thread_id":body.thread_id,
            }})
            metrics.record_request(latency_ms=0,error=True)
            raise HTTPException(
                status_code=400,
                detail="Yoour Msg was blocked by our security filters"
            )
        
        #---Step 2 : Cache lookup ----------------
        cached_response=cache.get(cleaned_message)
        if cached_response is not None:
            metrics.record_request(latency_ms=0,cache_hit=True)
            logger.info("Cache Hit!!!",extra={"extra_data":{
                "thread_id":body.thread_id,
            }})
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
            )


        #==========step 3: Invoke LangGraph Agent ==============
        try:
            result=agent.invoke(cleaned_message)
        except Exception as  e:
            logger.error("Agent Invokation failed: {e}",extra={"extra_data":{
                "thread_id":body.thread_id,
                "error":str(e),
            }})
            metrics.record_request(latency_ms=0,error=True)
            raise HTTPException(
                status_code=500,
                detail="An error ocurred while processing your request.",
            )

        response_text=result["response"]
        model_used=result["model_used"]

        #-----Step 4:Output validation-----------
        validated_response,output_warnings=security.check_output(response_text)
        security_notes.extend(output_warnings)

        #------Step 5 :cache store---------
        cache.set(cleaned_message,validated_response)

    #-------step 6: log and Record Metrics----------
    input_tokens=int(len(cleaned_message.split())*1.3)
    output_tokens=int(len(validated_response.split())*1.3)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,output_tokens=output_tokens,
        cache_hit=False,
    )

    if security_notes:
        logger.INFO("Security notes",extra={"extra_data":{
            "notes":security_notes,
            "thread_id":body.thread_id,

        }})

    logger.info("Request Completed",extra={"extra_data":{
        "thread_id":body.thread_id,
        "model_used":model_used,
        "latency_ms":round(timer.elapsed_ms,2),
    }})

    return ChatResponse(
        response=validated_response,
        thread_id=body.thread_id,
        model_used=model_used,
        cached=False,
        processing_time_ms=round(timer.elapsed_ms,2),
    )
            