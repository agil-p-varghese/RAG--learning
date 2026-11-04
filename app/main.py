import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langsmith import traceable
from dotenv import load_dotenv
from app.config import get_settings

from app.models import (
    ChatRequest, ChatResponse, HealthResponse, MetricsResponse, ErrorResponse
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector, RequestTimer
from app.agent import ProductionAgent

load_dotenv()

security: SecurityPipeline = None
cache: ResponseCache = None
metrics: MetricsCollector = None
agent: ProductionAgent = None
logger = get_logger()

# ==== LifeSpan (startup/shutdown) ======

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes all components on startup and cleans up on shutdown.
    This is the Modern FastAPI Pattern (replaces @app.on_event)."""

    global security, cache, metrics, agent
    settings = get_settings()

    logger.info("Starting Production API......", extra={"extra_data": {
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "tracing_enabled": settings.langchain_tracing_v2,
    }})

    # Initialize Components
    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = ProductionAgent()

    logger.info("All Components are initialized, Ready to serve Requests !!!!")

    yield  # app is running

    # Shutdown
    logger.info("Shutting down...", extra={"extra_data": metrics.summary})

# ===== Rate limiter setup =====
limiter = Limiter(key_func=get_remote_address)

# ==== FastAPI App ==============
app = FastAPI(
    title="Production LangGraph API",
    description="A production Ready Chat API with security, caching, observability",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter

# ==== Exception Handler ========
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning("Rate limit exceeded", extra={"extra_data": {"client": get_remote_address(request)}})
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )

# =====================================================
# =========== ENDPOINTS ===============================
# =====================================================

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.
    Flow:
    1. Security Check (Injection + PII masking)
    2. Cache lookup
    3. LangGraph agent Invoke (if cache miss)
    4. Output Validation
    5. Cache store
    6. Return response
    """

    with RequestTimer() as timer:
        security_notes = []
        
        # ----- Security check -------------
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning("Request blocked by security", extra={"extra_data": {
                "reason": notes,
                "thread_id": body.thread_id,
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters."
            )
        
        # --- Step 2: Cache lookup ----------------
        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info("Cache Hit!!!", extra={"extra_data": {
                "thread_id": body.thread_id,
            }})
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
            )

        # ========== Step 3: Invoke LangGraph Agent ==============
        try:
            result = agent.invoke(cleaned_message)
        except Exception as e:
            logger.error(f"Agent Invocation failed: {e}", extra={"extra_data": {
                "thread_id": body.thread_id,
                "error": str(e),
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request.",
            )

        response_text = result["response"]
        model_used = result["model_used"]

        # ----- Step 4: Output validation -----------
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # ------ Step 5: Cache store ---------
        cache.set(cleaned_message, validated_response)

    # ------- Step 6: Log and Record Metrics ----------
    input_tokens = int(len(cleaned_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens, 
        output_tokens=output_tokens,
        cache_hit=False,
    )

    if security_notes:
        logger.info("Security notes", extra={"extra_data": {
            "notes": security_notes,
            "thread_id": body.thread_id,
        }})

    logger.info("Request Completed", extra={"extra_data": {
        "thread_id": body.thread_id,
        "model_used": model_used,
        "latency_ms": round(timer.elapsed_ms, 2),
    }})

    return ChatResponse(
        response=validated_response,
        thread_id=body.thread_id,
        model_used=model_used,
        cached=False,
        processing_time_ms=round(timer.elapsed_ms, 2),
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker and Kubernetes."""
    settings = get_settings()

    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status="Healthy" if all_healthy else "degraded",
        environment=settings.app_env,
        checks=checks,
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Metrics for monitoring Dashboard."""
    summary = metrics.summary
    return MetricsResponse(**summary)

@app.get("/cache/stats")
async def cache_stats():
    """Cache performance statistics."""
    return cache.stats