"""API requests and response models
pydantic models for input validation and response structure"""

from pydantic import BaseModel,Field
from datetime import datetime

class ChatRequest(BaseModel):
    """Incomming chat request"""
    message:str=Field(
        ...,
        min_length=1,
        max_length=10000,
        description="the users message to the agent"
    )

    thread_id:str=Field(
        default="default",
        description="conversation thread id"
    )

class ChatResponse(BaseModel):
    """chat response returned to the client"""
    response:str
    thread_id:str
    model_used:str
    cached:bool=False
    processing_time_ms:float
    timestamp:str =Field(default_factory=lambda :datetime.now(timezone.uti))

class HealthResponse(BaseModel):
    """health check response"""
    status:str="healthy"
    environment:str
    version:str="1.0.0"
    checks:dict={}

class MetricsResponse(BaseModel):
    """metrics endpoint response"""

    total_requests:int
    total_errors:int
    error_rate:str
    avg_latency_ms:float
    cache_hit_rate:str
    total_input_tokens:int
    total_outptut_tokens:int

class ErrorResponse(BaseModel):
    """std error response"""
    error:str
    detail:str |None=None
    request_id:str |None=None