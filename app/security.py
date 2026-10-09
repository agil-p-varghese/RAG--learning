"""
Security & PII Handling Techniques
Protecting LLM Application in Production
"""
import re
from typing import Optional
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

#==Input Sanitation==
#to redo tommorwo