"""
cost optimisation Patterns.
Reducing llm cost in production.
"""
import hashlib
import json
from typing import Optional,Callable
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_cores.prompts import ChatPromptTemplate
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

#===========Model Routing ===============

class ModelRouter:
    """routes queries to appropriate model based on complexity"""
    
    def __init__(self):
        self.cheap_model=ChatOpenAI(model="gpt-4o-mini",temperature=0)
        self.expensive_model=ChatOpenAI(model="gpt-4o",temperature=0)
        self.classifier=ChatOpenAI(model="gpt-4o-mini",temperature=0)

    def classify_complexity(self,query:str)->str:
        """Classify query based on complexity"""
        prompt=ChatPromptTemplate.from_templates(
            """
            Classify this query's complexity as 'simple' or 'complex'.

            Simple: Basic facts, short answers, simple calculations
            Complex: Analysis, reasoning, creative tasks, multi-step problems

            Query: {query}

            Respond with only: simple or complex 
            """
        )

        response=self.classifier.invoke(prompt.format(query=query))
        return response.content.strip().lower()

    @traceable(name="routed_query")
    def invoke(self,query:str)->tuple[str,str,float]:
        """
        route and invoke query.
        Returns:(response,model_used,estimated_cost)"""

        complexity=self.classify_complexity(query)