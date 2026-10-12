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

        if complexity=="simple":
            model=self.cheap_model
            model_name="gpt-4o-mini"
            cost_per_1k=0.0015
        else:
            model=self.expensive_model
            model_name="gpt-4o"
            cost_per_1k=0.0025

        response=model.invoke(query)

        #estimate cost ---rough
        tokens=len(query.split()) * 1.3
        estimated_cost=(tokens/1000)*cost_per_1k

        return response.content,model_name,estimated_cost

def demo_model_routing():
    """demonstrate model routing"""
    router=ModelRouter()
    queries = [
        "What is 2 + 2?",  # Simple
        "Analyze the economic implications of AI on the job market.",  # Complex
        "What color is the sky?",  # Simple
    ]

    print("Model Routing Demo:\n")

    total_cost=0
    for query in queries:
        result,model,cost=router.invoke(query)
        total_cost+=cost
        print(f"Query : {query[:50]} .....")
        print(f"Model : {model}")
        print(f"estimated_cost : ${cost:.6f}")
        print(f"Response : {result[:60]}............")

        print(f"total_estimated_cost : ${total_cost:.6f}")
#==========sematic cahing=====================

class SematicCache:
    """cache responses sematic similarity matching """
    def __init__(self,similarity_threshold:float=0.9):
        self.cache={}
        self.threshold=similarity_threshold
        self.embedder=ChatOpenAI(model="gpt-40-mini",temperature=0)

        