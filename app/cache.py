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

    def _hash_query(self,query:str)->str:
        """Create hash Of normalized Query"""
        normalized=query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self,query:str)->Optional[str]:
        """get cached response if similar query exists"""
        query_hash=self._hash_query(query)

        #Exact Match
        if query_hash in self.cache:
            return self.cache[query_hash]["response"]
        
        #could add embedding based similarity here
        #for demo ,just use Exact match

        return None
    
    def set(self,query:str,response:str):
        """cache a response"""
        query_hash=self._hash_query(query)
        self.cache[query_hash]={"query":query,"response":response}

    def stats(self)->dict:
        return {"cached_queries":len(self.cache)}

class CachedLLM:
    """llm wrapper with caching"""

    def __init__(self):
        self.llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)
        self.cache=SematicCache()
        self.cache_hits=0
        self.cache_misses=0

    @traceable(name="cached_invoke")
    def invoke(self,query:str)->tuple[str,bool]:
        """invoke with caching.
        Returns :(response,from_cache)"""

        #check cache
        cached=self.cache.get(query)
        if cached:
            self.cache_hits+=1
            return cached,True
        
        #Call LLM
        self.cache_misses+=1
        response=self.llm.invoke(query)
        result=response.content

        #Cache result
        self.cache.set(query,result)

        return result,False
    
    def get_stats(self)->dict:
        total=self.cache_hits+self.cache_misses
        hit_rate=self.cache_hits/total if total>0 else 0
        return {
            "hits":self.cache_hits,
            "misses":self.cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
        }

def demo_caching():
    """Demonstrate Caching"""
    llm=CachedLLM()
    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Python?",  # Cache hit
        "What is python?",  # Cache hit (normalized)
        "What is Rust?",
    ]
    print("\nCaching Demo:\n")

    for query in queries:
        reult,from_cache=llm.invoke(query)
        source ="CACHE" if from_cache else "LLM"
        print(f"[{source}] {query} ->> {result[:30]}.......")
    
    print(f"\n Stats: {llm.get_stats()}")

#=========Token Budgetting==============
