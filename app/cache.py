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
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

#======================Helper to initialize the Hugging face chat models==================
def get_hf_chat_model(repo_id:str,temperature:float=0)->ChatHuggingFace:
    """Initialize a Hugging Face endpoint and wraps it in a ChatHuggingFace MODEL"""
    llm=HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text_generation",
        temperature=temperature,
    )
    return ChatHuggingFace(llm=llm)
#===========Model Routing ===============

class ModelRouter:
    """routes queries to appropriate model based on complexity"""
    
    def __init__(self):
        self.cheap_model=get_hf_chat_model("Qwen/Qwen2.5-7B-Instruct",temperature=0)
        self.expensive_model=get_hf_chat_model("meta-llama/Meta-Llama-3-70B-Instruct",temperature=0)
        self.classifier=get_hf_chat_model("Qwen/Qwen2.5-7B-Instruct",temperature=0)

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
            model_name="Qwen/Qwen2.5-7B-Instruct"
            cost_per_1k=0.0015
        else:
            model=self.expensive_model
            model_name="meta-llama/Meta-Llama-3-70B-Instruct"
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
        self.embedder=get_hf_chat_model("Qwen/Qwen2.5-7B-Instruct", temperature=0)

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
        self.llm=get_hf_chat_model("Qwen/Qwen2.5-7B-Instruct", temperature=0)
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

class TokenBudget:
    """Track and limit token usasge"""
    def __init__(self,max_tokens_per_request: int=4000):
        self.max_per_request=max_tokens_per_request
        self.usage={"total_input":0,"total_output":0,"requests":0}

    def estimate_tokens(self,text:str)->int:
        """Rough token estimation (actual would use tiktoken)"""
        return int(len(text.spilt())*1.3)
    
    def check_budget(self,text:str)->tuple[bool,int]:
        """check if request is within budget"""
        tokens=self.estimate_tokens(text)
        return tokens<=self.max_per_request, tokens
    
    def record_usage(self,input_tokens:int,output_tokens:int):
        """Record Token usage"""
        self.usage["total_input"]+=input_tokens
        self.usage["total_output"]+=output_tokens
        self.usage["requests"]+=1
    
    def get_stats(self)->dict:
        return {
            **self.usage,
            "total_tokens":self.usage["total_input"]+self.usage["total_output"],
            "avg_per_request":(
                (self.usage["total_input"] + self.usage["total_output"])
                / max(self.usage["requests"], 1)
            ),
        }

class BudgetedLLM:
    """LLM with token budgeting"""
    def __init__(self,max_tokens:int=4000):
        self.llm=get_hf_chat_model("Qwen/Qwen2.5-7B-Instruct", temperature=0)
        self.budget=TokenBudget(max_tokens_per_request=max_tokens)

    @traceable(name="budgeted_invoke")
    def invoke(self,query: str)->str:
        #check budget
        within_budget,tokens=self.check_budget(query)

        if not within_budget:
            raise ValueError(f"query exceeds token budget: {tokens} > {self.budget.max_per_request}")

        #Execute
        response=self.llm.invoke(query)
        result=response.content

        #Record usage
        output_tokens=self.estimate_tokens(result)
        self.budget.record_usage(tokens,output_tokens)
        return result

    def get_stats(self)->dict:
        return self.budget.get_stats()

def demo_token_budgeting():
    """Demonstrate Token Budgeting"""
    llm=BudgetedLLM(max_tokens=100)
    queries = [
        "What is AI?",  # Within budget
        "Explain " + "very " * 100 + "complex topic",  # Over budget
    ]
    print("\nToken Budgeting Demo:\n")

    for query in queries:
        try:
            result=llm.invoke(query)
            print(f"✅ {query[:40]} .......-->> {result[:30]}..........")
        except ValueError as e:
            print(f"❌ {query[:40]}....-->{e}....")
    
    print(f"\n Usage : {llm.get_stats()}")

if __name__=="__main__":
    demo_model_routing()
    # demo_caching()
    #demo_token_budgeting()

# Production version would:
# 1. Embed the query into a vector
# 2. Search the cache by vector similarity
# 3. Return if similarity > threshold (e.g., 0.95)
