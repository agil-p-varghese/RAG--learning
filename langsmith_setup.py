import os
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

#Enable LangSmith Traceability
os.environ["LANGSMITH_TRACING"]="true"

@traceable(name="hf_basic_chaining")
def demo_hf_tracing():
    #initialize the hugging face endpoint
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        temperature=0,
    )


    chat_model=ChatHuggingFace(llm=llm)

    prompt=ChatPromptTemplate.from_template("explain {topic} in one sentence.")
    chain=prompt | chat_model | StrOutputParser()

    print("Running chain with HuggingFace and LangSmith traceablilty..... ")
    result=chain.invoke({"topic":"machine learning"})
    print(f"Result : {result}")

if __name__=="__main__":
    demo_hf_tracing()
