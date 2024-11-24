from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from typing import Literal
import os

def format_retrieved_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_model(provider:Literal['openai','google','meta','anthropic']):
    if provider == "openai":
        return ChatOpenAI(temperature=0, model_name="gpt-4o-mini", strict = True)
    # elif provider == "anthropic":
    #     return ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet")
    # elif provider == "google":
    #     return ChatGoogleGenerativeAI(temperature=0, model_name="gemini-1.5-pro-exp-0801")
    # elif provider == "meta":
    #     return ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")
    elif provider == "azure":
        return AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                               openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                               api_key=os.environ.get("AZURE_OPENAI_KEY"),
                               temperature=0)
        