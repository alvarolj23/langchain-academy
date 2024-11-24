import os
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import AnyMessage, HumanMessage
import operator
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from typing import Literal
import os

def format_retrieved_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_model(provider:Literal['openai','google','meta','anthropic','azure']):
    if provider == "openai":
        return AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                               openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                               api_key=os.environ.get("AZURE_OPENAI_KEY"),
                               temperature=0)
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

class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

from pydantic import constr, BaseModel, Field, validator
    
# Define the input schema
class ProductMappingInput(BaseModel):
    """
    Input schema for retrieving product ID based on product name and country code.
    """
    product_name: str = Field(..., description="Name of the product to map to a product ID.")
    country_code: str = Field(..., description="Country code for filtering, e.g., 'DE', 'FR'.")



# Define the input schema
class ProductQuestionInput(BaseModel):
    """
    Structure for Product Question Input.
    """
    question: str = Field(..., description="User's question for similarity search.")
    product_id: str = Field(..., description="Product ID to filter the search results.")


# Define the input schema for the HTML tool
class HTMLUpdateInput(BaseModel):
    """
    Input schema for HTML content to be saved in the questionnaire.html file.
    """
    html_content: str = Field(..., description="HTML code to be written to or updated in questionnaire.html")

import os
from dotenv import load_dotenv
import sys

load_dotenv()


from langchain_core.tools import tool
from agent_validators import *

from langchain.agents import tool
# Reuse the constant vector store setup
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
import os

import dotenv
dotenv.load_dotenv()

# Use an Azure OpenAI account with a deployment of an embedding model
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Index name for the Azure Search service
index_name_mapping: str = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")

#os.environ["AZURE_AI_SEARCH_SERVICE_NAME"] = "<YOUR_SEARCH_SERVICE_NAME>"
os.environ["AZURE_AI_SEARCH_INDEX_NAME"] = index_name_mapping
#os.environ["AZURE_AI_SEARCH_API_KEY"] = "<YOUR_API_KEY>"

# Azure Search service name and API key
azure_ai_search_service_name = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
vector_store_password = os.getenv("AZURE_AI_SEARCH_API_KEY")

# Option 2: Use AzureOpenAIEmbeddings with an Azure account
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_deployment,
    openai_api_version=azure_openai_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_openai_api_key,
)

# Specify additional properties for the Azure client such as the following https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core/README.md#configurations
vector_store_mapping: AzureSearch = AzureSearch(
    azure_search_endpoint=azure_ai_search_service_name,
    azure_search_key=vector_store_password,
    index_name=index_name_mapping,
    embedding_function=embeddings.embed_query,
    # Configure max retries for the Azure client
    additional_search_client_options={"retry_total": 4},
) 

# Index name for the Azure Search service
index_name_mapping_lake: str = os.getenv("AZURE_AI_SEARCH_INDEX_NAME_LAKE")

#os.environ["AZURE_AI_SEARCH_SERVICE_NAME"] = "<YOUR_SEARCH_SERVICE_NAME>"
#os.environ["AZURE_AI_SEARCH_INDEX_NAME"] = index_name_mapping_lake

# Specify additional properties for the Azure client such as the following https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core/README.md#configurations
vector_store_document_lake: AzureSearch = AzureSearch(
    azure_search_endpoint=azure_ai_search_service_name,
    azure_search_key=vector_store_password,
    index_name=index_name_mapping_lake,
    embedding_function=embeddings.embed_query,
    # Configure max retries for the Azure client
    additional_search_client_options={"retry_total": 4},
) 
    
# Tool function for mapping product name to product ID with country filter
@tool(args_schema=ProductMappingInput)
def map_product_name_to_id(product_name: str, country_code: str) -> list[dict]:
    """
    Map a product name to its corresponding product ID by performing a similarity search, 
    filtered by the specified country code.
    """
    # Define filter using country code
    country_filter = f"availableInCountries/any(c: c eq '{country_code}')"

    # Perform similarity search with product name and country filter
    search_results = vector_store_mapping.similarity_search(
        query=product_name,
        k=5,  # Number of results to consider for mapping
        search_type="hybrid",
        filters=country_filter,  # Apply country filter
    )

    # Extract product IDs and other metadata
    matched_products = []
    for result in search_results:
        product_info = {
            "itemNumber": result.metadata.get("itemNumber"),
            "product_name": result.metadata.get("productName"),
            "manufacturer": result.metadata.get("manufacturer"),
            "description": result.metadata.get("description"),
        }
        if product_info not in matched_products:
            matched_products.append(product_info)

    return matched_products

# Tool function to perform search
@tool(args_schema=ProductQuestionInput)
def search_products_by_question(question: str, product_id: str) -> list[dict]:
    """
    Search for products based on a similarity match with the user's question, filtered by the specified product ID.
    """
    # Define the filter using product ID
    product_filter = f"productId eq '{product_id}'"

    # Perform similarity search with question
    search_results = vector_store_document_lake.similarity_search(
        query=question,
        k=5,  # Number of results to return
        search_type="hybrid",
        filters=product_filter,
    )

    # Process results
    matched_products = []
    for result in search_results:
        product_info = {
            "content": result.metadata.get("content")
        }
        if product_info not in matched_products:
            matched_products.append(product_info)

    return matched_products

# Tool function to save/update HTML content in questionnaire.html
@tool(args_schema=HTMLUpdateInput)
def save_html_to_file(html_content: str) -> str:
    """
    Save or update the HTML content in a file named 'questionnaire.html'.
    If the file exists, it will be overwritten with the new content.
    """
    file_path = "questionnaire.html"
    
    # Write HTML content to the file
    with open(file_path, "w") as file:
        file.write(html_content)

    return f"The HTML content has been successfully written to {file_path}."

# Tools
tools = [map_product_name_to_id, search_products_by_question, save_html_to_file]
tool_node = ToolNode(tools)

# Set up model
model = get_model('azure')
model = model.bind_tools(tools = tools)


def should_continue(state: MessagesState) -> Literal["tools", "human_feedback"]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "human_feedback"

#The commented part is because it breaks the UI with the input function
def should_continue_with_feedback(state: MessagesState) -> Literal["agent", "end"]:
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message, dict):
        if last_message.get("type","") == 'human':
            return "agent"
    if (isinstance(last_message, HumanMessage)):
        return "agent"
    return "end"


def call_model(state: MessagesState):
    messages = [SystemMessage(content=f"You are helpful assistant in Azelis, a global distributor of specialty chemicals and food ingredients in Belgium (Europe).\nAs reference, today is {datetime.now().strftime('%Y-%m-%d %H:%M, %A')}.\nKeep a friendly, professional tone.\nAvoid verbosity.\nConsiderations:\n- Don´t assume parameters in call functions that it didnt say.\n- MUST NOT force users how to write. Let them write in the way they want.\n- The conversation should be very natural like a secretary talking with a client.\n- Call only ONE tool at a time.")] + state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

def read_human_feedback(state: MessagesState):
    pass


# Build graph
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("human_feedback", read_human_feedback)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"human_feedback": 'human_feedback',
    "tools": "tools"}
)
workflow.add_conditional_edges(
    "human_feedback",
    should_continue_with_feedback,
    {"agent": 'agent',
    "end": END}
)
workflow.add_edge("tools", 'agent')

# Memory
checkpointer = MemorySaver()


app = workflow.compile(checkpointer=checkpointer, 
                       interrupt_before=['human_feedback'])

if __name__ == '__main__':
    while True:
        question = input("Put your question: ")

        for event in app.stream(
            {"messages": [
                HumanMessage(content=question)
                ]},
            config={"configurable": {"thread_id": 42}}
            ):
            if event.get("agent","") == "":
                continue
            else:
                msg = event['agent']['messages'][-1].content
                if msg == '':
                    continue
                else:
                    print(msg)