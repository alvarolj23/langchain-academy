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