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
