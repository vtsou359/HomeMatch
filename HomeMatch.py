"""
HomeMatch.py - Real Estate Recommendation System

This application helps users find their ideal home based on their preferences.
It uses AI to match buyer preferences with available real estate listings
stored in a vector database, providing personalized recommendations.
"""

# Import necessary libraries
import gradio as gr                          # For creating the web interface
#import pandas as pd                         # (Commented out) For data manipulation
from dotenv import load_dotenv               # For loading environment variables (like API keys)
#from langchain_community.document_loaders import CSVLoader  # (Commented out) For loading CSV data
from langchain_openai.chat_models import ChatOpenAI          # For accessing OpenAI's chat models
from langchain_openai.embeddings import OpenAIEmbeddings     # For creating text embeddings with OpenAI
from langchain_chroma.vectorstores import Chroma             # For vector database operations
from langchain.chains import RetrievalQA                     # For creating retrieval-based QA chains
from langchain.prompts import PromptTemplate                 # For creating structured prompts

# Constants for vector database configuration:
VDB_PATH = 'vdb'                  # Directory where the vector database is stored
VDB_NAME = 'real_estate_listings' # Name of the collection in the vector database

# Load environment variables from .env file (contains API keys):
load_dotenv()

# Initialize OpenAI Models:
# Create embeddings model for converting text to vector representations
embeddings = OpenAIEmbeddings()
# Initialize the chat model with specific parameters:
chat_llm = ChatOpenAI(temperature=0.0,      # 0.0 for deterministic, factual responses
                      model="gpt-4.1",      # Using GPT-4.1 for high-quality responses
                      max_tokens=2000,      # Limit response length
                      max_retries=1)        # Number of retry attempts if API call fails

# Load the Chroma vector database:
# This database contains vector representations of real estate listings
db = Chroma(persist_directory=VDB_PATH,           # Path to the stored database
            embedding_function=embeddings,        # Function to create embeddings
            collection_name=VDB_NAME)             # Name of the collection to use

# Set up the vector database as a retriever:
# MMR (Maximum Marginal Relevance) search balances relevance with diversity in results
vdb = db.as_retriever(search_type="mmr", 
                     search_kwargs={'k': 5,        # Return 5 documents
                                   'fetch_k': 15,   # Consider top 15 documents
                                   'lambda_mult': 0.5})  # Balance between relevance and diversity

def prompt_to_retriever(nhood: str, prefs: str, size: str, cost: str) -> str:
    """
    Creates a formatted prompt for the vector database retriever.

    This function takes user inputs about their preferences and formats them
    into a structured query that will be used to search the vector database
    for relevant real estate listings.

    Args:
        nhood (str): User's description of their ideal neighborhood
        prefs (str): User's general preferences for a house
        size (str): User's preferred house size
        cost (str): User's budget constraints

    Returns:
        str: A formatted prompt string ready to be used for vector database retrieval
    """
    # Create a template with placeholders for user inputs
    template = \
    """
    Description of the ideal neighborhood: {var0}
    Description of buyer's preferences: {var1}
    House size preference: {var2}
    Budget: {var3}
    """
    # Create a prompt template object from the template string
    prompt_template = PromptTemplate.from_template(template)

    # Fill in the template with the user's inputs
    res = prompt_template.format(var0=nhood,
                                var1=prefs,
                                var2=size,
                                var3=cost
                                )
    return res

def prompt_to_llm(nhood: str, prefs: str, size: str, cost: str, context) -> str:
    """
    Creates a formatted prompt for the language model with user preferences and retrieved listings.

    This function takes user inputs and the retrieved real estate listings context,
    and formats them into a structured prompt for the language model. The prompt
    includes instructions for the AI on how to format its response and what information
    to include in the recommendations.

    Args:
        nhood (str): User's description of their ideal neighborhood
        prefs (str): User's general preferences for a house
        size (str): User's preferred house size
        cost (str): User's budget constraints
        context (list): List of document objects containing retrieved real estate listings

    Returns:
        str: A formatted prompt string ready to be sent to the language model
    """
    # Create the base template with instructions for the AI and placeholders for user inputs
    template = \
    """
    You are a helpful real estate recommendation engine that helps buyers find their ideal home based on their preferences.
    According only to the context provided you must answer their query.

    -- Buyer's preferences:
    The user has the following preferences:
        > Description of the ideal neighborhood: {var0}
        > Description of buyer's preferences: {var1}
        > House size preference: {var2}
        > Budget: {var3}

    -- Output format rules:
    > Based on the preferences above and the provided content, recommend a house/houses that meets the buyer's needs.
    > Provide your recommendations strictly in markdown bullet points.
    > Always augment the description of real estate listings (context).
    > The augmentation should personalize the listing without changing factual information.
    > First, provide a summarisation of houses provided in the context (in bullet points). Second provide the final recommendation (recommended house).
    > The final recommendation's descriptions in markdown bullets should be unique, appealing, and tailored to the buyer's preferences.

    -- Provided Real Estate Listings Context:
    """
    # Append each retrieved document to the template
    for idx, doc in enumerate(context):
        template += \
            f"""Document {idx}:
            {doc.page_content}
            ---"""

    # Create a prompt template object from the template string
    prompt_template = PromptTemplate.from_template(template)

    # Fill in the template with the user's inputs
    res = prompt_template.format(
	    var0=nhood,
        var1=prefs,
        var2=size,
        var3=cost
    )
    return res

def process_input(neighborhood, preferences, house_size, house_cost):
    """
    Main processing function that handles the complete recommendation workflow.

    This function orchestrates the entire recommendation process:
    1. Creates a retrieval query from user inputs
    2. Retrieves relevant real estate listings from the vector database
    3. Creates a prompt for the language model with user preferences and retrieved listings
    4. Generates personalized recommendations using the language model

    Args:
        neighborhood (str): User's description of their ideal neighborhood
        preferences (str): User's general preferences for a house
        house_size (str): User's preferred house size
        house_cost (str): User's budget constraints

    Returns:
        str: Formatted recommendations from the language model
    """
    # Step 1 & 2: Retrieve relevant documents based on user's preferences
    # Create a query for the vector database using the user's inputs
    retrieval_query = prompt_to_retriever(neighborhood, preferences, house_size, house_cost)
    # Use the query to retrieve relevant real estate listings from the vector database
    docs = vdb.invoke(input=retrieval_query)

    # Step 3: Create the final prompt for input to the language model
    # Combine user preferences with retrieved listings in a structured prompt
    query = prompt_to_llm(neighborhood, preferences, house_size, house_cost, docs)

    # Step 4: Generate the final answer using the language model
    # Send the prompt to the OpenAI model and get personalized recommendations
    llm_output = chat_llm.invoke(input=query)

    # Return the text content of the language model's response
    return llm_output.content


# Create Gradio web interface for user interaction
# Gradio provides an easy way to create web interfaces for machine learning models
with gr.Blocks() as app:
    # Add a title to the interface
    gr.Markdown("# HomeMatch: Real Estate Recommendation System")

    # Create a two-column layout
    with gr.Row():
        # Left column for user inputs
        with gr.Column():
            # Input field for neighborhood preferences
            neighborhood = gr.Textbox(
                label="Neighborhood",
                placeholder="What is the ideal neighborhood that you would like to live in?",
                lines=3  # Height of the text box
            )
            # Input field for general house preferences
            preferences = gr.Textbox(
                label="Preferences",
                placeholder="What are your personal house preferences? Tell me whatever you imagine your house to be like.",
                lines=3
            )
            # Input field for house size preferences
            house_size = gr.Textbox(
                label="House Size",
                placeholder="What is the size of your ideal house?",
                lines=2
            )
            # Input field for budget constraints
            house_cost = gr.Textbox(
                label="Budget",
                placeholder="What is the house cost you can afford? Is there any limit range for your budget?",
                lines=2
            )
            # Submit button to trigger the recommendation process
            submit_btn = gr.Button("Submit")

        # Right column for displaying results
        with gr.Column():
            # Output area for displaying recommendations in markdown format
            gr.Markdown('## Recommendation Outputs')
            output = gr.Markdown(container=True)

    # Connect the submit button to the process_input function
    # When clicked, it will pass all input values to the function and display the result in the output area
    submit_btn.click(
        fn=process_input,  # Function to call when button is clicked
        inputs=[neighborhood, preferences, house_size, house_cost],  # Input fields to pass to the function
        outputs=output,  # Where to display the function's output,
	    show_progress='full'
    )

# Main entry point of the application
if __name__ == "__main__":
    # Launch the Gradio web interface
    # This will start a local web server and open the interface in a browser
    app.launch()
