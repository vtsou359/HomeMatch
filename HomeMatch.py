import gradio as gr
#import pandas as pd
from dotenv import load_dotenv
#from langchain_community.document_loaders import CSVLoader
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Constants:
VDB_PATH = 'vdb'
VDB_NAME = 'real_estate_listings'

# Environment variables:
load_dotenv()

# OpenAI Models:
embeddings = OpenAIEmbeddings()
chat_llm = ChatOpenAI(temperature=0.0,
                      model="gpt-4.1",
                      max_tokens=2000,
                      max_retries=1)

# Loading Chroma DB:
db = Chroma(persist_directory=VDB_PATH,
            embedding_function=embeddings,
            collection_name=VDB_NAME)

# VectorDB loading:
vdb = db.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 15, 'lambda_mult': 0.5})

def prompt_to_retriever(nhood: str, prefs: str, size: str, cost: str) -> str:
    template = \
    """
    Description of the ideal neighborhood: {var0}
    Description of buyer's preferences: {var1}
    House size preference: {var2}
    Budget: {var3}
    """
    prompt_template = PromptTemplate.from_template(template)
    res = prompt_template.format(var0=nhood,
                                var1=prefs,
                                var2=size,
                                var3=cost
                                )
    return res

def prompt_to_llm(nhood: str, prefs: str, size: str, cost: str, context) -> str:
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
    for idx, doc in enumerate(context):
        template += \
            f"""Document {idx}:
            {doc.page_content}
            ---"""

    prompt_template = PromptTemplate.from_template(template)
    res = prompt_template.format(var0=nhood,
                                var1=prefs,
                                var2=size,
                                var3=cost)
    return res

def process_input(neighborhood, preferences, house_size, house_cost):
    # Retrieve relevant documents based on user's preferences
    retrieval_query = prompt_to_retriever(neighborhood, preferences, house_size, house_cost)
    docs = vdb.invoke(input=retrieval_query)

    # Create the final prompt for input to LLM
    query = prompt_to_llm(neighborhood, preferences, house_size, house_cost, docs)

    # Generate the final answer using the LLM
    llm_output = chat_llm.invoke(input=query)

    return llm_output.content

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# HomeMatch: Real Estate Recommendation System")

    with gr.Row():
        with gr.Column():
            neighborhood = gr.Textbox(
                label="Neighborhood",
                placeholder="What is the ideal neighborhood that you would like to live in?",
                lines=3
            )
            preferences = gr.Textbox(
                label="Preferences",
                placeholder="What are your personal house preferences? Tell me whatever you imagine your house to be like.",
                lines=3
            )
            house_size = gr.Textbox(
                label="House Size",
                placeholder="What is the size of your ideal house?",
                lines=2
            )
            house_cost = gr.Textbox(
                label="Budget",
                placeholder="What is the house cost you can afford? Is there any limit range for your budget?",
                lines=2
            )
            submit_btn = gr.Button("Submit")

        with gr.Column():
            output = gr.Markdown()

    submit_btn.click(
        fn=process_input,
        inputs=[neighborhood, preferences, house_size, house_cost],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
