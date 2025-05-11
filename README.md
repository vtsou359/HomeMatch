# HomeMatch: AI-Powered Real Estate Recommendation System

HomeMatch is an intelligent real estate recommendation system that helps buyers find their ideal home based on their preferences. The system uses advanced AI techniques including Retrieval Augmented Generation (RAG) to provide personalized property recommendations.

![HomeMatch](https://img.shields.io/badge/HomeMatch-Real%20Estate%20AI-blue)
![LangChain](https://img.shields.io/badge/LangChain-AI%20Framework-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4.1-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-purple)

## Project Overview

HomeMatch combines vector databases and large language models to create a powerful recommendation engine for real estate listings. Users can input their preferences about neighborhoods, house features, size requirements, and budget constraints, and the system will retrieve and recommend the most suitable properties.

### Key Features

- **Natural Language Input**: Users can describe their preferences in natural language
- **Personalized Recommendations**: AI-generated recommendations tailored to user preferences
- **Semantic Search**: Uses embeddings to find semantically relevant properties
- **Interactive Web Interface**: Easy-to-use Gradio web application

## Technology Stack

- **LangChain**: Framework for building LLM applications
- **OpenAI GPT-4.1**: Large language model for generating recommendations
- **OpenAI Embeddings**: For converting text to vector representations
- **Chroma DB**: Vector database for storing and retrieving embeddings
- **Gradio**: Web interface for user interaction
- **Python**: Programming language
- **Pandas**: Data manipulation and analysis

## Project Structure

```
HomeMatch/
├── HomeMatch.py                              # Main application file with Gradio UI
├── data/
│   ├── real_estate_listings.json       # Generated real estate data in JSON format
│   └── real_estate_listings_formatted.csv  # Formatted data for vector database
├── notebooks/
│   ├── ntb1_DataGeneration.ipynb       # Notebook for generating synthetic data
│   ├── ntbk2_CreatingVDB.ipynb         # Notebook for creating vector database
│   └── ntb3_RAG_System_Proto.ipynb     # Notebook for RAG system prototype
├── vdb/                                # Vector database storage
├── src/                                # Source code modules
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/HomeMatch.git
   cd HomeMatch
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Application

To run the HomeMatch web application:

```
python HomeMatch.py
```

This will start the Gradio web interface, typically accessible at http://127.0.0.1:7860 in your web browser.

### Using the Application

1. Fill in the form fields with your preferences:
   - **Neighborhood**: Describe your ideal neighborhood
   - **Preferences**: Describe your personal house preferences
   - **House Size**: Specify your size requirements
   - **Budget**: Indicate your budget constraints

2. Click the "Submit" button to generate recommendations

3. View the personalized recommendations in the output panel

## Notebook Descriptions

### 1. ntb1_DataGeneration.ipynb

This notebook generates synthetic real estate listings data using OpenAI's GPT-4.1 model. It:
- Creates 50 detailed property listings with various attributes
- Saves the data in both JSON and CSV formats
- Each listing includes neighborhood, price, bedrooms, house size, description, and neighborhood description

### 2. ntbk2_CreatingVDB.ipynb

This notebook creates and tests a vector database using LangChain and Chroma:
- Loads the formatted CSV data from the first notebook
- Uses OpenAI embeddings to convert text data into vector representations
- Creates a Chroma vector database and persists it to the 'vdb' directory
- Tests the vector database using a RAG approach with different retrieval methods

### 3. ntb3_RAG_System_Proto.ipynb

This notebook implements the RAG system prototype:
- Provides a clear introduction to the RAG architecture
- Sets up the necessary components (embeddings, LLM, vector database)
- Defines prompt functions for retrieval and LLM generation
- Demonstrates the workflow with example user inputs
- Shows how to retrieve relevant documents and generate recommendations

## Next Steps and Recommendations

### Potential Improvements

1. **Data Enhancement**:
   - Incorporate real estate data from actual listings
   - Add more features like images, property age, amenities, etc.

2. **Model Improvements**:
   - Fine-tune embeddings specifically for real estate domain
   - Experiment with different retrieval methods and parameters
   - Implement user feedback loop to improve recommendations

3. **User Interface Enhancements**:
   - Add visualization of property locations on a map
   - Include filtering options for more specific searches
   - Implement user accounts and saved preferences

4. **Deployment**:
   - Deploy as a web application using a cloud provider
   - Implement caching for faster responses
   - Add monitoring and analytics

5. **Additional Features**:
   - Price prediction functionality
   - Neighborhood analysis and comparison
   - Integration with real estate APIs for live data

