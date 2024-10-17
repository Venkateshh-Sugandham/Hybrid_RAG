# Hybrid_RAG

An end-to-end Document Query Hybrid Retrieval-Augmented Generation (RAG) solution using Langchain and Pinecone DB. This system combines the strengths of both retrieval and generation to provide accurate and contextualized answers from a document corpus.

## Technologies Used
- **Python**: The core programming language for building the RAG solution.
- **Langchain**: Manages the retrieval and generation process, enabling seamless interaction between the document corpus and the generative AI.
- **Pinecone DB**: Used as a vector database for efficient document embedding retrieval.
- **OpenAI API**: Powers the generative AI component.
- **Pydantic**: Used for data validation and serialization.
- **Streamlit**: Provides a simple and interactive UI for querying documents.

## Installation

### Step 1: Clone this repository
```
git clone https://github.com/Venkateshh-Sugandham/Hybrid_RAG.git
cd Hybrid_RAG
```
### Step 2: Create a Virtual Environment
```
python3 -m venv venv
```
### Step 3: Activate the Virtual Environment
```
venv\Scripts\activate
```
### Step 4: Install the required dependencies
```
pip install -r requirements.txt
```
### Step 5: Set Up Environment Variables
Create a .env file in the root directory and add your API keys for Pinecone and OpenAI:
```
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```
### Step 6: Initialize Pinecone Index
You need to initialize the Pinecone index before running the application. You can do this using the following Python script:
```
python initialize_pinecone_index.py
```
### Step 7: Run the Application
```
streamlit run app.py
```
This will launch the Streamlit app in your browser. You can then upload documents, input queries, and retrieve contextualized answers based on your document corpus.
