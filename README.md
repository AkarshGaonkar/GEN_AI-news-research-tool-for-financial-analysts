This project is an inspiration after a similar tool I saw online. 
The twist? This is a completely free and fully local alternative, at the cost of minimal accuracy and speed.
This allows for full control over the developement procedure and complete security.

I saw a smimilar tool online that used the openAI api to access their top models. Being a broke college kid doesn't allow for 
that luxury. My next throught, "lets use a free api like deepseek!". The problem, security issues. They use the data we sewnd through the API, THE API IS FREE FOR A REASON!

So this project is fully based on open suource models. The model in use here is a small open source language model from google, and 
most of the other NLP tools in use are from the Hugging Face SDK. Shout out hugging face!

**Analyst buddy: A News Research Tool** 

A free AI-powered news research tool that analyzes news articles and answers questions about them. No API keys required!

**How It Works**

Input: You provide URLs of news articles (up to 3)
Processing: The app downloads and analyzes the content
AI Analysis: Creates a searchable knowledge base from the articles
Q&A: Ask questions and get AI-generated answers with sources

**Libraries Used & Their Functions**

**Core Framework**

streamlit - Creates the web interface with input forms, buttons, and displays

langchain - Framework that connects different AI components together

langchain-community - Additional LangChain tools for HuggingFace integration

**AI & Machine Learning**

transformers - HuggingFace library that loads and runs AI language models

torch - PyTorch framework that powers the neural networks behind the scenes

sentence-transformers - Converts text into numerical vectors for similarity search

**Document Processing**

unstructured - Downloads web pages and extracts readable text from HTML

faiss-cpu - Facebook's library for fast similarity search through large amounts of text

pickle - Python's built-in library to save and load the processed data

**Utilities**

python-dotenv - Loads environment variables (for API keys if needed)

os, time - Built-in Python libraries for file operations and timing

**Technical Workflow**

**Step 1**: Text Extraction
URLs → unstructured → Raw Text

Takes news article URLs

Downloads the web pages

Extracts readable text content

**Step 2**: Text Processing
Raw Text → RecursiveCharacterTextSplitter → Text Chunks

Breaks long articles into smaller, manageable pieces

Each chunk is ~1000 characters with 200-character overlap

This helps the AI focus on relevant sections

**Step 3**: Creating Embeddings
Text Chunks → HuggingFaceEmbeddings → Vector Numbers

Converts each text chunk into a list of numbers (vectors)

Similar text chunks will have similar numbers

Uses the "all-MiniLM-L6-v2" model for this conversion

**Step 4**: Building Search Index
Vector Numbers → FAISS → Searchable Database

Creates a fast-searchable database of all text chunks

FAISS can quickly find the most relevant chunks for any question

**Step 5**: Question Answering
Question → FAISS Search → Relevant Chunks → FLAN-T5 → Answer

Your question gets converted to vectors

FAISS finds the most relevant article chunks

FLAN-T5 model reads those chunks and generates an answer

Sources are provided for transparency

**AI Models Used**

Language Model: Google FLAN-T5

Purpose: Reads text and answers questions

Size: Base model (~250MB) or Small model (~80MB for cloud)

**Why: Specifically trained for question-answering tasks**

Embedding Model: all-MiniLM-L6-v2

Purpose: Converts text into searchable numbers

Size: ~80MB

Why: Fast, accurate, and works well with news content

**Data Flow Diagram
[News URLs]

↓

[Web Scraping]

↓

[Text Chunks]

↓

[Vector Embeddings]

↓

[FAISS Index] → [User Question]

↓

[Relevant Chunks] ← [Search]

↓

[FLAN-T5 Model]

↓

[Generated Answer + Sources]**

System Requirements

Python: 3.8 or higher

RAM: 4GB minimum, 8GB recommended

Storage: ~500MB for first-time model downloads

Internet: Required for downloading models and accessing news URLs

**Quick Start**

copy the repo using the following command:

**git clone "repo link here"**

Install dependencies:

bash

**pip install -r requirements.txt**

Run the app:

**streamlit run news_research_tool.py**


Use Cases

Journalists: Quickly analyze multiple news sources on a topic

Researchers: Extract key information from news articles

Students: Understand complex news stories with Q&A

General Users: Stay informed by asking specific questions about current events


Performance Notes

First run: Downloads models (~500MB) - takes 2-5 minutes

Subsequent runs: Much faster as models are cached

Processing time: 30-60 seconds per article set

GPU support: Automatically uses GPU if available for faster processing


Privacy & Security

Local processing: All AI happens on your machine

No data sent: Your articles and questions stay private

No tracking: No analytics or user data collection


