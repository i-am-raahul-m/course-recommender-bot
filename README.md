# course-recommender-bot
The Course Recommendation Bot is an AI-powered system designed to help users identify relevant self-study resources—such as books and academic courses—based on their natural language queries. The project leverages generative AI and retrieval techniques to match user intents (e.g., “Help me learn Cryptography”) with appropriate entries from a curated knowledge base consisting of .txt and .pdf files containing course syllabi and certification curricula. This solution showcases the application of Retrieval-Augmented Generation (RAG) in the education domain, enabling personalized, context-aware recommendations that align with the learner’s goals.

# Objective:

The goal of the project is to:
•	Parse and preprocess .txt and .pdf documents containing syllabus and curriculum data.
•	Allow users to input free-form questions describing the subject they want to learn.
•	Extract meaningful keywords from the user query using a pretrained Flan-T5 model.
•	Create a vector store of curriculum chunks using embedding models for efficient retrieval.
•	Re-rank retrieved chunks using a keyword match count heuristic.
•	Provide accurate recommendations using the Mistral-7B-Instruct model.
This project empowers learners with fast, relevant suggestions for books and course materials aligned to their goals.

# Tools and Technologies Used:

IDE: Platform	Jupyter Notebook (.ipynb) via Google Colab

Programming Language: Python

Document Loading: TextLoader, PyPDFLoader from langchain.document_loaders

Text Splitting: RecursiveCharacterTextSplitter from langchain.text_splitter

Embeddings: HuggingFaceEmbeddings from langchain.embeddings

Vector Store: Chroma from langchain.vectorstores

Keyword Extraction: google/flan-t5-large via Hugging Face transformers

LLM for Answering: mistralai/Mistral-7B-Instruct-v0.2 via transformers

Libraries Used: transformers, torch, pypdf, langchain, langchain-community, langchain-chroma, langchain-text-splitters, transformers, os, bitsandbytes


