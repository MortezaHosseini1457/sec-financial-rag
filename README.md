# SEC Financial RAG

A simple Retrieval-Augmented Generation (RAG) system for answering questions about SEC financial filings.

## Project Overview

This project builds a basic financial RAG pipeline using NVIDIA's SEC filings.

The pipeline is:

**SEC EDGAR → Document Parsing → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer**

## Technologies

* Python
* EDGAR / SEC filings
* Unstructured
* LangChain
* BGE Embeddings
* FAISS
* LLM API
* Google Colab

## Current Features

* Retrieve latest 10-K NVIDIA SEC filings
* Parse filings into structured text
* Split documents into chunks
* Generate vector embeddings
* Store embeddings in FAISS
* Retrieve relevant sections based on a question
* Generate answers using an LLM

## Example Questions

* What was NVIDIA's revenue in fiscal 2025?
* What are the main risks NVIDIA faces according to its 10-K?
* What are NVIDIA's major business segments?

## Project Status

🚧 Initial RAG prototype

Future improvements may include source citations, better retrieval, reranking, hybrid search, and support for multiple companies and filings.
