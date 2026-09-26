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

## Installation
pip install -r requirements.txt


## Current Features

* Retrieve latest 10-K NVIDIA SEC filings
* Parse filings into structured text
* Split documents into chunks
* Generate vector embeddings
* Store embeddings in FAISS
* Retrieve relevant sections based on a question
* Generate answers using an LLM

## Example Question

* What are the main risks NVIDIA faces according to its 10-K?
answer:
Based on the provided context, the main risks NVIDIA faces are:

- **Risks related to business trends influenced by climate change concerns:** NVIDIA's business could be negatively impacted by concerns around the high absolute energy requirements of its GPUs, despite their more energy-efficient design and operation relative to alternative computing platforms. (Source: "We also face risks related to business trends that may be influenced by climate change concerns...").
- **Risks related to business investments or acquisitions:** NVIDIA may not be able to realize the potential benefits of business investments or acquisitions, and may not be able to successfully integrate acquired companies, which could hurt its ability to grow its business, develop new products, or sell its products. (Source: "We may not be able to realize the potential benefits of business investments or acquisitions, and we may not be able to successfully integrate acquired companies, which could hurt our ability to grow our business, develop new products or sell our products.").
- **Risks related to government mandates and regulations:** The USG has already imposed license conditions that limit the ability of foreign firms to create and offer large-scale GPU clusters, and may impose additional conditions such as requiring chip tracking and throttling mechanisms. Such mandates could introduce system vulnerabilities, expose NVIDIA to significant risk and potential liability, negatively impact demand for its products, and could have a material impact on its business. Even draft bills can negatively impact business (e.g., China's government publicly questioned whether H20 products have built-in vulnerabilities, discouraging customers from purchasing). (Source: "For example, the USG already imposed license conditions that limit the ability of foreign firms to create and offer as a service large-scale GPU clusters...").
- **Risk related to competition:** Competition could adversely impact NVIDIA’s market share and financial results. (Source: "Competition could adversely impact our market share and financial results.").

The context also mentions that the Company entered into multi-year cloud service agreements and offers standalone software solutions, but it does not identify these as primary risks; it states the business models or strategies may not be successful, but that is not listed as a main risk in the same way.

## Project Status

🚧 Initial RAG prototype

Future improvements may include source citations, better retrieval, reranking, hybrid search, and support for multiple companies and filings.
