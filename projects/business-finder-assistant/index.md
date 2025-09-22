---
layout: subpage
hero: /img/projects/business-finder-assistant/business-finder-assistant.webp
---

<title>Large Language Model (LLM)–Powered Business Finder Assistant Using LangChain, ChromaDB, and Retrieval-Augmented Generation (RAG)</title>

By John Ivan Diaz

A simple Large Language Model-powered chatbot that helps users find information about a business. It uses People Data Labs 2019 Global Company Dataset from Kaggle, the Sentence Transformers All-MiniLM-L6-v2 embedding model, and the Meta Llama 3.2 1B Instruct LLM from HuggingFace. While the chatbot is simple and retrieves only a small fraction of the dataset, its goal is to demonstrate LangChain, ChromaDB, and Retrieval-Augmented Generation (RAG) for LLM orchestration, vector storage, and retrieval.

<tag>Large Language Models (LLMs)</tag>
<tag>LangChain</tag>
<tag>ChromaDB</tag>
<tag>Retrieval-Augmented Generation (RAG)</tag>
<tag>Generative AI</tag>

<a href="https://github.com/ivanintelligence/business-finder-assistant" class="arrow-link">See source code</a>

<hr class="hr-custom">
<br>

<h1>Discovery Phase</h1>

<h2>Use Case Definition</h2>

Natural Language Processing (NLP) enables computers to understand the semantic meaning of text. Traditional NLP uses techniques like rule-based methods and statistical models, which, although useful, often compromise flexibility and accuracy. Advancements in Large Language Models (LLMs) have enabled more natural and context-aware understanding through deep learning and transformers.

This project aims to create a simple chatbot that helps users find information about businesses using Large Language Models. It accepts questions from users, retrieves context from a dataset containing company information, and responds with answers grounded in the retrieved information. The goal is to keep the chatbot simple while demonstrating the use of frameworks such as LangChain, ChromaDB, and Retrieval-Augmented Generation (RAG).

<h2>Data Exploration</h2>

The project used the People Data Labs 2019 Global Company Dataset from Kaggle. It comprises over 7 million CSV records from companies, including domain, year founded, industry, size range, locality, country, LinkedIn URL, and current number of employees.

Sample row data:
```code
name: IBM  
domain: ibm.com  
year founded: 1911  
industry: Information Technology and Services  
size range: 10001+  
locality: New York, New York, United States  
country: United States  
linkedin url: linkedin.com/company/ibm  
current employee estimate: 274,047  
total employee estimate: 716,906
```

For a lightweight demonstration, the scope of this project retrieved only 100,000 rows of data.

<h2>Architecture and Algorithm Selection</h2>

The project used the Sentence Transformers All-MiniLM-L6-v2 embedding model and the Meta Llama 3.2 1B Instruct LLM from Hugging Face.

<h1>Development Phase</h1>

<h2>Data Pipeline Creation</h2>

<h3>Dataset Ingestion</h3>

The People Data Labs 2019 Global Company Dataset is preprocessed using a text splitter to break long text into smaller chunks. Each chunk of text is tokenized, normalized, and converted into embeddings using the Sentence Transformers All-MiniLM-L6-v2 embedding model. The embeddings are stored in ChromaDB.

<h3>Inference</h3>

The user types a question in the Gradio GUI. This input text is tokenized, normalized, and converted into embeddings using the same Sentence Transformers All-MiniLM-L6-v2 embedding model. These embeddings are compared with the stored dataset embeddings in ChromaDB using Retrieval-Augmented Generation (RAG) to retrieve the most relevant texts. The retrieved texts are passed to the Meta LLaMA 3.2 1B Instruct LLM to generate a natural language response. This response is displayed back to the user in the Gradio GUI.

Text splitting, embedding calls, and ChromaDB integration are handled using the LangChain framework.

<h2>Evaluation</h2>

<h3>Sample Prompts and Responses</h3>

<figure>
  <img src="/img/projects/business-finder-assistant/sample-prompts-and-responses-1.webp">
</figure>
<figure>
  <img src="/img/projects/business-finder-assistant/sample-prompts-and-responses-2.webp">
</figure>
<figure>
  <img src="/img/projects/business-finder-assistant/sample-prompts-and-responses-3.webp">
</figure>

<h3>Groud Truth</h3>

Tencent based in Shenzhen, Guangdong, China:
```code
name: Tencent  
domain: tencent.com  
year founded: 1998  
industry: Internet  
size range: 10001+  
locality: Shenzhen, Guangdong, China  
country: China  
linkedin url: linkedin.com/company/tencent  
current employee estimate: 37,574  
total employee estimate: 42,617
```

<br>

<h3>Guide for Local Testing</h3>

<ol>
  <li>
    Download the dataset from Kaggle
  </li>
</ol>

<a href="https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset/data" class="arrow-link">Go to dataset</a>

<ol start="2">
  <li>
    Create a folder named "data" and place the dataset in it
  </li>
</ol>

<ol start="3">
  <li>
    Clone the repository from GitHub
  </li>
</ol>

<a href="https://github.com/ivanintelligence/business-finder-assistant" class="arrow-link">Go to repository</a>

<ol start="4">
  <li>
    Create a virtual environment
  </li>
</ol>

```code
python -m venv venv
```

<ol start="5">
  <li>
    Activate the virtual environment
  </li>
</ol>

```code
source venv/bin/activate
```

<ol start="6">
  <li>
    Install dependencies
  </li>
</ol>

```code
pip install -r requirements.txt
```

<ol start="7">
  <li>
    Create a .env file and place your HuggingFace Access Token in it
  </li>
</ol>

```code
HUGGINGFACEHUB_API_TOKEN=hf...
```

Note: The project uses meta-llama/Llama-3.2-1B. Make sure to have access from repo authors.

<ol start="8">
  <li>
    Ingest the dataset
  </li>
</ol>

```code
python ingest_database.py
```

<ol start="9">
  <li>
    Run inference
  </li>
</ol>

```code
python chatbot.py
```