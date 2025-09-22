---
layout: subpage
hero: /img/projects/business-finder-assistant.webp
---

<title>Large Language Model (LLM)–Powered Business Finder Assistant Using LangChain, ChromaDB, and Retrieval-Augmented Generation (RAG)</title>

By John Ivan Diaz

A simple Large Language Model-powered chatbot that helps users find information about a business. It uses People Data Labs 2019 Global Company Dataset from Kaggle, the Sentence Transformers All-MiniLM-L6-v2 embedding model, and the Meta Llama 3.2 1B Instruct LLM from HuggingFace. While the chatbot is simple and retrieves only a small fraction of the dataset, its goal is to demonstrate LangChain, ChromaDB, and Retrieval-Augmented Generation (RAG) for LLM orchestration, vector storage, and retrieval.

<tag>Large Language Models (LLMs)</tag>
<tag>LangChain</tag>
<tag>ChromaDB</tag>
<tag>Retrieval Augmented Generation (RAG)</tag>
<tag>Hugging Face</tag>

<a href="https://github.com/ivanintelligence/business-finder-assistant" class="arrow-link">See source code</a>

<hr class="hr-custom">
<br>

<h3>Data Exploration</h3>

The project uses the People Data Labs 2019 Global Company Dataset from Kaggle. It comprises over 7 million CSV records from companies, including domain, year founded, industry, size range, locality, country, LinkedIn URL, and current number of employees.

Sample row data:
```code
name: IBM
industry: Information Technology and Services
locality: New York, New York, United States
country: United States
year founded: 1911
size range: 10000+
current employee size: 274047
domain: ibm.com
linkedin url: linkedin.com/company/ibm
```

For a lightweight demonstration, the scope of this project retrieves only 100,000 rows of data out of more than 7 million.

<h3>Architecture and Algorithm Selection</h3>

The project uses the Sentence Transformers All-MiniLM-L6-v2 embedding model and the Meta Llama 3.2 1B Instruct LLM from Hugging Face.

<h3>Instructions to test code locally</h3>

<ol>
  <li>
    Download the dataset from Kaggle
  </li>
</ol>

<u>https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset/data</u>

<ol start="2">
  <li>
    Create a folder named "data" and place the dataset in it
  </li>
</ol>

<ol start="3">
  <li>
    Clone the repository from my GitHub
  </li>
</ol>

<u>https://github.com/ivanintelligence/business-finder-assistant</u>

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