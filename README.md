# NLP-based Similar Symptom Search for Pets Using Electronic Medical Record (EMR) Data

We developed an NLP-based similar symptom search service using pet Electronic Medical Record (EMR) data from animal hospitals, which also estimates the probability of heart disease based on the identified symptoms. Additionally, we built a heart disease prediction service that uses EMR data to provide probability estimates for heart disease based on similar cases and recommends relevant tests. For efficient search functionality, we designed and implemented a vector database using an SBERT-based text embedding model and Milvus.

## Prototype

We implemented a web prototype using Streamlit. You can test it at <a href=" https://82ff-203-252-192-163.ngrok-free.app">this link</a>.
<br/>
<div align=center><img src="https://github.com/user-attachments/assets/206bf1c9-44a8-4a0a-96b5-bbd8fe4fa344"></div>

## Usage

1. Set up for using <a href="https://milvus.io/docs/install_standalone-docker.md">milvus</a>
2. Install <a href="https://milvus.io/docs/install-pymilvus.md">pymilvus</a>
3. Create embeddings for data insert and query (refer to emr_embedding.ipynb)
