# NLP-based Similar Symptom Search for Pets Using Electronic Medical Record (EMR) Data

We developed an NLP-based similar symptom search service using pet Electronic Medical Record (EMR) data from animal hospitals, which also estimates the probability of heart disease based on the identified symptoms. Additionally, we built a heart disease prediction service that uses EMR data to provide probability estimates for heart disease based on similar cases and recommends relevant tests. For efficient search functionality, we designed and implemented a vector database using an SBERT-based text embedding model and Milvus.

## Prototype

We implemented a web prototype using Streamlit. You can test it at <a href="https://eb54-203-252-192-163.ngrok-free.app">this link</a>.
<br/>
<!--<div align=center><img src="https://github.com/user-attachments/assets/d24dd4dd-2891-4921-bc91-4f559a03072b" width=80% heigth=80%></div>-->
<br/>
<div align=center><img src="https://github.com/user-attachments/assets/ae1cd34b-cde4-4ca6-b850-cea1eba59400" width=85% heigth=85%></div>

## Usage

1. Set up for using <a herf="https://milvus.io/docs/install_standalone-docker.md">milvus</a>
2. Install <a herf="https://milvus.io/docs/install-pymilvus.md">pymilvus</a>
3. Create embeddings for data insert and query (refer to emr_embedding.ipynb)
