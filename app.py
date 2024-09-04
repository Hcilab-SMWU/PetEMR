import streamlit as st
from streamlit_option_menu import option_menu
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from pymilvus import connections, Collection #, FieldSchema,  DataType, Collection, utility

# Connect to milvus
HOST = 'localhost'
PORT = '19530'
COLLECTION_NAME = 'PetEMR_db'
INDEX_TYPE = 'IVF_FLAT'
EMBEDDING_FIELD_NAME = 'soap_embedding'

connections.connect(host=HOST, port=PORT)

collection = Collection(name=COLLECTION_NAME)
# st.write(collection)

# Use embedding model
tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sroberta-multitask')
model = AutoModel.from_pretrained('jhgan/ko-sroberta-multitask')

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embedder(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embedding

def embedding(text):
    text_embedding = embedder(text)
    text_embedding = text_embedding.squeeze(0)
    text_embedding = text_embedding.tolist()

    return text_embedding

def searching(query_embedding):
    index_params = {'metric_type': 'IP', 'index_type': INDEX_TYPE, 'params': {'nlist': 1}}
    result = collection.search(
            data = [query_embedding],
            anns_field = EMBEDDING_FIELD_NAME,
            param = index_params,
            limit = 100,
            output_fields = ['complaint', 'soap', 'label']
        )
    
    result_list = []
    for hits in result:
        for hit in hits:
            result_list.append(hit.to_dict())

    results = []
    for i in range(len(result_list)):
        if result_list[i]['distance'] > 80:
            results.append(result_list[i])
            
    return results

# Page layout
# 1. Sidebar
with st.sidebar:
    menu = option_menu("Menu", ["Main", "Search"])

# 2. Main page
def main_page():
    st.title("pet EMR")

# 3. Search page
def search_page():
    st.title("Search")

    breed = st.selectbox('반려동물 종을 선택하세요.', ('선택', '강아지', '고양이', '기타'))
    search_button = False

    if breed == '강아지':
        with st.form("search_form"):
            symptoms = st.text_input("반려동물의 증상을 입력하세요.")
            search_button = st.form_submit_button("Search")  
    elif breed == '선택':
        st.write("")
    else:
        st.write(f"{breed}는 아직 지원하지 않습니다😿😿")

    if search_button:
        text_embedding = embedding(symptoms)
        results = searching(text_embedding)

        for i in range(len(results)):
            st.write(f"{i}: {results[i]}")
            st.write("")

page_names = {'Main': main_page, 'Search': search_page}
page_names[menu]()