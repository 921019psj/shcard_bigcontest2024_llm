import os
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import streamlit as st
import google.generativeai as genai

# 경로 설정
data_path = './data'
module_path = './modules'

# Google Gemini API 설정
genai.configure(api_key="AIzaSyD1eKM8Wo6kW4p1UnflQKUzl8Oi-85p7v8")
model = genai.GenerativeModel("gemini-1.5-flash")

# CSV 파일 로드
csv_file_path = "JEJU_DATA.csv"
df = pd.read_csv(os.path.join(data_path, csv_file_path), encoding='cp949')
df = df[df['기준연월'] == df['기준연월'].max()].reset_index(drop=True)

# Streamlit App UI 설정
st.set_page_config(page_title="🍊참신한 제주 맛집!")
st.title("혼저 옵서예!👋")
st.subheader("군맛난 제주 밥집🧑‍🍳 추천해드릴게예")
image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRTHBMuNn2EZw3PzOHnLjDg_psyp-egZXcclWbiASta57PBiKwzpW5itBNms9VFU8UwEMQ&usqp=CAU"
image_html = f"""<div style="display: flex; justify-content: center;">
    <img src="{image_path}" alt="centered image" width="50%">
</div>"""
st.markdown(image_html, unsafe_allow_html=True)

# 대화 세션 저장 및 불러오기
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = {"id": None, "messages": [], "title": ""}

# 사이드바에서 대화 기록 관리
st.sidebar.header("💬 대화 기록 관리")

# 새로운 대화 시작 버튼
if st.sidebar.button("새로운 대화 시작"):
    new_conversation = {"id": len(st.session_state.conversations) + 1, "messages": [], "title": ""}
    st.session_state.conversations.append(new_conversation)
    st.session_state.current_conversation = new_conversation

# 대화 세션 선택
if st.session_state.conversations:
    conversation_titles = [f"{conv['title'] or '대화 세션 ' + str(conv['id'])}" for conv in st.session_state.conversations]
    selected_conversation_title = st.sidebar.selectbox("대화 세션 선택", conversation_titles)

    # 선택된 대화 세션 로드
    selected_conversation = next(
        (conv for conv in st.session_state.conversations if (conv['title'] or f"대화 세션 {conv['id']}") == selected_conversation_title), None)
    
    if selected_conversation:
        st.session_state.current_conversation = selected_conversation
        st.session_state.messages = selected_conversation["messages"]
    else:
        st.session_state.messages = []
else:
    st.session_state.messages = []

# 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 채팅 내역 초기화 버튼
def clear_chat_history():
    if st.session_state.current_conversation:
        st.session_state.current_conversation["messages"] = []
    st.session_state.messages = []
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face 사전 학습된 모델 로드
model_name = "jhgan/ko-sroberta-multitask"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = AutoModel.from_pretrained(model_name).to(device)
except Exception as e:
    st.error(f"Error loading tokenizer or model: {e}")

# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index.index')):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return index
    else:
        raise FileNotFoundError(f"{index_path} 파일이 존재하지 않습니다.")

# 텍스트 임베딩 함수
def embed_text(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.squeeze().cpu().numpy()
    except Exception as e:
        st.error(f"Error in embedding text: {e}")
        return None

# FAISS 기반 응답 생성 함수
def generate_response_with_faiss(question, df, embeddings, model, embed_text, index_path=os.path.join(module_path, 'faiss_index.index'), k=3):
    try:
        index = load_faiss_index(index_path)
        query_embedding = embed_text(question).reshape(1, -1)
        distances, indices = index.search(query_embedding, k*3)
        filtered_df = df.iloc[indices[0, :]].copy().reset_index(drop=True)

        if filtered_df.empty:
            return "질문과 일치하는 가게가 없습니다."

        reference_info = "\n".join(filtered_df['text'])
        prompt = f"질문: {question} \n대답해줄때 업종별로 가능하면 하나씩 추천해줘. 그리고 추가적으로 오래된 맛집과 새로운 맛집을 각각 추천해줘.\n참고할 정보: {reference_info}\n응답:"
        response = model.generate_content(prompt)
        return response
    except Exception as e:
        return f"응답 생성 중 오류가 발생했습니다: {e}"

# 사용자 입력 처리 및 응답 생성
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.current_conversation["messages"].append({"role": "user", "content": prompt})
    
    # 제목 자동 설정 (질문 요약)
    st.session_state.current_conversation["title"] = prompt[:15] + "..."  # 질문을 요약하여 제목 설정
    
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response_with_faiss(prompt, df, None, model, embed_text)
                full_response = response if isinstance(response, str) else response.text
                st.write(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.current_conversation["messages"].append({"role": "assistant", "content": full_response})  
