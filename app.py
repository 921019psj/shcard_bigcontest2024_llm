import os
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import streamlit as st
import google.generativeai as genai

# Security: API key management
try:
    genai.configure(api_key="YOUR_ACTUAL_API_KEY")
    model = genai.GenerativeModel("gemini-1.5-flash")
except KeyError:
    st.error("API key is not set in the secrets.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing the model: {e}")
    st.stop()

# Streamlit page setup
st.set_page_config(page_title="🍊참신한 제주 레스토랑!", layout="wide")
st.title("혼저 옵서예!👋")
st.subheader("군맛난 제주 밥집🧑‍🍳 추천해드릴게예")

# Image display
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRTHBMuNn2EZw3PzOHnLjDg_psyp-egZXcclWbiASta57PBiKwzpW5itBNms9VFU8UwEMQ&usqp=CAU"
st.image(image_url, use_column_width=True)

# Load and preprocess data
data_path = './data'
csv_file_path = "JEJU_DATA.csv"

def load_csv(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='cp949')
        df = df[df['기준연월'] == df['기준연월'].max()].reset_index(drop=True)
        return df
    else:
        st.error(f"{file_path} does not exist.")
        return pd.DataFrame()

df = load_csv(os.path.join(data_path, csv_file_path))

# FAISS and embedding setup
module_path = './modules'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sroberta-multitask")
embedding_model = AutoModel.from_pretrained("jhgan/ko-sroberta-multitask").to(device)

embeddings = np.load(os.path.join(module_path, 'embeddings_array_file.npy'))
faiss_index = faiss.read_index(os.path.join(module_path, 'faiss_index.index'))

# Embed text function
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# Response generation function
def generate_response_with_faiss(question, df, k=3):
    query_embedding = embed_text(question).reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, k * 3)
    filtered_df = df.iloc[indices[0, :]].copy().reset_index(drop=True).head(k)

    if filtered_df.empty:
        return "질문과 일치하는 가게가 없습니다."

    reference_info = "\n".join(filtered_df['text'])
    prompt = (
        f"질문: {question}\n"
        f"대답해줄 때 업종별로 가능하면 하나씩 추천해줘. "
        f"그리고 추가적으로 오래된 맛집과 새로운 맛집을 각각 추천해줘.\n"
        f"참고할 정보: {reference_info}\n응답:"
    )
    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text

# Conversation history management
history_path = os.path.join(module_path, 'conversation_history.json')

def save_conversation_history(conversations):
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)
    st.success("Conversation history saved.")

def load_conversation_history():
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversation_history()
    if not st.session_state.conversations:
        st.session_state.conversations.append({"id": str(uuid.uuid4()), "title": "", "messages": []})
    st.session_state.current_conversation = st.session_state.conversations[0]

# Sidebar setup
with st.sidebar:
    st.header("💾 대화 저장")
    if st.button("대화 저장"):
        save_conversation_history(st.session_state.conversations)

# Chat input and response handling
chat_input = st.chat_input("질문을 입력하세요:")
if chat_input:
    current_conv = st.session_state.current_conversation

    # Automatically set the title for the conversation based on the first question
    if not current_conv["title"]:
        current_conv["title"] = (chat_input[:15] + "...") if len(chat_input) > 15 else chat_input

    # Append user message
    user_message = {"role": "user", "content": chat_input, "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    current_conv["messages"].append(user_message)
    st.markdown(f"**사용자:** {chat_input}")

    # Generate assistant response
    response = generate_response_with_faiss(chat_input, df)
    assistant_message = {"role": "assistant", "content": response, "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    current_conv["messages"].append(assistant_message)
    st.markdown(f"**어시스턴트:** {response}")

    # Save the conversation history
    save_conversation_history(st.session_state.conversations)
