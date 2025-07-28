# app.py

import streamlit as st
import os
import getpass
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

# 🔐 OpenAI API Key 입력 받기
openai_api_key = getpass.getpass("🔑 OpenAI API Key를 입력하세요: ")
os.environ["OPENAI_API_KEY"] = openai_api_key

# 벡터 DB 불러오기
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectordb = Chroma(persist_directory="./db", embedding_function=embedding)

# LLM 준비
llm = OpenAI(openai_api_key=openai_api_key, temperature=0)

# RAG QA 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff"
)


# Streamlit UI 구성
st.set_page_config(page_title="전자공학과 AI 비서", page_icon="🤖")
st.title("📘 전자공학과 학사 정보 챗봇")

question = st.text_input("무엇이 궁금한가요?", placeholder="예: 졸업 학점은 몇 점인가요?")

if question:
    with st.spinner("검색 중..."):
        answer = qa_chain.run(question)
    st.success("💬 답변:")
    st.write(answer)
