import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# ✅ .env에서 OPENAI_API_KEY 불러오기
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ✅ Streamlit 초기 설정
st.set_page_config(page_title="📘 GPT-4 vs RAG 챗봇", layout="wide")
st.title("🤖 GPT-4 vs 📄 RAG 챗봇 비교")
st.write("✅ 앱이 정상적으로 실행되었습니다.")

# 📤 PDF 업로드
uploaded_file = st.file_uploader("📄 문서를 업로드하세요 (PDF)", type=["pdf"])

# 🔧 함수 정의
def load_pdf(_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    return loader.load_and_split()

def create_vectorstore(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)
    return Chroma.from_documents(docs, OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key))

def build_rag_chain(_vectorstore):
    retriever = _vectorstore.as_retriever()
    system_prompt = """당신은 문서를 기반으로 질문에 대답하는 친절한 챗봇입니다. \
다음 context를 참고해서 질문에 정중하고 정확하게 답해주세요. \
모르면 모른다고 답하세요. 한국어로 답하세요. \
{context}"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
    return (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def gpt4_response(query):
    model = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
    return model.predict(query)

# 💬 질문 입력
query = st.text_input("❓ 경북대학교 관련 질문을 입력하세요")

# ✅ 응답 비교 출력
if uploaded_file and query:
    with st.spinner("PDF 처리 중..."):
        pages = load_pdf(uploaded_file)
        vectorstore = create_vectorstore(pages)
        rag_chain = build_rag_chain(vectorstore)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌐 GPT‑4 기본 응답")
        gpt_answer = gpt4_response(query)
        st.write(gpt_answer)

    with col2:
        st.subheader("📄 PDF 기반 RAG 응답")
        rag_answer = rag_chain.invoke(query)
        st.write(rag_answer)
