import streamlit as st
import base64
import random
import os
import glob
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic

# ✅ API 키 불러오기
load_dotenv()
key = os.getenv("CLAUDE_API_KEY")
if not key:
    st.error("❌ .env 파일에 CLAUDE_API_KEY가 없습니다.")
    st.stop()
os.environ["ANTHROPIC_API_KEY"] = key

# ✅ base64 로고
def load_logo_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_logo_base64("assets/knu_logo.png")

# ✅ PDF 전부 로딩 함수 (data/*.pdf)
def load_all_pdfs_from_folder(folder_path):
    pages = []
    for path in glob.glob(os.path.join(folder_path, "*.pdf")):
        loader = PyPDFLoader(path)
        pages.extend(loader.load_and_split())
    return pages

# ✅ RAG 체인 생성
@st.cache_resource
def create_rag_chain(uploaded_file=None, use_only_uploaded=False):
    pages = []

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages.extend(loader.load_and_split())

    if not use_only_uploaded:
        pages.extend(load_all_pdfs_from_folder("data"))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 경북대학교에 관한 정보를 제공하는 AI 도우미입니다. "
                   "아래 문서 내용을 참고하여 정확하고 공손하게 한국어로 답변해 주세요. 이모지도 함께 사용하세요.\n\n{context}"),
        ("human", "{input}")
    ])

    return (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "input": RunnablePassthrough()}
        | prompt
        | ChatAnthropic(model="claude-3-haiku-20240307")
        | StrOutputParser()
    )

# ✅ 페이지 설정
st.set_page_config(page_title="📘 경북대 챗봇", layout="centered")

# ✅ 사이드바
with st.sidebar:
    st.image("assets/knu_logo2.png", width=200)
    st.markdown("### 학사일정")
    st.markdown("""
    - 🗓️ 개강: **2025.09.01**
    - 📦 수강꾸러미 신청: **07.22 ~ 07.24**
    - 🖋️ 수강신청: **08.12 ~ 08.14**
    - 📝 중간고사: **10.22 ~ 10.28**
    - 💳 등록금 납부: **08.25 ~ 08.28**
    """)

    st.markdown("### 📤 PDF 업로드")
    uploaded_file = st.file_uploader("문서 업로드 (선택)", type=["pdf"])
    mode = st.radio("문서 사용 방식", ["기본 문서 + 업로드 문서", "업로드 문서만 사용"])

    st.markdown("### 📄 기본 문서 다운로드")
    for path in glob.glob("data/*.pdf"):
        with open(path, "rb") as f:
            st.download_button(f"📄 {os.path.basename(path)}", f.read(), file_name=os.path.basename(path), mime="application/pdf")

# ✅ 세션 상태 초기화 및 체인 구성
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요! 📘 경북대 학사 도우미입니다. 무엇이든 물어보세요!"}]

if (
    "rag_chain" not in st.session_state or
    "last_uploaded_name" not in st.session_state or
    "last_mode" not in st.session_state or
    st.session_state["last_uploaded_name"] != (uploaded_file.name if uploaded_file else None) or
    st.session_state["last_mode"] != mode
):
    st.session_state["rag_chain"] = create_rag_chain(
        uploaded_file=uploaded_file,
        use_only_uploaded=(mode == "업로드 문서만 사용")
    )
    st.session_state["last_uploaded_name"] = uploaded_file.name if uploaded_file else None
    st.session_state["last_mode"] = mode

# ✅ 상단 로고 및 타이틀
st.markdown(f"""
    <div style='display:flex; align-items:center; justify-content:center; gap:20px; margin-bottom:30px;'>
        <img src="data:image/png;base64,{logo_base64}" style="height:80px;">
        <h2 style='font-size: 45px; font-weight: 700;'>경북대학교 AI 도우미</h2>
    </div>
""", unsafe_allow_html=True)

# ✅ 이전 메시지 출력
for i, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "assistant":
        mascot = "assets/mascot_hello.png" if i == 0 else (
            "assets/mascot_graduate.png" if any(k in st.session_state["messages"][i-1]["content"] for k in ["졸업", "학위"]) else random.choice([
                "assets/mascot.png", "assets/mascot_love.png", "assets/mascot_alarm.png"])
        )
        col1, col2 = st.columns([1, 8])
        with col1: st.image(mascot, width=120)
        with col2:
            st.markdown(f"<div style='background:#fff;padding:15px;border-radius:20px;border:1px solid #ddd;'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='text-align:right;margin-bottom:15px;'>
                <div style='display:inline-block;background:#b71c1c;color:white;padding:15px 20px;border-radius:20px;'>
                    {msg['content']}
                </div>
            </div>
        """, unsafe_allow_html=True)

# ✅ 자주 묻는 질문 버튼
faq = ["휴학은 어떻게 하나요?", "복학 신청은 어디서 하나요?", "수강신청 일정은 언제인가요?", "성적 열람은 어디서 하나요?", "학생증 발급은 어떻게 하나요?"]
cols = st.columns(len(faq))
for i, q in enumerate(faq):
    if cols[i].button(q):
        st.session_state["messages"].append({"role": "user", "content": q})
        with st.spinner("답변 생성 중..."):
            res = st.session_state["rag_chain"].invoke(q)
            st.session_state["messages"].append({"role": "assistant", "content": res})
            st.rerun()

# ✅ 사용자 입력
if user_input := st.chat_input("질문을 입력하세요 (예: 수강신청 일정은?)"):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.spinner("답변 생성 중..."):
        res = st.session_state["rag_chain"].invoke(user_input)
        st.session_state["messages"].append({"role": "assistant", "content": res})
        st.rerun()
