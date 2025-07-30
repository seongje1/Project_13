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

# ✅ API 키 로드
load_dotenv()
key = os.getenv("CLAUDE_API_KEY")
if not key:
    st.error("❌ .env에 CLAUDE_API_KEY가 없습니다.")
    st.stop()
os.environ["ANTHROPIC_API_KEY"] = key

# ✅ 로고 base64
def load_logo_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
logo_base64 = load_logo_base64("assets/knu_logo.png")

# ✅ PDF 로딩
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
        ("system", 
         "당신은 경북대학교에 관한 정보를 제공하는 AI 도우미입니다. "
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

# ✅ 스타일
st.markdown("""
    <style>
    body {
        font-family: 'Noto Sans KR', sans-serif;
    }
    .block-container {
        padding-left: 5rem;
        padding-right: 5rem;
        padding-top: 2rem;
    }

    /* 사이드바 기본 배경 및 텍스트 */
    section[data-testid="stSidebar"] {
        background-color: #b71c1c;
        color: white;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* 흰 배경 요소 내부의 텍스트는 검정색으로 덮어쓰기 */
    section[data-testid="stSidebar"] .stDownloadButton button,
    section[data-testid="stSidebar"] .stDownloadButton button *,
    section[data-testid="stSidebar"] .stFileUploader,
    section[data-testid="stSidebar"] .stFileUploader *,
    section[data-testid="stSidebar"] input {
        color: black !important;
    }

    /* ➕ 문서 업로드(선택), 문서 사용 방식 라벨만 흰색 유지 */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio > div > label {
        color: white !important;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ 사이드바
with st.sidebar:
    st.image("assets/knu_logo2.png", width=200)
    st.markdown("###  학사일정")
    st.markdown("""
    - 🗓️ 개강: **2025.09.01**
    - 📦 수강꾸러미 신청: **07.22 ~ 07.24**  
    - 🖋️ 수강신청: **08.12 ~ 08.14**  
    - 📝 중간고사: **10.22 ~ 10.28**  
    - 💳 등록금 납부: **08.25 ~ 08.28**
    """)

    with st.expander("🔗 바로가기 링크"):
        st.markdown("- [경북대학교 홈페이지](https://www.knu.ac.kr)")
        st.markdown("- [종합정보시스템](https://appfn.knu.ac.kr/login.knu?agentId=4)")
        st.markdown("- [수강신청 페이지](https://sugang.knu.ac.kr)")
        st.markdown("- [시간표 조회 시스템](https://knuin.knu.ac.kr/public/stddm/lectPlnInqr.knu)")

    with st.expander("📄 문서 다운로드"):
        for pdf_path in glob.glob("data/*.pdf"):
            with open(pdf_path, "rb") as f:
                filename = os.path.basename(pdf_path)
                st.download_button(f"📄 {filename}", f.read(), file_name=filename, mime="application/pdf")

    with st.expander("📤 문서 추가 업로드"):
        uploaded_file = st.file_uploader("문서 업로드 (선택)", type=["pdf"])
        mode = st.radio("문서 사용 방식", ["기본 문서 + 업로드 문서", "업로드 문서만 사용"])

# ✅ 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요! 📘 경북대 학사 도우미입니다.", "mascot": "assets/mascot_hello.png"}]

if (
    "rag_chain" not in st.session_state or
    st.session_state.get("last_uploaded_name") != (uploaded_file.name if uploaded_file else None) or
    st.session_state.get("last_mode") != mode
):
    st.session_state["rag_chain"] = create_rag_chain(
        uploaded_file=uploaded_file,
        use_only_uploaded=(mode == "업로드 문서만 사용")
    )
    st.session_state["last_uploaded_name"] = uploaded_file.name if uploaded_file else None
    st.session_state["last_mode"] = mode

# ✅ 타이틀
st.markdown(f"""
    <div style='display:flex; align-items:center; justify-content:center; gap: 15px; margin-bottom: 32px;'>
        <img src="data:image/png;base64,{logo_base64}" style="height:80px;">
        <h2 style='font-size: 45px; font-weight: 700; color:#212121;'>경북대학교 AI 도우미</h2>
    </div>
""", unsafe_allow_html=True)

# ✅ 메시지 출력
for msg in st.session_state["messages"]:
    if msg["role"] == "assistant":
        mascot_img = msg.get("mascot", "assets/mascot.png")
        col1, col2 = st.columns([1, 8])
        with col1:
            st.image(mascot_img, width=130)
        with col2:
            st.markdown(f"""
                <div style='position:relative; background-color:#ffffff;
                            padding:15px 20px; border-radius:20px;
                            border: 1px solid #e0e0e0;
                            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
                            max-width: 90%; margin-bottom:15px;'>
                  {msg['content']}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='text-align:right; margin-bottom:15px;'>
                <div style='background:#b71c1c; color:white;
                            padding:15px 20px; border-radius:20px;
                            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
                            display:inline-block; max-width:85%;'>
                    {msg['content']}
                </div>
            </div>
        """, unsafe_allow_html=True)

# ✅ 자주 묻는 질문 버튼
frequent_questions = [
    "휴학은 어떻게 하나요?",
    "복학 신청은 어디서 하나요?",
    "수강신청 일정은 언제인가요?",
    "성적 열람은 어디서 하나요?",
    "학생증 발급은 어떻게 하나요?"
]
cols = st.columns(len(frequent_questions))
for idx, q in enumerate(frequent_questions):
    if cols[idx].button(q):
        st.session_state["messages"].append({"role": "user", "content": q})
        with st.spinner("답변 생성 중..."):
            response = st.session_state["rag_chain"].invoke(q)
            mascot_img = (
                "assets/mascot_graduate.png" if any(k in q for k in ["졸업", "졸업요건", "졸업논문", "졸업학점", "학위"]) else
                random.choice(["assets/mascot.png", "assets/mascot_love.png", "assets/mascot_alarm.png"])
            )
            st.session_state["messages"].append({"role": "assistant", "content": response, "mascot": mascot_img})
            st.rerun()

# ✅ 사용자 입력
if user_input := st.chat_input("질문을 입력하세요 (예: 수강신청 일정은 언제인가요?)"):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.spinner("답변 생성 중..."):
        response = st.session_state["rag_chain"].invoke(user_input)
        mascot_img = (
            "assets/mascot_graduate.png" if any(k in user_input for k in ["졸업", "졸업요건", "졸업논문", "졸업학점", "학위"]) else
            random.choice(["assets/mascot.png", "assets/mascot_love.png", "assets/mascot_alarm.png"])
        )
        st.session_state["messages"].append({"role": "assistant", "content": response, "mascot": mascot_img})
        st.rerun()
