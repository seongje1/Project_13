import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 🔐 API 키 로드
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
print("✅ Loaded Key:", os.getenv("OPENAI_API_KEY"))

# 🌐 Streamlit 설정
st.set_page_config(page_title="📘 경북대 챗봇", layout="centered")

# 🏫 상단 로고 & 타이틀
col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/knu_logo.png", width=70)
with col2:
    st.markdown("<h2 style='margin-top:18px;'>경북대학교 AI 도우미</h2>", unsafe_allow_html=True)

# 📁 PDF 폴더 내 문서 통합 처리
PDF_DIR = "./data"
pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

if not pdf_files:
    st.error("❌ data 폴더에 PDF가 없습니다.")
    st.stop()

@st.cache_resource
def build_combined_rag(pdf_dir):
    all_docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            for p in pages:
                p.metadata["source"] = filename
            all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(all_docs)

    # ✅ OpenAI 임베딩 사용
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    system_prompt = """당신은 경북대학교 학사 관련 문서를 기반으로 정중한 한국어로 답변하는 AI 도우미입니다. \
가장 관련된 정보를 바탕으로 답변하고, 모르면 모른다고 말해주세요.\n{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    chain = (
        {"context": retriever | (lambda d: "\n\n".join(doc.page_content for doc in d)), "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ✅ 체인 초기화
chain = build_combined_rag(PDF_DIR)

# ✅ 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요! 📘 경북대 학사 관련 문서 기반으로 궁금한 걸 물어보세요 :)"}
    ]

# ✅ 대화 렌더링 (상황에 따른 마스코트 변경)
for i, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "assistant":
        if i == 0:
            mascot_img = "assets/mascot_hello.png"
        elif i == len(st.session_state["messages"]) - 1:
            mascot_img = "assets/mascot_alarm.png"
        else:
            mascot_img = "assets/mascot.png"

        col1, col2 = st.columns([1, 5])
        with col1:
            st.image(mascot_img, width=90)
        with col2:
            st.markdown(f"""
                <div style='background-color:#f0f2f6; padding:15px 20px; 
                            border-radius:15px; margin-bottom:15px;
                            border: 1px solid #ccc; box-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
                    {msg['content']}
                </div>
            """, unsafe_allow_html=True)

    elif msg["role"] == "user":
        st.markdown(f"<p style='text-align:right; color:#dcdcdc;'>🙋‍♂️ {msg['content']}</p>", unsafe_allow_html=True)

# ✅ 사용자 입력
if query := st.chat_input("질문을 입력하세요 (예: 휴학 신청은 어떻게 하나요?)"):
    st.session_state["messages"].append({"role": "user", "content": query})
    st.markdown(f"<p style='text-align:right; color:#dcdcdc;'>{query}</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("assets/mascot.png", width=90)
    with col2:
        st.markdown("""
            <div style='background-color:#fffbe6; padding:15px 20px; 
                        border-radius:15px; margin-bottom:15px;
                        border: 1px solid #f7d674; box-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
                잠시만 기다려 주세요. 관련 정보를 찾고 있어요... 🕵️‍♂️
            </div>
        """, unsafe_allow_html=True)

    with st.spinner("문서를 검색하고 있어요..."):
        try:
            answer = chain.invoke(query)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        except Exception as e:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"❌ 오류 발생: {e}\n\n🔑 OpenAI API 키가 유효한지 확인해 주세요."
            })
