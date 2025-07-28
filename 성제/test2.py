# 🧠 1. 기본 모듈 및 API 키 설정
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import os
import getpass


# 🔐 Claude API Key 입력 받기
claude_api_key = getpass.getpass("Claude API Key를 입력하세요: ")
os.environ["ANTHROPIC_API_KEY"] = claude_api_key

# 📁 2. PDF 로딩 (경북대 PDF 폴더 경로)
folder_path = "C:/Users/KDT10/OneDrive/바탕 화면/경북대학교"
loader = PyPDFDirectoryLoader(folder_path)
documents = loader.load()

# ✅ 불러온 PDF 수 확인
print(f"📄 불러온 PDF 문서 수: {len(documents)}개")

# 📚 3. 청크 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"🧩 생성된 청크 수: {len(chunks)}개")

# 📌 4. 임베딩 모델 (한국어 HuggingFace 모델 사용)
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 🧠 5. 벡터 저장소 생성 (Chroma 사용)
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# 🤖 6. Claude 기반 LLM 설정
llm = ChatAnthropic(model="claude-3-haiku-20240307")

# 🔄 7. RAG 체인 구성
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(lambda x: f"다음 문서를 참고해서 질문에 답하세요:\n\n{x['context']}\n\n질문: {x['question']}")
    | llm
)

# 🧪 8. 테스트 질문
query = "배준현 교수님 이메일에 대해 알려줘"
response = rag_chain.invoke(query)
print(f"\n📢 답변:\n{response.content}")
