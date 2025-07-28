import os
import getpass
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 🔐 OpenAI API Key 입력 받기
openai_api_key = getpass.getpass("🔑 OpenAI API Key를 입력하세요: ")
os.environ["OPENAI_API_KEY"] = openai_api_key

# 1. PDF 파일 경로
pdf_path = "C:/Users/KDT21/lsm/Project_13/수민/학과별 생활 가이드북(Blue Book) 학사 자료.pdf"

# 2. PDF 불러오기
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# 3. 텍스트 쪼개기
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
documents = text_splitter.split_documents(pages)

# 4. OpenAI 임베딩 생성기
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# 5. Chroma DB로 저장
vectordb = Chroma.from_documents(documents, embedding, persist_directory="./db")
vectordb.persist()

print("✅ 벡터 저장 완료! 이제 질문할 수 있는 준비 끝!")
