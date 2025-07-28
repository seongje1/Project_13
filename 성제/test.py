import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from getpass import getpass

# ✅ 1. OpenAI API 키 설정
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# ✅ 2. 경로 설정
pdf_dir = r"C:\Users\KDT10\OneDrive\바탕 화면\경북대학교"

# ✅ 3. 모든 PDF 파일 불러오기
loaders = []
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_dir, filename)
        loaders.append(PyPDFLoader(file_path))

print(f"📄 불러온 PDF 파일 수: {len(loaders)}")

# ✅ 4. 문서 로드 + 청크화
docs = []
for loader in loaders:
    docs.extend(loader.load())

print(f"📃 전체 문서 페이지 수: {len(docs)}")

# ✅ 5. 청크 나누기
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)

print(f"🧩 생성된 청크 수: {len(chunks)}")

# ✅ 청크 내용 일부 출력 (상위 3개만 예시)
for i, chunk in enumerate(chunks[:30]):
    print(f"\n🧩 청크 {i+1}:\n{chunk.page_content[:]}")  # 처음 500자까지만 출력

# ✅ 6. 벡터 저장소 생성
vectorstore = Chroma.from_documents(
    chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./knu_vectorstore"  # 로컬에 저장
)

vectorstore.persist()
print("✅ 벡터 저장소 생성 완료!")

# ✅ 7. 질의응답 체인 구성
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=retriever
)

# ✅ 8. 질문 예시
while True:
    query = input("\n💬 질문을 입력하세요 (종료하려면 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa_chain.run(query)
    print("🤖 답변:", answer)
