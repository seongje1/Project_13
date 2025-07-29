import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from bert_score import score as bert_score
from getpass import getpass

# ✅ 1. OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = getpass("🔐 OpenAI API 키를 입력하세요: ")

# ✅ 2. 경로 설정
pdf_dir = r"C:\_vscode\Project_13\성제\경북대학교"

# ✅ 3. 모든 PDF 파일 불러오기
loaders = [PyPDFLoader(os.path.join(pdf_dir, f))
           for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
print(f"📄 불러온 PDF 파일 수: {len(loaders)}")

# ✅ 4. 문서 로드 + 청크화
docs = []
for loader in loaders:
    docs.extend(loader.load())
print(f"📃 전체 문서 페이지 수: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"🧩 생성된 청크 수: {len(chunks)}")

# ✅ 5. 벡터 저장소 생성
vectorstore = Chroma.from_documents(
    chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./knu_vectorstore"
)
vectorstore.persist()
print("✅ 벡터 저장소 생성 완료!")

# ✅ 6. 질의응답 체인 구성
retriever = vectorstore.as_retriever()
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=retriever
)
gpt_direct = ChatOpenAI(model="gpt-3.5-turbo")

# ✅ 7. 비교 루프 시작
while True:
    query = input("\n💬 질문을 입력하세요 (종료하려면 'exit'): ")
    if query.lower() == "exit":
        break

    print("🤖 RAG 기반 응답 생성 중...")
    rag_answer = rag_chain.run(query)

    print("🤖 일반 GPT 응답 생성 중...")
    gpt_answer = gpt_direct.predict(query)

    print("\n✅ [RAG 응답]:\n", rag_answer)
    print("\n✅ [GPT 단독 응답]:\n", gpt_answer)

    # ✅ BERT-Score 계산
    P, R, F1 = bert_score(
        [rag_answer],  # candidate
        [gpt_answer],  # reference
        lang="ko",     # 한국어일 경우
        model_type="xlm-roberta-large"  # 한국어 지원 모델
    )

    print(f"\n📊 BERT-Score 유사도")
    print(f"Precision: {P.mean():.4f}")
    print(f"Recall   : {R.mean():.4f}")
    print(f"F1       : {F1.mean():.4f}")
