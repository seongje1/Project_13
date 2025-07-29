from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from bert_score import score as bert_score
import os
import getpass

# 🔐 Claude API Key 입력 받기
claude_api_key = getpass.getpass("Claude API Key를 입력하세요: ")
os.environ["ANTHROPIC_API_KEY"] = claude_api_key

# 📁 PDF 로딩
folder_path = r"C:/_vscode/Project_13/성제/경북대학교"
loader = PyPDFDirectoryLoader(folder_path)
documents = loader.load()
print(f"📄 불러온 PDF 문서 수: {len(documents)}개")

# 📚 청크 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"🧩 생성된 청크 수: {len(chunks)}개")

# 🤖 임베딩 및 벡터저장소 생성
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# 🧠 Claude 모델
llm = ChatAnthropic(model="claude-3-haiku-20240307")

# 🔄 RAG 체인
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(lambda x: f"다음 문서를 참고해서 질문에 답하세요:\n\n{x['context']}\n\n질문: {x['question']}")
    | llm
)

# 🔁 반복 루프 시작
while True:
    query = input("\n💬 평가할 질문을 입력하세요 (종료하려면 'exit'): ")
    if query.strip().lower() == "exit":
        print("👋 프로그램을 종료합니다.")
        break

    reference_answer = input("📘 해당 질문의 모범 답변(reference)을 입력하세요:\n")
    if reference_answer.strip().lower() == "exit":
        print("👋 프로그램을 종료합니다.")
        break

    # (1) RAG 응답
    print("\n🔍 RAG 기반 Claude 응답 생성 중...")
    rag_response = rag_chain.invoke(query).content
    print(f"\n📢 [RAG 응답]:\n{rag_response}")

    # (2) BERT-Score 계산
    print("\n📊 BERT-Score 계산 중...")

    P_rag, R_rag, F1_rag = bert_score(
        [rag_response], [reference_answer],
        lang="ko", model_type="xlm-roberta-large"
    )

    # (3) 결과 출력
    print("\n✅ BERT-Score 결과 (기준: 모범 답변)")
    print("🧠 RAG 기반 Claude 응답:")
    print(f"  - Precision: {P_rag.mean():.4f}")
    print(f"  - Recall   : {R_rag.mean():.4f}")
    print(f"  - F1 Score : {F1_rag.mean():.4f}")
