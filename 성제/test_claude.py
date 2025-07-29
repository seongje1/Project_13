# 🧠 1. 기본 모듈 및 API 키 설정
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

# 📁 2. PDF 로딩
folder_path = r"C:/_vscode/Project_13/성제/경북대학교"
loader = PyPDFDirectoryLoader(folder_path)
documents = loader.load()
print(f"📄 불러온 PDF 문서 수: {len(documents)}개")

# 📚 3. 청크 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"🧩 생성된 청크 수: {len(chunks)}개")

# 📌 4. HuggingFace 임베딩 모델
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 🧠 5. 벡터 저장소 생성
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# 🤖 6. Claude 모델 설정
llm = ChatAnthropic(model="claude-3-haiku-20240307")

# 🔄 7. RAG 체인 구성
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(lambda x: f"다음 문서를 참고해서 질문에 답하세요:\n\n{x['context']}\n\n질문: {x['question']}")
    | llm
)

# 🧪 8. 비교 질문 수행
query = input("\n💬 비교할 질문을 입력하세요: ")

# (1) RAG 응답
print("\n🔍 RAG 기반 응답 생성 중...")
rag_response = rag_chain.invoke(query).content
print(f"\n📢 [RAG 응답]:\n{rag_response}")

# (2) 일반 Claude 직접 질의 응답
print("\n🤖 일반 Claude 응답 생성 중...")
gpt_response = llm.invoke(query).content
print(f"\n📢 [일반 GPT 응답]:\n{gpt_response}")

# (3) BERT-Score 계산
print("\n📊 BERT-Score 계산 중...")
P, R, F1 = bert_score(
    [rag_response],
    [gpt_response],
    lang="ko",  # 한국어
    model_type="xlm-roberta-large"  # or jhgan/ko-sbert-nli
)

"""
1. 두 문장을 각각 BERT 모델을 통해 토큰 임베딩으로 변환
2. 두 문장의 토큰쌍 간 cosine similarity 계산 (모든 토큰 조합)
3. Precision, Recall, F1 계산
"""

print("\n✅ BERT-Score 결과:")
print(f"  - Precision: {P.mean():.4f}")
print(f"  - Recall   : {R.mean():.4f}")
print(f"  - F1 Score : {F1.mean():.4f}")
