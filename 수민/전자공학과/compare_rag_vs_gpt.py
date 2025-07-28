import os
import getpass
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

# 🔐 OpenAI API Key 입력 받기
api_key = getpass.getpass("🔑 OpenAI API Key를 입력하세요: ")
os.environ["OPENAI_API_KEY"] = api_key

# 1. RAG 체인 (문서 기반 GPT)
embedding = OpenAIEmbeddings(openai_api_key=api_key)
vectordb = Chroma(persist_directory="./db", embedding_function=embedding)
llm_rag = OpenAI(openai_api_key=api_key, temperature=0)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm_rag,
    retriever=vectordb.as_retriever(),
    chain_type="stuff"
)

# 2. GPT 단독 (문서 기반 X)
llm_gpt = OpenAI(openai_api_key=api_key, temperature=0)

# 3. 질문 입력
question = "2025년 3월 학사일정표 내용 및 일정에 대해 알려줘"

# 4. RAG 답변
rag_answer = rag_chain.run(question)

# 5. GPT 단독 답변
gpt_answer = llm_gpt.invoke(question).strip()

# 6. 결과 출력
print("🧪 질문:", question)
print("\n✅ [RAG 기반 응답]")
print(rag_answer)
print("\n❌ [Open AI 단독 응답]")
print(gpt_answer)
