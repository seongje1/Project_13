import os
import getpass
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

# 🔐 OpenAI API Key 입력 받기
api_key = getpass.getpass("🔑 OpenAI API Key를 입력하세요: ")
os.environ["OPENAI_API_KEY"] = api_key

# 1. 벡터DB 다시 불러오기
embedding = OpenAIEmbeddings(openai_api_key=api_key)
vectordb = Chroma(persist_directory="./db", embedding_function=embedding)

# 2. 질문에 답할 LLM
llm = OpenAI(openai_api_key=api_key, temperature=0)

# 3. 검색 + 답변 연결
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff"
)

# 4. 예시 질문
question = "졸업하려면 몇 학점 들어야 하나요??"
answer = qa_chain.run(question)

print("❓ 질문:", question)
print("💬 답변:", answer)
