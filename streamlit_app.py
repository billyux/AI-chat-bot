
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from datetime import datetime
import sqlite3
import os
import tempfile

st.set_page_config(page_title="리포트 기반 종목 추천 챗봇", layout="centered")
st.title("📊 미래에셋 리포트 기반 종목 추천 챗봇")

api_key = st.sidebar.text_input("🔐 HyperCLOVA X API 키 입력", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    llm = OpenAI(temperature=0.7)
else:
    st.warning("API 키를 입력해주세요.")
    st.stop()

def init_db():
    conn = sqlite3.connect("feedback.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            investor_type TEXT,
            question TEXT,
            answer TEXT
        )""")
    conn.commit()
    conn.close()

def save_feedback(investor_type, question, answer):
    conn = sqlite3.connect("feedback.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO feedback (timestamp, investor_type, question, answer)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M"), investor_type, question, answer
    ))
    conn.commit()
    conn.close()

def fetch_report_links(limit=5):
    base = "https://securities.miraeasset.com"
    list_url = f"{base}/bbs/board/message/list.do?categoryId=1521"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(list_url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.select("div.board_list a"):
        title = a.text.strip()
        href = a.get("href")
        if href and "view.do" in href:
            links.append((title, base + href))
        if len(links) >= limit:
            break
    return links

def fetch_pdf_urls(report_links):
    headers = {"User-Agent": "Mozilla/5.0"}
    pdfs = []
    for title, detail_url in report_links:
        resp = requests.get(detail_url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        a = soup.find("a", href=lambda x: x and x.endswith(".pdf"))
        if a:
            pdfs.append((title, a["href"] if a["href"].startswith("http") else "https://securities.miraeasset.com" + a["href"]))
    return pdfs

def load_report_documents():
    report_links = fetch_report_links()
    pdf_urls = fetch_pdf_urls(report_links)
    docs = []
    for title, pdf_url in pdf_urls:
        try:
            resp = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
            tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(pdf_url))
            with open(tmp_path, "wb") as f:
                f.write(resp.content)
            loader = PyPDFLoader(tmp_path)
            docs += loader.load_and_split()
        except Exception as e:
            st.error(f"보고서 '{title}' 로딩 실패: {e}")
    return docs

def build_qa_chain():
    docs = load_report_documents()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return qa

init_db()
user_type = st.selectbox("투자 성향을 선택하세요", ["성장", "배당", "가치", "단타"])
question = st.text_input("관심 있는 질문이나 조건을 입력하세요")

if question:
    with st.spinner("리포트를 분석 중입니다..."):
        qa_chain = build_qa_chain()
        answer = qa_chain.run(question)
    st.success("📌 추천 결과")
    st.write(answer)
    save_feedback(user_type, question, answer)

    st.markdown("---")
    st.markdown("🧾 분석에 사용된 보고서 목록 및 링크:")
    links = fetch_report_links()
    for title, url in links:
        st.markdown(f"- [{title}]({url})")
