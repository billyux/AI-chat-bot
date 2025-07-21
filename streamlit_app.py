import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_naver import ChatClovaX, ClovaXEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from datetime import datetime
import sqlite3
import os
import tempfile

# 페이지 설정
st.set_page_config(page_title="리포트 기반 종목 추천 챗봇", layout="centered")
st.title("📊 미래에셋 리포트 기반 종목 추천 챗봇")

# 사이드바에서 API 키 입력
api_key = st.sidebar.text_input("🔐 CLOVA API 키 입력", type="password")
if not api_key:
    st.warning("API 키를 입력해주세요.")
    st.stop()

# CLOVA Studio API 키 환경변수 설정
os.environ["CLOVA_API_KEY"] = api_key

# LLM 및 임베딩 초기화
llm = ChatClovaX(model="hyperclova-x-large")
embeddings = ClovaXEmbeddings(model="hyperclova-x-embedding")

# DB 초기화 및 피드백 저장 함수
def init_db():
    conn = sqlite3.connect("feedback.db")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            investor_type TEXT,
            question TEXT,
            answer TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def save_feedback(investor_type, question, answer):
    conn = sqlite3.connect("feedback.db")
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO feedback (timestamp, investor_type, question, answer)
        VALUES (?, ?, ?, ?)
        """,
        (datetime.now().strftime("%Y-%m-%d %H:%M"), investor_type, question, answer)
    )
    conn.commit()
    conn.close()

# 보고서 링크 및 PDF 추출
def fetch_report_links(limit=5):
    base = "https://securities.miraeasset.com"
    url = f"{base}/bbs/board/message/list.do?categoryId=1521"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
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
    base = "https://securities.miraeasset.com"
    pdfs = []
    for title, detail_url in report_links:
        resp = requests.get(detail_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        a = soup.find("a", href=lambda x: x and x.endswith(".pdf"))
        if a:
            href = a["href"]
            full = href if href.startswith("http") else base + href
            pdfs.append((title, full))
    return pdfs

# QA 체인 구축
def build_qa_chain():
    # 보고서 로딩
    docs = []
    for title, pdf_url in fetch_pdf_urls(fetch_report_links()):
        try:
            data = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"}).content
            tmp = os.path.join(tempfile.gettempdir(), os.path.basename(pdf_url))
            with open(tmp, "wb") as f:
                f.write(data)
            docs.extend(PyPDFLoader(tmp).load_and_split())
        except Exception as e:
            st.error(f"보고서 '{title}' 로딩 실패: {e}")
    # 텍스트 분할 & 벡터 생성
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(texts, embeddings)
    # RetrievalQA 체인 구성
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

# 메인 실행
init_db()
user_type = st.selectbox("투자 성향을 선택하세요", ["성장", "배당", "가치", "단타"])
question = st.text_input("관심 있는 질문이나 조건을 입력하세요")

if question:
    with st.spinner("분석 중입니다..."):
        qa = build_qa_chain()
        answer = qa.run(question)
    st.success("📌 추천 결과")
    st.write(answer)
    save_feedback(user_type, question, answer)

    st.markdown("---")
    st.markdown("🧾 사용된 보고서:")
    for title, url in fetch_report_links():
        st.markdown(f"- [{title}]({url})")
