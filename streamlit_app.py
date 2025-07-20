import streamlit as st import requests from bs4 import BeautifulSoup from langchain.text_splitter import CharacterTextSplitter import openai from langchain.chat_models import ChatOpenAI from langchain.vectorstores import FAISS from langchain.chains import RetrievalQA from langchain.document_loaders import PyPDFLoader from datetime import datetime import sqlite3 import os import tempfile

커스텀 HyperCLOVA X 임베딩 래퍼 클래스

def HyperclovaEmbeddings(api_key=None, model="hyperclova-x-embedding", base_url=None): class _Wrapper: def init(self, api_key, model, base_url): self.model = model openai.api_key = api_key or os.getenv("OPENAI_API_KEY") if base_url: openai.api_base = base_url.rstrip("/") def embed_documents(self, texts): response = openai.Embedding.create( model=self.model, input=texts ) return [datum.embedding for datum in response.data] return _Wrapper(api_key, model, base_url)

페이지 설정

st.set_page_config(page_title="리포트 기반 종목 추천 챗봇", layout="centered") st.title("📊 미래에셋 리포트 기반 종목 추천 챗봇")

사이드바에서 API 정보 입력

api_key = st.sidebar.text_input("🔐 API 키 입력", type="password") base_url = st.sidebar.text_input("🌐 API Base URL", value="https://api.hyperclova.naver.com/v1")

if not api_key: st.warning("API 키를 입력해주세요.") st.stop()

환경변수 설정

os.environ["OPENAI_API_KEY"] = api_key os.environ["OPENAI_API_BASE"] = base_url

LLM 초기화

llm = ChatOpenAI( model_name="hyperclova-x-large", temperature=0.7, openai_api_key=api_key, openai_api_base=base_url )

임베딩 초기화

embeddings = HyperclovaEmbeddings(api_key=api_key, base_url=base_url)

DB 초기화 및 피드백 저장 함수

def init_db(): conn = sqlite3.connect("feedback.db") cur = conn.cursor() cur.execute(""" CREATE TABLE IF NOT EXISTS feedback ( id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, investor_type TEXT, question TEXT, answer TEXT )""") conn.commit() conn.close()

def save_feedback(investor_type, question, answer): conn = sqlite3.connect("feedback.db") cur = conn.cursor() cur.execute(""" INSERT INTO feedback (timestamp, investor_type, question, answer) VALUES (?, ?, ?, ?) """, ( datetime.now().strftime("%Y-%m-%d %H:%M"), investor_type, question, answer )) conn.commit() conn.close()

보고서 링크 및 PDF 추출 함수

def fetch_report_links(limit=5): base = "https://securities.miraeasset.com" list_url = f"{base}/bbs/board/message/list.do?categoryId=1521" headers = {"User-Agent": "Mozilla/5.0"} resp = requests.get(list_url, headers=headers) soup = BeautifulSoup(resp.text, "html.parser") links = [] for a in soup.select("div.board_list a"): title = a.text.strip() href = a.get("href") if href and "view.do" in href: links.append((title, base + href)) if len(links) >= limit: break return links

def fetch_pdf_urls(report_links): base = "https://securities.miraeasset.com" headers = {"User-Agent": "Mozilla/5.0"} pdfs = [] for title, detail_url in report_links: resp = requests.get(detail_url, headers=headers) soup = BeautifulSoup(resp.text, "html.parser") a = soup.find("a", href=lambda x: x and x.endswith(".pdf")) if a: href = a["href"] pdfs.append((title, href if href.startswith("http") else base + href)) return pdfs

QA 체인 구성 함수

def build_qa_chain(): report_links = fetch_report_links() pdf_urls = fetch_pdf_urls(report_links) docs = [] for title, pdf_url in pdf_urls: try: resp = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"}) tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(pdf_url)) with open(tmp_path, "wb") as f: f.write(resp.content) loader = PyPDFLoader(tmp_path) docs.extend(loader.load_and_split()) except Exception as e: st.error(f"보고서 '{title}' 로딩 실패: {e}") splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100) texts = splitter.split_documents(docs) vectordb = FAISS.from_documents(texts, embeddings) qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever()) return qa

메인 실행 로직

init_db() user_type = st.selectbox("투자 성향을 선택하세요", ["성장", "배당", "가치", "단타"]) question = st.text_input("관심 있는 질문이나 조건을 입력하세요")

if question: with st.spinner("리포트를 분석 중입니다..."): qa_chain = build_qa_chain() answer = qa_chain.run(question) st.success("📌 추천 결과") st.write(answer) save_feedback(user_type, question, answer) st.markdown("---") st.markdown("🧾 분석에 사용된 보고서 목록 및 링크:") for title, url in fetch_report_links(): st.markdown(f"- {title}")

