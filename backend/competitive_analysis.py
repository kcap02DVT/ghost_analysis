import os
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import List, Dict, Optional
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import openai

# --- Clés API ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

class NVIDIAEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str = NVIDIA_API_KEY,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model_name: str = "baai/bge-m3",
        encoding_format: str = "float",
        truncate: str = "NONE",
    ):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.encoding_format = encoding_format
        self.truncate = truncate
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 8
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                input=batch_texts,
                model=self.model_name,
                encoding_format=self.encoding_format,
                extra_body={"truncate": self.truncate}
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name,
            encoding_format=self.encoding_format,
            extra_body={"truncate": self.truncate}
        )
        return response.data[0].embedding

def build_rag_pipeline(file_path: str, persist_directory: Optional[str] = None) -> RetrievalQA:
    # Charger le document
    if file_path.lower().endswith(".pdf"):
        loader = UnstructuredPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf8")
    docs = loader.load()

    # Découper en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embeddings + ChromaDB
    embeddings = NVIDIAEmbeddings()
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    if persist_directory:
        vectordb.persist()

    # Construire la RetrievalQA chain
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm_rag = ChatOpenAI(
        openai_api_key=NVIDIA_API_KEY,
        base_url="https://integrate.api.nvidia.com/v1",
        model="mistralai/mistral-small-24b-instruct",
        temperature=0
    )
    return RetrievalQA.from_chain_type(
        llm=llm_rag,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

class CompanyAnalysisState(BaseModel):
    company_name: str
    links: List[str] = []
    extracted_texts: List[str] = []
    summaries: List[str] = []
    strategy_recommendations: str = ""

# Configurer LLM principal
llm = ChatOpenAI(
    openai_api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    model="mistralai/mistral-small-24b-instruct",
    temperature=0.3,
)

# Outil de recherche Tavily
search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=TAVILY_API_KEY
)

@tool
def extract_text_from_url(url: str) -> str:
    """Extrait le texte brut d'une page web en utilisant BeautifulSoup.
    
    Args:
        url: L'URL de la page à extraire
        
    Returns:
        Le texte extrait (limité à 5000 caractères) ou un message d'erreur
    """
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=' ')[:5000]
    except Exception as e:
        return f"Erreur extraction {url}: {e}"

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un analyste stratégique expérimenté spécialisé dans l'étude des concurrents."),
    ("user", "Voici le contenu extrait concernant un concurrent potentiel de notre entreprise {company_name}:\n{content}\n\nDonne uniquement les éléments stratégiques clés de ce concurrent (cible, positionnement, forces, faiblesses).")
])

def summarize_with_company_context(content: str, company_name: str) -> str:
    formatted = summary_prompt.format(content=content, company_name=company_name)
    resp = llm.invoke(formatted)
    return getattr(resp, "content", resp)

diff_with_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es Fred, expert en marketing stratégique et différenciation concurrentielle."),
    ("user",
     "Contexte interne (issu de la RAG pour notre entreprise):\n{rag_context}\n\n"
     "Contexte externe (veille sur le concurrent {competitor_name} et autres similaires):\n{competitor_summary}\n\n"
     "Sur ces bases, propose 3 axes concrets de différenciation pour notre entreprise (décrite dans le contexte RAG) face à {competitor_name} et ses similaires.")
])

def suggest_strategic_differentiation(
    competitor_name: str,
    competitor_summary: str,
    rag_chain: RetrievalQA
) -> str:
    rag_resp = rag_chain({"query": "Contexte stratégique général, positionnement et offre de notre entreprise"})
    rag_context = rag_resp["result"]

    formatted = diff_with_rag_prompt.format(
        rag_context=rag_context,
        competitor_name=competitor_name,
        competitor_summary=competitor_summary
    )
    resp = llm.invoke(formatted)
    return getattr(resp, "content", resp)

def build_analysis_graph() -> StateGraph:
    graph = StateGraph(CompanyAnalysisState)

    def search_step(state: CompanyAnalysisState):
        results = search_tool.invoke({"query": f"informations sur {state.company_name} et entreprises similaires secteur santé digital"})
        state.links = [r.get("url") for r in results if r.get("url")]
        return state

    def extract_step(state: CompanyAnalysisState):
        state.extracted_texts = [extract_text_from_url(url) for url in state.links]
        return state

    def summarize_step(state: CompanyAnalysisState):
        state.summaries = [
            summarize_with_company_context(text, company_name=state.company_name) 
            for text in state.extracted_texts
        ]
        return state

    def strategy_step(state: CompanyAnalysisState, rag_chain: RetrievalQA):
        combined = "\n\n".join(state.summaries)
        state.strategy_recommendations = suggest_strategic_differentiation(
            competitor_name=state.company_name,
            competitor_summary=combined,
            rag_chain=rag_chain
        )
        return state

    # Brancher les étapes
    graph.add_node("search", search_step)
    graph.add_node("extract", extract_step)
    graph.add_node("summarize", summarize_step)
    graph.add_node("strategy", strategy_step)

    graph.set_entry_point("search")
    graph.add_edge("search", "extract")
    graph.add_edge("extract", "summarize")
    graph.add_edge("summarize", "strategy")

    return graph

async def analyze_competitor(file_path: str, company_name: str) -> Dict:
    # 1. Construire le RAG
    rag_chain = build_rag_pipeline(file_path, persist_directory="chroma_db")

    # 2. Initialiser et compiler le graph
    graph = build_analysis_graph()
    compiled = graph.compile()

    # 3. Exécuter l'analyse
    state = CompanyAnalysisState(company_name=company_name)
    final_state = compiled.invoke(state)

    # 4. Retourner les résultats
    return {
        "company_name": company_name,
        "summaries": final_state.summaries,
        "strategy_recommendations": final_state.strategy_recommendations,
        "links": final_state.links
    } 