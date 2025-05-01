import os
import json
import re
import time
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from typing import List, Dict, Optional
import base64
from io import BytesIO
import numpy as np
import traceback
import openai
from langchain_core.embeddings import Embeddings
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate

import shutil
from pathlib import Path
import uvicorn

from competitive_analysis import build_rag_pipeline, analyze_competitor

app = FastAPI(title="Company Analysis API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration LinkedIn
EMAIL = ""
PASSWORD = ""

# Création du dossier pour stocker les fichiers uploadés
UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

# Variable globale pour stocker la chaîne RAG
rag_chain = None

# Modèles Pydantic pour l'API
class CompanyAnalysisState(BaseModel):
    company_name: str
    links: List[str] = []
    extracted_texts: List[str] = []
    summaries: List[str] = []
    strategy_recommendations: str = ""
    swot_lists: Optional[Dict[str, List[str]]] = None
    pestel_lists: Optional[Dict[str, List[str]]] = None
    swot_image_path: Optional[str] = None
    pestel_image_path: Optional[str] = None
    competitor_analysis: List[str] = []
    linkedin_analysis: Optional[str] = None
    rag_context: Optional[str] = None
    differentiation_strategy: Optional[str] = None

class CompanyAnalysisRequest(BaseModel):
    company_name: str

class CompanyAnalysisResponse(BaseModel):
    company_name: str
    summaries: List[str]
    competitor_analysis: List[str]
    strategy_recommendations: str
    swot_lists: Dict[str, List[str]]
    swot_image: str
    pestel_image: str
    porter_forces: Dict[str, List[str]]
    porter_image: str
    bcg_matrix: Dict[str, Dict[str, float]]
    bcg_image: str
    mckinsey_7s: Dict[str, str]
    mckinsey_image: str
    sources: List[str]
    linkedin_analysis: Optional[str] = None
    rag_context: Optional[str] = None

# Configuration des clés API
os.environ["TAVILY_API_KEY"] = ""
llm = ChatOpenAI(
    openai_api_key="",
    base_url="https://integrate.api.nvidia.com/v1",
    model="mistralai/mistral-small-24b-instruct",
    temperature=0.3,
)
search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.environ["TAVILY_API_KEY"]
)

# ... existing code ...

def generate_swot_image_base64(strengths, weaknesses, opportunities, threats):
    """Génère l'image SWOT et la retourne en base64"""
    fig = generate_swot_image(strengths, weaknesses, opportunities, threats)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return image_base64

def generate_pestel_image(pestel_lists: Optional[Dict[str, List[str]]] = None, filename: Optional[str] = None) -> plt.Figure:
    """Génère une image PESTEL avec la structure et les données"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Définition des couleurs et positions
    colors = {
        'P':'#E8D5B3','E':'#D5E8D4','S':'#D4E1F5',
        'T':'#F5D4E1','N':'#F5F0D4','L':'#E1D4F5',
    }
    rects = {
        'P':(0.00,0.5,0.33,0.5),'E':(0.33,0.5,0.33,0.5),
        'S':(0.66,0.5,0.34,0.5),'T':(0.00,0.0,0.33,0.5),
        'N':(0.33,0.0,0.33,0.5),'L':(0.66,0.0,0.34,0.5),
    }
    headings = {
        'P':'Political',
        'E':'Economic',
        'S':'Social',
        'T':'Technological',
        'N':'Environmental',
        'L':'legal',
    }
    
    # Mapping des clés PESTEL
    key_mapping = {
        'P': 'political',
        'E': 'economic',
        'S': 'social',
        'T': 'technological',
        'N': 'environmental',
        'L': 'legal'
    }
    
    # Dessin des rectangles, titres et points
    for k,(x,y,w,h) in rects.items():
        # Rectangle de fond
        ax.add_patch(Rectangle((x,y),w,h,facecolor=colors[k],edgecolor='none',zorder=0))
        
        # Titre
        ax.text(x+0.02,y+h-0.05,headings[k],
                fontsize=16,fontweight='bold',va='top',zorder=1)
        
        # Points
        if pestel_lists and key_mapping[k] in pestel_lists:
            points = pestel_lists[key_mapping[k]]
            for i, point in enumerate(points):
                ax.text(x+0.02,y+h-0.15-i*0.08,f"• {point}",
                       fontsize=12,va='top',zorder=1)
    
    if filename:
        plt.savefig(filename,bbox_inches='tight')
    return fig

def generate_pestel_image_base64(pestel_lists: Optional[Dict[str, List[str]]] = None):
    """Génère l'image PESTEL et la retourne en base64"""
    fig = generate_pestel_image(pestel_lists)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return image_base64

# Fonctions LinkedIn
def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def login_linkedin(driver):
    driver.get("https://www.linkedin.com/login")
    time.sleep(2)
    driver.find_element("id", "username").send_keys(EMAIL)
    driver.find_element("id", "password").send_keys(PASSWORD)
    driver.find_element("css selector", "button.btn__primary--large").click()
    time.sleep(3)

def fetch_and_scroll(driver, url: str) -> str:
    """Charge la page LinkedIn et fait défiler pour charger plus de contenu."""
    driver.get(url)
    time.sleep(3)  # Attendre le chargement initial
    
    # Faire défiler plusieurs fois pour charger plus de contenu
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_attempts = 0
    max_attempts = 3
    
    while scroll_attempts < max_attempts:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Attendre le chargement du nouveau contenu
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
            
        last_height = new_height
        scroll_attempts += 1
    
    return driver.page_source

def parse_company_with_posts_and_full(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    name_tag = soup.find(
        "h1",
        class_=lambda c: c and ("org-top-card-summary__title" in c or "text-heading-xlarge" in c)
    )
    about_tag = soup.find(attrs={"data-test-definition-text": True})
    posts = []
    for post_div in soup.find_all("div", class_=lambda c: c and "feed-shared-update-v2" in c):
        text_el = post_div.find("div", class_=lambda c: c and "feed-shared-text" in c)
        if text_el:
            posts.append(text_el.get_text(strip=True))
    main_tag = soup.find("main")
    full_text = main_tag.get_text("\n", strip=True) if main_tag else soup.get_text("\n", strip=True)
    return {
        "name": name_tag.get_text(strip=True) if name_tag else "",
        "description": about_tag.get_text(strip=True) if about_tag else "",
        "posts": "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(posts)),
        "full_text": full_text
    }

def get_linkedin_analysis(company_name: str) -> str:
    try:
        chrome_opts = Options()
        chrome_opts.add_argument("--headless")
        chrome_opts.add_argument("--disable-gpu")
        chrome_opts.add_argument("--no-sandbox")
        chrome_opts.add_argument("--disable-dev-shm-usage")
        
        try:
            service = Service(
                executable_path="/usr/local/bin/chromedriver",
                service_args=['--verbose']
            )
            
            driver = webdriver.Chrome(
                service=service,
                options=chrome_opts
            )
            
            driver.set_page_load_timeout(30)
            driver.implicitly_wait(10)
            
            try:
                login_linkedin(driver)

                company_id = company_name.lower().replace(" ", "-")
                url = f"https://www.linkedin.com/company/{company_id}"
                
                html = fetch_and_scroll(driver, url)
                data = parse_company_with_posts_and_full(html)
                
                if not data["full_text"]:
                    raise ValueError("Aucune donnée n'a été trouvée sur la page LinkedIn")

                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a strategic intelligence expert. Analyze the following LinkedIn information and provide a structured summary of the key points regarding the company, its recent activities, and its market presence."),
                    ("user", data["full_text"])
                ])
                
                formatted = prompt.format(content=data["full_text"])
                res = llm.invoke(formatted)
                content = res.content if hasattr(res, "content") else res
                
                # Formatage supplémentaire
                content = content.replace("\n", "<br>")
                # Nettoyer les balises br multiples
                content = re.sub(r'(<br>){3,}', '<br><br>', content)
                # Formater les titres
                content = re.sub(r'####\s*([^<]+)', r'<br><strong>\1</strong><br>', content)
                # Formater les puces
                content = re.sub(r'•\s*', '<br>• ', content)
                # Formater le texte en gras
                content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
                # Nettoyer les espaces multiples
                content = re.sub(r'\s+', ' ', content)
                
                return f"<strong>Analyse LinkedIn de {company_name}</strong><br><br>{content}"

            finally:
                driver.quit()

        except Exception as e:
            print(f"Erreur WebDriver: {str(e)}")
            return f"<strong>Note:</strong> L'analyse LinkedIn n'a pas pu être effectuée pour {company_name}. Erreur de configuration du navigateur: {str(e)}"

    except Exception as e:
        print(f"Erreur lors de l'analyse LinkedIn: {str(e)}")
        return f"<strong>Note:</strong> L'analyse LinkedIn n'a pas pu être effectuée pour {company_name}. Erreur: {str(e)}"

@app.post("/analyze")
async def analyze_company(
    company_name: str = Form(...),
    file: UploadFile = File(None)
) -> Dict:
    """
    Endpoint pour analyser une entreprise, avec ou sans document.
    Si un document est fourni, utilise RAG + recherche web.
    Sinon, utilise uniquement la recherche web.
    """
    try:
        # Initialisation de l'état
        state = CompanyAnalysisState(company_name=company_name)
        print("Starting analysis for company:", company_name)

        # Initialiser le RAG si un document est fourni
        rag_chain = None
        if file:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                rag_chain = build_rag_pipeline(
                    file_path=file_path,
                    persist_directory="chroma_db"
                )
                # Récupérer le contexte RAG
                if rag_chain:
                    rag_resp = rag_chain({"query": "Contexte stratégique général, positionnement et offre de notre entreprise"})
                    state.rag_context = rag_resp["result"]
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        # Recherche d'informations sur les concurrents
        print("Searching for information...")
        search_query = f"{state.company_name} concurrents"
        print(f"Search query: {search_query}")
        search_results = search_tool.invoke({"query": search_query})
        print(f"Search returned {len(search_results)} results")
        state.links = [r["url"] for r in search_results]
        print(f"Found {len(state.links)} links")
        print("Sample links:", state.links[:2] if state.links else "No links found")
        
        # Extraction et analyse des textes
        print("Extracting and analyzing texts...")
        state.extracted_texts = [extract_text_from_url(u) for u in state.links]
        print(f"Extracted {len(state.extracted_texts)} texts from URLs")
        print("Sample extracted text:", state.extracted_texts[0][:100] if state.extracted_texts else "No texts extracted")
        
        state.summaries = [summarize_competitor_info(t) for t in state.extracted_texts]
        print(f"Generated {len(state.summaries)} summaries")
        print("Sample summary:", state.summaries[0][:100] if state.summaries else "No summaries generated")
        print("Text analysis completed")
        
        # Génération des recommandations stratégiques avec RAG
        print("Generating strategic recommendations...")
        combined_summary = "\n\n".join(state.summaries)
        state.differentiation_strategy = suggest_strategic_differentiation(
            competitor_name=state.company_name,
            competitor_summary=combined_summary,
            rag_chain=rag_chain
        )
        
        # Génération SWOT
        print("Generating SWOT analysis...")
        state = swot_lists_step(state)
        
        # Génération PESTEL
        print("Generating PESTEL analysis...")
        state = pestel_lists_step(state)
        
        # Analyse LinkedIn
        print("Starting LinkedIn analysis...")
        linkedin_analysis = get_linkedin_analysis(state.company_name)
        
        # Génération des images en base64
        print("Generating visualizations...")
        swot_image = generate_swot_image_base64(
            state.swot_lists['strengths'],
            state.swot_lists['weaknesses'],
            state.swot_lists['opportunities'],
            state.swot_lists['threats']
        )
        
        pestel_image = generate_pestel_image_base64(state.pestel_lists)
        
        # Génération des analyses Porter, BCG et McKinsey
        print("Generating Porter's Five Forces...")
        porter_raw = generate_porter_forces(company_name)
        porter_forces = json.loads(porter_raw) if isinstance(porter_raw, str) else porter_raw
        porter_image = generate_porter_image(porter_forces)

        print("Generating BCG Matrix...")
        bcg_raw = generate_bcg_matrix(company_name)
        bcg_matrix = json.loads(bcg_raw) if isinstance(bcg_raw, str) else bcg_raw
        bcg_image = generate_bcg_image(bcg_matrix)

        print("Generating McKinsey 7S...")
        mckinsey_raw = generate_mckinsey_7s(company_name)
        mckinsey_7s = json.loads(mckinsey_raw) if isinstance(mckinsey_raw, str) else mckinsey_raw
        mckinsey_image = generate_mckinsey_image(mckinsey_7s)

        print("Preparing response...")
        response_data = {
            "company_name": str(state.company_name),
            "summaries": list(state.summaries) if state.summaries else [],
            "competitor_analysis": list(state.summaries) if state.summaries else [],
            "strategy_recommendations": str(state.differentiation_strategy),
            "swot_lists": dict(state.swot_lists) if state.swot_lists else {},
            "swot_image": str(swot_image),
            "pestel_image": str(pestel_image),
            "porter_forces": dict(porter_forces),
            "porter_image": str(porter_image),
            "bcg_matrix": dict(bcg_matrix),
            "bcg_image": str(bcg_image),
            "mckinsey_7s": dict(mckinsey_7s),
            "mckinsey_image": str(mckinsey_image),
            "sources": list(state.links) if state.links else [],
            "linkedin_analysis": str(linkedin_analysis) if linkedin_analysis else None,
            "rag_context": str(state.rag_context) if state.rag_context else None
        }
        
        # Création de l'objet CompanyAnalysisResponse avec les données validées
        response = CompanyAnalysisResponse(**response_data)
        return response.model_dump()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {
        "message": "Bienvenue sur l'API d'analyse d'entreprise",
        "endpoints": {
            "/analyze": "POST - Analyser une entreprise (body: {company_name: string})",
        }
    }

@tool
def extract_text_from_url(url: str) -> str:
    """Extrait le texte brut d'une page HTML (jusqu'à 5000 caractères)."""
    print(f"Extracting text from URL: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        print(f"HTTP Status Code: {r.status_code}")
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ").strip()[:5000]
        print(f"Extracted text length: {len(text)}")
        return text
    except Exception as e:
        print(f"Error extracting text from {url}: {str(e)}")
        return f"Erreur extraction {url}: {e}"

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an experienced strategic analyst."),
    ("user", "Here is the extracted content:\n{content}\n\nProvide only the key strategic insights.")
])

@tool
def summarize_competitor_info(content: str) -> str:
    """Retourne les points stratégiques clés du contenu fourni."""
    print(f"Summarizing content of length: {len(content)}")
    formatted = summary_prompt.format(content=content)
    res = llm.invoke(formatted)
    content = res.content if hasattr(res, "content") else res
    
    # Supprimer l'introduction si elle existe
    content = re.sub(r'^.*?1\.', '1.', content)
    
    # Formater les titres numérotés
    content = re.sub(r'(\d+\.\s*\*\*[^*]+\*\*)', r'<br><br><strong>\1</strong>', content)
    
    # Ajouter un saut de ligne avant chaque tiret de liste (mais pas les mots composés)
    content = re.sub(r'(?:^|\n|\s)\s*-\s+', r'<br><br>• ', content)
    
    # Formater le texte en gras
    content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
    
    # Nettoyer les espaces multiples
    content = re.sub(r'\s+', ' ', content)
    
    # Nettoyer les balises br multiples
    content = re.sub(r'(<br>){3,}', '<br><br>', content)
    
    print(f"Generated summary of length: {len(content)}")
    return content.strip()

# Prompt pour la différenciation stratégique (RAG + web)
diff_with_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Fred, an expert in strategic marketing and competitive differentiation."),
    ("user",
     "Internal context (from RAG for our company):\n{rag_context}\n\n"
     "External context (monitoring of competitor {competitor_name} and similar ones):\n{competitor_summary}\n\n"
     "Based on this, suggest 3 concrete differentiation strategies for our company (described in the RAG context) against {competitor_name} and similar competitors. For each strategy:\n"
     "1. Description of the opportunity\n"
     "2. Created competitive advantage\n"
     "3. Concrete implementation actions\n\n"
     "IMPORTANT: The company to differentiate is the one described in the RAG context, NOT {competitor_name}.")
])


def suggest_strategic_differentiation(
    competitor_name: str,
    competitor_summary: str,
    rag_chain: Optional[RetrievalQA] = None
) -> str:
    """Suggère des stratégies de différenciation basées sur l'analyse RAG interne et l'analyse des concurrents."""
    
    rag_context = ""
    if rag_chain:
        # Récupérer le contexte interne via RAG
        rag_resp = rag_chain({"query": "Contexte stratégique général, positionnement et offre de notre entreprise"})
        rag_context = rag_resp["result"]
    
    # Formater et appeler le LLM
    formatted = diff_with_rag_prompt.format(
        rag_context=rag_context if rag_context else "Pas de contexte RAG disponible.",
        competitor_name=competitor_name,
        competitor_summary=competitor_summary
    )
    resp = llm.invoke(formatted)
    content = getattr(resp, "content", resp)
    
    # Supprimer l'introduction si elle existe
    content = re.sub(r'^.*?###\s*Differentiation Strategy 1:', '<h3>Differentiation Strategy 1:', content)
    
    # Formater les titres des stratégies
    content = re.sub(r'###\s*Differentiation Strategy (\d):', r'<h3 class="text-xl font-bold mt-6 mb-4">Differentiation Strategy \1:</h3>', content)
    
    # Formater les sections principales
    content = re.sub(r'1\.\s*Description of the Opportunity:', r'<p class="font-semibold mt-4 mb-2">1. Description of the Opportunity:</p>', content)
    content = re.sub(r'2\.\s*Created Competitive Advantage:', r'<p class="font-semibold mt-4 mb-2">2. Created Competitive Advantage:</p>', content)
    content = re.sub(r'3\.\s*Concrete Implementation Actions:', r'<p class="font-semibold mt-4 mb-2">3. Concrete Implementation Actions:</p>', content)
    
    # Formater les listes d'actions
    content = re.sub(r'-\s+([^<\n]+)', r'<li class="ml-6 mb-2">• \1</li>', content)
    content = re.sub(r'(<li.*?</li>\s*)+', r'<ul class="list-none mt-2 mb-4">\n\1</ul>', content)
    
    # Formater les paragraphes de texte
    content = re.sub(r'([^>])\n([^<])', r'\1</p><p class="mb-4">\2', content)
    
    # Nettoyer les espaces multiples et les sauts de ligne
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'<\/p>\s*<p>', '</p><p>', content)
    
    # Ajouter des div pour chaque stratégie
    content = re.sub(
        r'(<h3[^>]*>Differentiation Strategy \d:.*?)(?=<h3|$)',
        r'<div class="strategy-section bg-white/5 rounded-lg p-6 mb-8">\1</div>',
        content,
        flags=re.DOTALL
    )
    
    return f'<div class="differentiation-strategies">{content}</div>'

swot_prompt = ChatPromptTemplate.from_messages([
("system",
 "You are a strategic analysis expert. For the company '{company_name}', "
 "provide exactly 5 strengths, 5 weaknesses, 5 opportunities, and 5 threats. "
 "Each item must contain 1 to 2 words maximum, without commas or conjunctions. "
 "Respond only with a JSON object containing four keys: "
 "strengths, weaknesses, opportunities, threats."),
("user", "{company_name}")

])

@tool
def generate_swot_lists(company_name: str) -> str:
    """Renvoie un JSON très court SWOT (4 listes de 5 items)."""
    payload = swot_prompt.format(company_name=company_name)
    res = llm.invoke(payload)
    return res.content if hasattr(res, "content") else res

def generate_swot_image(strengths, weaknesses, opportunities, threats, filename=None):
    """Génère l'image SWOT"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    colors = {'S':'#C0DFE5','W':'#FDD9B5','O':'#D9EECF','T':'#F7CFC7'}
    rects = {
        'S': (0.0,0.5,0.5,0.5),
        'W': (0.5,0.5,0.5,0.5),
        'O': (0.0,0.0,0.5,0.5),
        'T': (0.5,0.0,0.5,0.5),
    }

    # 1) Fond pastel
    for k,(x,y,w,h) in rects.items():
        ax.add_patch(Rectangle((x, y), w, h,
                               facecolor=colors[k],
                               edgecolor='none',
                               zorder=0))

    # 2) Titres et listes à puces
    headings = {'S':'Strengths','W':'Weaknesses','O':'Opportunities','T':'Threats'}
    data     = {'S':strengths,'W':weaknesses,'O':opportunities,'T':threats}
    for k,(x,y,w,h) in rects.items():
        ax.text(x + 0.02, y + h - 0.05,
                headings[k],
                fontsize=16, fontweight='bold', va='top', zorder=1)
        for i,item in enumerate(data[k]):
            ax.text(x + 0.02,
                    y + h - 0.12 - i*0.04,
                    u'\u2022 ' + item,
                    fontsize=12, va='top', zorder=1)

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    return fig

def swot_lists_step(state: CompanyAnalysisState) -> CompanyAnalysisState:
    raw = generate_swot_lists(state.company_name)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group(1))
        else:
            raise ValueError(f"Impossible de parser le JSON SWOT:\n{raw}")
    state.swot_lists = {k.strip().lower(): v for k,v in parsed.items()}
    return state

# Ajout des fonctions PESTEL que vous avez fournies
pestel_prompt = ChatPromptTemplate.from_messages([
("system",
 "You are a strategy expert. For the company '{company_name}', "
 "provide exactly 5 political, 5 economic, 5 social, "
 "5 technological, 5 environmental, and 5 legal factors. "
 "Each item must contain 1 to 2 words maximum, without commas or conjunctions. "
 "Respond only with a JSON object with 6 keys: "
 "political, economic, social, technological, environmental, legal."),
("user", "{company_name}")
])

@tool
def generate_pestel_lists(company_name: str) -> str:
    """Renvoie un JSON PESTEL (6 listes de 5 items très courts)."""
    payload = pestel_prompt.format(company_name=company_name)
    res = llm.invoke(payload)
    return res.content if hasattr(res, "content") else res

def pestel_lists_step(state: CompanyAnalysisState) -> CompanyAnalysisState:
    raw = generate_pestel_lists(state.company_name)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
        parsed = json.loads(m.group(1)) if m else {}
    state.pestel_lists = {k.strip().lower(): v for k,v in parsed.items()}
    return state

# Prompts pour les nouvelles analyses
porter_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert in strategic analysis. For the company \"{company_name}\", "
     "analyze Porter's Five Forces. For each of the five forces (including central competitive rivalry), "
     "provide exactly 3 factors of 1–2 words each. "
     "Respond ONLY with a JSON object containing these five keys: "
     "`rivalry`, `new_entrants`, `substitutes`, `buyer_power`, `supplier_power`."),
    ("user", "{company_name}")
])

bcg_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert in strategic analysis. For the company \"{company_name}\", "
     "identify 4 products/services (each named in 1–2 words) and position them on the BCG Matrix. "
     "For each product, estimate its relative market share (0–2) and its market growth rate (0–20). "
     "Respond ONLY with a JSON object with 4 keys (the product names), each containing an object with "
     "`market_share` and `growth_rate`."),
    ("user", "{company_name}")
])

mckinsey_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert in strategic analysis. For the company \"{company_name}\", "
     "analyze the McKinsey 7S Model. For each of the 7 elements, provide exactly 1–2 words. "
     "Respond ONLY with a JSON object containing seven keys: "
     "`strategy`, `structure`, `systems`, `style`, `staff`, `skills`, `shared_values`."),
    ("user", "{company_name}")
])

@tool
def generate_porter_forces(company_name: str) -> str:
    """Returns a JSON with Porter's Five Forces (5 lists of 3 items)."""
    payload = porter_prompt.format(company_name=company_name)
    res = llm.invoke(payload)
    raw = res.content if hasattr(res, "content") else res
    try:
        # Essayer de parser directement
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        # Si échec, chercher un objet JSON dans la réponse
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
        if m:
            # Vérifier que le JSON extrait est valide
            try:
                json.loads(m.group(1))
                return m.group(1)
            except json.JSONDecodeError:
                raise ValueError(f"JSON invalide dans la réponse Porter:\n{raw}")
        else:
            raise ValueError(f"Impossible de trouver un JSON dans la réponse Porter:\n{raw}")

@tool
def generate_bcg_matrix(company_name: str) -> str:
    """Returns a JSON mapping 4 products onto the BCG Matrix."""
    payload = bcg_prompt.format(company_name=company_name)
    res = llm.invoke(payload)
    raw = res.content if hasattr(res, "content") else res
    try:
        # Essayer de parser directement
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        # Si échec, chercher un objet JSON dans la réponse
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
        if m:
            # Vérifier que le JSON extrait est valide
            try:
                json.loads(m.group(1))
                return m.group(1)
            except json.JSONDecodeError:
                raise ValueError(f"JSON invalide dans la réponse BCG:\n{raw}")
        else:
            raise ValueError(f"Impossible de trouver un JSON dans la réponse BCG:\n{raw}")

@tool
def generate_mckinsey_7s(company_name: str) -> str:
    """Returns a JSON with the 7S elements as 1–2 word entries."""
    payload = mckinsey_prompt.format(company_name=company_name)
    res = llm.invoke(payload)
    raw = res.content if hasattr(res, "content") else res
    try:
        # Essayer de parser directement
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        # Si échec, chercher un objet JSON dans la réponse
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
        if m:
            # Vérifier que le JSON extrait est valide
            try:
                json.loads(m.group(1))
                return m.group(1)
            except json.JSONDecodeError:
                raise ValueError(f"JSON invalide dans la réponse McKinsey:\n{raw}")
        else:
            raise ValueError(f"Impossible de trouver un JSON dans la réponse McKinsey:\n{raw}")

# Fonctions de génération d'images
def generate_porter_image(forces: Dict[str, List[str]], filename: Optional[str] = None) -> str:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Positions
    center = (5, 5)
    force_positions = {
        "new_entrants": (5, 9),
        "supplier_power": (1.5, 3.5),
        "buyer_power": (8.5, 3.5),
        "substitutes": (5, 1),
    }
    force_titles = {
        "rivalry": "Competitive Rivalry",
        "new_entrants": "Threat of New Entrants",
        "supplier_power": "Bargaining Power of Suppliers",
        "buyer_power": "Bargaining Power of Buyers",
        "substitutes": "Threat of Substitutes"
    }

    # Cercle central pour la rivalité
    central = Circle(center, 2.2, fill=True, facecolor='#D9E8F5', edgecolor='#336699', linewidth=2)
    ax.add_patch(central)
    ax.text(center[0], center[1] + 0.4, force_titles["rivalry"],
            ha='center', va='center', fontsize=12, fontweight='bold')
    for i, item in enumerate(forces.get("rivalry", [])):
        y = center[1] - 0.1 - i * 0.4
        ax.text(center[0], y, f"• {item}", ha='center', va='center', fontsize=9)

    # Cercles extérieurs + flèches
    for force, pos in force_positions.items():
        circ = Circle(pos, 1.2, fill=True, facecolor='#E8F0F8', edgecolor='#336699', linewidth=2)
        ax.add_patch(circ)
        ax.text(pos[0], pos[1] + 0.3, force_titles[force],
                ha='center', va='center', fontsize=11, fontweight='bold')
        for j, item in enumerate(forces.get(force, [])):
            y = pos[1] + 0.1 - (j + 1) * 0.4
            ax.text(pos[0], y, f"• {item}", ha='center', va='center', fontsize=9)
        dx, dy = center[0] - pos[0], center[1] - pos[1]
        norm = (dx**2 + dy**2)**0.5
        start = (pos[0] + dx/norm*1.2, pos[1] + dy/norm*1.2)
        end = (center[0] - dx/norm*2.2, center[1] - dy/norm*2.2)
        ax.annotate("", xy=end, xytext=start,
                   arrowprops=dict(arrowstyle="->", color="#336699", lw=1.5))

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def generate_bcg_image(products: Dict[str, Dict[str, float]], filename: Optional[str] = None) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 20)
    ax.axvline(x=1, color='black', linestyle='-', alpha=0.5)
    ax.axhline(y=10, color='black', linestyle='-', alpha=0.5)

    # Quadrant titles
    ax.text(0.5, 15, "QUESTION MARK", ha='center', fontsize=12, fontweight='bold')
    ax.text(1.5, 15, "STAR", ha='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 1, "DOG", ha='center', fontsize=12, fontweight='bold')
    ax.text(1.5, 1, "CASH COW", ha='center', fontsize=12, fontweight='bold')

    # Background colors
    ax.add_patch(Rectangle((0, 10), 1, 10, facecolor='#FFD699', alpha=0.3))
    ax.add_patch(Rectangle((1, 10), 1, 10, facecolor='#99D6FF', alpha=0.3))
    ax.add_patch(Rectangle((0, 0), 1, 10, facecolor='#FF9999', alpha=0.3))
    ax.add_patch(Rectangle((1, 0), 1, 10, facecolor='#99FF99', alpha=0.3))

    ax.set_xlabel('Relative Market Share', fontweight='bold')
    ax.set_ylabel('Market Growth Rate (%)', fontweight='bold')

    # Bubbles
    default_size = 900
    for product, values in products.items():
        x = values.get('market_share', 0.5)
        y = values.get('growth_rate', 5)
        ax.scatter(x, y, s=default_size, alpha=0.6, edgecolors='black', linewidths=1)
        ax.text(x, y - 0.3, product, ha='center', va='center', fontweight='bold', fontsize=9)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def generate_mckinsey_image(model_7s: Dict[str, str], filename: Optional[str] = None) -> str:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    colors = {
        'strategy': '#FFB6C1',
        'structure': '#ADD8E6',
        'systems': '#FFDAB9',
        'style': '#98FB98',
        'staff': '#D8BFD8',
        'skills': '#FFFACD',
        'shared_values': '#E6E6FA'
    }

    center = (5, 5)
    radius = 1.2

    # Central circle: Shared Values
    central = Circle(center, radius, fill=True, facecolor=colors['shared_values'],
                    edgecolor='black', linewidth=1.5)
    ax.add_patch(central)
    ax.text(center[0], center[1] + 0.3, "Shared Values",
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(center[0], center[1] - 0.3, model_7s.get('shared_values', ''),
            ha='center', va='center', fontsize=9, wrap=True)

    # Outer elements
    elements = [
        ('strategy', 'Strategy'),
        ('structure', 'Structure'),
        ('systems', 'Systems'),
        ('style', 'Style'),
        ('staff', 'Staff'),
        ('skills', 'Skills')
    ]
    angles = np.linspace(0, 2*np.pi, len(elements), endpoint=False)
    radius_outer = 3.5

    for (key, title), angle in zip(elements, angles):
        x = center[0] + radius_outer * np.cos(angle)
        y = center[1] + radius_outer * np.sin(angle)
        circ = Circle((x, y), radius, fill=True,
                     facecolor=colors[key], edgecolor='black', linewidth=1.5)
        ax.add_patch(circ)
        ax.text(x, y + 0.3, title,
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x, y - 0.3, model_7s.get(key, ''),
                ha='center', va='center', fontsize=9, wrap=True)
        ax.add_line(plt.Line2D([center[0], x], [center[1], y],
                              color='gray', linestyle='-', linewidth=1.5, alpha=0.6))

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
