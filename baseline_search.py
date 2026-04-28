"""
baseline_search.py
==================
End-to-end search + eval pipeline for all 10 profession configs.

Usage:
    python SearchRankingRound/baseline_search.py                             # baseline (ANN order)
    python SearchRankingRound/baseline_search.py --rerank llm               # GPT-4o scoring
    python SearchRankingRound/baseline_search.py --rerank bm25              # BM25 on filtered pool
    python SearchRankingRound/baseline_search.py --rerank dense             # FAISS bi-encoder
    python SearchRankingRound/baseline_search.py --rerank hybrid            # BM25 + Dense RRF
    python SearchRankingRound/baseline_search.py --rerank crossencoder      # cross-encoder (query, doc)
    python SearchRankingRound/baseline_search.py --config tax_lawyer        # single config
    python SearchRankingRound/baseline_search.py --config tax_lawyer --dry-run  # skip eval POST

Install deps:
    pip install turbopuffer voyageai requests openai rank_bm25 sentence-transformers faiss-cpu
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import os

from dotenv import load_dotenv
import openai
import requests
import turbopuffer
import voyageai

load_dotenv()

# Make enhancements.py importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent))
from enhancements import (
    candidate_to_text,
    LocalBM25,
    DenseRetriever,
    HybridRetriever,
    CrossEncoderReranker,
)

# ── Credentials (loaded from .env) ────────────────────────────────────────────

TURBOPUFFER_API_KEY = os.environ["TURBOPUFFER_API_KEY"]
VOYAGE_API_KEY      = os.environ["VOYAGE_API_KEY"]
OAI_KEY             = os.environ["OAI_KEY"]
EVAL_EMAIL          = os.environ["EVAL_EMAIL"]
EVAL_ENDPOINT       = "https://mercor-dev--search-eng-interview.modal.run/evaluate"
TPUF_NAMESPACE      = "search-test-v4"
TPUF_REGION         = "aws-us-west-2"

# ── Query configs (all 10 professions) ────────────────────────────────────────
#
# Fields:
#   config_path        : .yml name used by the eval endpoint (server-side)
#   query              : semantic string embedded for ANN retrieval
#   hard               : hard filter criteria — failing ANY = final_score 0
#   soft               : hints for re-ranker (not enforced, used for scoring)
#
# Hard filter keys supported by passes_hard_filters():
#   require_degrees        list[str]  – must match at least one (lowercase, e.g. "jd")
#   require_fos_keywords   list[str]  – at least one must appear in deg_fos (substring)
#   require_title_keywords list[str]  – at least one must appear in exp_titles (substring)
#   require_min_exp_bucket int        – max(exp_years buckets) must be >= this value
#   require_max_exp_bucket int        – max(exp_years buckets) must be <= this value
#   require_currently_employed bool   – must have at least one current role

QUERY_CONFIGS = [
    {
        "config_path": "tax_lawyer.yml",
        "query": (
            "tax attorney JD law corporate tax planning IRS compliance "
            "transactions legal counsel tax litigation"
        ),
        "hard": {
            "require_degrees": ["jd"],
            "require_min_exp_bucket": 3,   # 3+ years confirmed hard criterion
        },
        "soft": {
            # confirmed from eval response:
            # corporate_transaction_experience, irs_audit_experience, legal_writing_expertise
        },
    },
    {
        "config_path": "junior_corporate_lawyer.yml",
        "query": (
            "junior associate corporate lawyer law firm USA Europe Canada "
            "contracts M&A 2-4 years international regulatory compliance"
        ),
        "hard": {
            # xlsx: "Graduate of reputed law school in USA/Europe/Canada" — no JD required
            "require_fos_keywords": ["law", "legal", "jurisprudence"],
            "require_min_exp_bucket": 1,   # 2+ years
            "require_max_exp_bucket": 3,   # junior = not 5+ or 10+ year veterans
        },
        "soft": {},
    },
    {
        "config_path": "radiology.yml",
        "query": (
            "radiologist MD MBBS medical doctor imaging diagnostic radiology "
            "fellowship X-ray CT MRI interpretation India US"
        ),
        "hard": {
            # Fix: Indian/Pakistani MDs hold MBBS (stored as "bachelor's" not "doctorate")
            # Use deg_fos for medicine keywords instead of degree type
            "require_fos_keywords": [
                "medicine", "mbbs", "m.b.b.s", "mbchb", "doctor of medicine",
                "medicine and surgery", "clinical medicine",
            ],
            "require_title_keywords": ["radiolog"],
        },
        "soft": {},
    },
    {
        "config_path": "doctors_md.yml",
        "query": (
            "general practitioner family medicine physician MD primary care "
            "outpatient US top medical school clinical practice telemedicine"
        ),
        "hard": {
            # KEY FINDING: MD degrees are stored as "md" in deg_degrees, NOT "doctorate"
            "require_degrees": ["md"],
            "require_min_exp_bucket": 1,        # 2+ years clinical practice
            "require_title_keywords": [
                "general practitioner", "family medicine", "primary care",
                "family physician", "gp", "physician",
            ],
        },
        "soft": {},
    },
    {
        "config_path": "biology_expert.yml",
        "query": (
            "biologist PhD molecular biology US university research laboratory "
            "genetics cell biology publications biochemistry"
        ),
        "hard": {
            "require_degrees": ["doctorate"],
            "require_us_uk_canada_undergrad": True,  # xlsx: undergrad in US/UK/Canada
            "require_fos_keywords": [
                "biology", "biochemistry", "molecular", "genetics",
                "microbiology", "cell biology", "biological"
            ],
        },
        "soft": {},
    },
    {
        "config_path": "anthropology.yml",
        "query": (
            "PhD student anthropology sociology economics ethnography fieldwork "
            "cultural labor migration top US university dissertation recent 2022 2023 2024"
        ),
        "hard": {
            "require_degrees": ["doctorate"],
            "require_doctorate_start_year": 2022,   # "started within last 3 years"
            "require_fos_keywords": [
                "anthropology", "archaeology", "ethnography",
                "cultural studies", "social anthropology",
                "sociology", "economics",
            ],
        },
        "soft": {},
    },
    {
        "config_path": "mathematics_phd.yml",
        "query": (
            "mathematician PhD pure applied mathematics US university research "
            "postdoc academia proof theory numerical analysis statistics"
        ),
        "hard": {
            "require_degrees": ["doctorate"],
            "require_us_uk_canada_undergrad": True,  # xlsx: undergrad in US/UK/Canada
            "require_fos_keywords": [
                "math", "mathematics", "applied mathematics",
                "statistics", "pure mathematics"
            ],
        },
        "soft": {},
    },
    {
        "config_path": "quantitative_finance.yml",
        "query": (
            "MBA Harvard Wharton Booth Kellogg Sloan Columbia Stanford "
            "Wall Street investment bank portfolio manager quantitative finance "
            "risk derivatives trading"
        ),
        "hard": {
            # Fix: drop require_us_mba school filter — it was eliminating everyone.
            # ANN query now targets M7 MBA finance profiles directly.
            # Server will gate on m7_mba; we just need to get MBAs with finance exp.
            "require_degrees": ["mba"],
            "require_min_exp_bucket": 3,       # 3+ years
        },
        "soft": {},
    },
    {
        "config_path": "bankers.yml",
        "query": (
            "investment banker MBA healthcare mergers acquisitions M&A "
            "corporate finance deal execution private equity"
        ),
        "hard": {
            "require_degrees": ["mba"],
            "require_us_mba": True,            # must be US university
            "require_min_exp_bucket": 1,       # 2+ years IB/corporate finance/M&A
            "require_title_keywords": [
                "analyst", "associate", "banker",
                "investment", "finance", "banking"
            ],
        },
        "soft": {
            # confirmed from eval response:
            # healthcare_investment_banking_experience,
            # healthcare_ma_transactions, healthcare_metrics_knowledge
        },
    },
    {
        "config_path": "mechanical_engineers.yml",
        "query": (
            "mechanical engineer BSME CAD FEA CFD design manufacturing "
            "product development thermodynamics ANSYS SolidWorks"
        ),
        "hard": {
            "require_fos_keywords": [
                "mechanical engineering", "aerospace engineering",
                "manufacturing engineering", "mechanical"
            ],
            "require_title_keywords": [
                "engineer", "mechanical", "design",
                "manufacturing", "product development"
            ],
            "require_min_exp_bucket": 3,       # 3+ years professional experience
        },
        "soft": {},
    },
]

# Keyed lookup for easy CLI access
CONFIGS_BY_NAME = {
    c["config_path"].replace(".yml", ""): c for c in QUERY_CONFIGS
}

# ── Clients ───────────────────────────────────────────────────────────────────

def make_tpuf_namespace():
    client = turbopuffer.Turbopuffer(api_key=TURBOPUFFER_API_KEY, region=TPUF_REGION)
    return client.namespace(TPUF_NAMESPACE)

def make_voyage_client():
    return voyageai.Client(api_key=VOYAGE_API_KEY)

def make_openai_client():
    return openai.OpenAI(api_key=OAI_KEY)

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_query(voyage: voyageai.Client, text: str) -> list[float]:
    response = voyage.embed(text, model="voyage-3")
    return response.embeddings[0]

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(ns, embedding: list[float], top_k: int = 200) -> list:
    """ANN retrieval from Turbopuffer. Returns raw result rows."""
    result = ns.query(
        rank_by=("vector", "ANN", embedding),
        top_k=top_k,
        include_attributes=True,
    )
    return result.rows or []

# ── Hard filtering ────────────────────────────────────────────────────────────

US_MBA_SCHOOLS = {
    "harvard", "wharton", "stanford", "booth", "kellogg", "sloan", "haas",
    "columbia", "tuck", "stern", "fuqua", "ross", "darden", "yale", "anderson",
    "tepper", "mccombs", "johnson", "kenan-flagler", "olin", "fisher", "mendoza",
    "kelley", "marshall", "smith", "moore", "goizueta", "babcock", "weatherhead",
    "mit", "duke", "northwestern", "michigan", "virginia", "cornell", "nyu",
    "carnegie mellon", "emory", "unc", "vanderbilt", "notre dame", "georgetown",
    "boston university", "bu", "american", "fordham", "tulane",
    # full-name variants that substring-matching might miss
    "university of chicago", "university of pennsylvania", "university of michigan",
    "university of virginia", "university of california", "university of texas",
    "massachusetts institute", "dartmouth college",
}

TOP_US_MED_SCHOOLS = {
    "johns hopkins", "harvard", "stanford", "ucsf", "mayo", "columbia",
    "perelman", "yale", "duke", "university of michigan", "vanderbilt",
    "baylor", "weill cornell", "cornell", "washington university", "wustl",
    "emory", "northwestern", "feinberg", "pittsburgh", "ucla", "ucsd",
    "uc san diego", "uc davis", "uc irvine", "university of virginia",
    "unc", "university of north carolina", "university of minnesota",
    "university of wisconsin", "keck", "mount sinai", "icahn",
    "dartmouth", "tufts", "boston university", "university of chicago",
    "pritzker", "case western", "ut southwestern", "university of texas",
    "ohio state", "indiana university", "university of florida",
    "university of colorado", "nyu", "grossman", "rush",
    "albert einstein", "jefferson", "temple", "suny", "stony brook",
    "rochester", "university of buffalo", "brown", "university of cincinnati",
    "university of louisville", "university of kentucky", "tulane",
    "university of miami", "florida state", "university of arizona",
    "university of utah", "oregon health", "creighton", "loyola",
    "george washington", "georgetown", "thomas jefferson",
}

# US/UK/Canada institution keywords for undergrad location checks
US_UK_CANADA_UNIV_KEYWORDS = {
    # US — specific institutions
    "mit", "harvard", "stanford", "yale", "princeton", "columbia", "cornell",
    "brown", "dartmouth", "upenn", "northwestern", "duke", "university of chicago",
    "johns hopkins", "caltech", "carnegie mellon", "rice", "emory", "georgetown",
    "vanderbilt", "notre dame", "wake forest", "tufts", "brandeis", "tulane",
    "nyu", "new york university", "boston university", "northeastern", "drexel",
    "george washington", "american university", "rensselaer", "university of rochester",
    "case western", "lehigh", "villanova", "fordham",
    "university of michigan", "university of virginia", "university of north carolina",
    "university of texas", "university of florida", "university of washington",
    "university of illinois", "university of wisconsin", "university of minnesota",
    "university of colorado", "university of arizona", "university of california",
    "university of southern california", "university of pittsburgh",
    "ohio state", "penn state", "michigan state", "purdue", "indiana university",
    "virginia tech", "georgia tech", "rutgers", "suny", "cuny",
    "uc berkeley", "ucla", "ucsd", "ucsb", "uc davis",
    # UK
    "oxford", "cambridge", "imperial college", "ucl", "university college london",
    "lse", "london school of economics", "university of edinburgh",
    "university of manchester", "university of bristol", "university of glasgow",
    "university of warwick", "university of exeter", "university of bath",
    "university of nottingham", "university of birmingham", "university of leeds",
    "university of sheffield", "university of southampton", "durham university",
    "king's college london", "queen mary", "university of liverpool",
    "newcastle university", "university of st andrews", "heriot-watt",
    # Canada
    "university of toronto", "mcgill", "university of british columbia",
    "university of waterloo", "queen's university", "western university",
    "university of alberta", "university of calgary", "dalhousie",
    "simon fraser", "university of ottawa", "york university",
    "universite de montreal", "concordia", "laval",
}


def _max_exp_bucket(attrs: dict) -> int:
    buckets = [int(y) for y in (attrs.get("exp_years") or []) if str(y).isdigit()]
    return max(buckets) if buckets else 0


def _has_us_mba(attrs: dict) -> bool:
    """Check MBA from a known US university."""
    degrees = [d.lower() for d in (attrs.get("deg_degrees") or [])]
    if "mba" not in degrees:
        return False
    schools_blob = " ".join(attrs.get("deg_schools") or []).lower()
    return any(school in schools_blob for school in US_MBA_SCHOOLS)


def _has_top_us_md(attrs: dict) -> bool:
    """Check MD degree from a top US medical school (for doctors_md config)."""
    degrees = [d.lower() for d in (attrs.get("deg_degrees") or [])]
    schools = attrs.get("deg_schools") or []
    for i, deg in enumerate(degrees):
        if deg == "md" and i < len(schools):      # MD stored as "md" not "doctorate"
            school = schools[i].lower()
            if any(kw in school for kw in TOP_US_MED_SCHOOLS):
                return True
    return False


def _has_us_uk_canada_undergrad(attrs: dict) -> bool:
    """Check if the bachelor's degree is from a US/UK/Canada institution."""
    degrees = [d.lower() for d in (attrs.get("deg_degrees") or [])]
    schools = attrs.get("deg_schools") or []
    for i, deg in enumerate(degrees):
        if deg == "bachelor's" and i < len(schools):
            school = schools[i].lower()
            if any(kw in school for kw in US_UK_CANADA_UNIV_KEYWORDS):
                return True
    return False


def _has_recent_doctorate_start(attrs: dict, min_year: int) -> bool:
    """Check if any doctorate degree started in min_year or later."""
    degrees = [d.lower() for d in (attrs.get("deg_degrees") or [])]
    start_years = attrs.get("deg_start_years") or []
    for i, deg in enumerate(degrees):
        if deg == "doctorate" and i < len(start_years):
            try:
                if int(start_years[i]) >= min_year:
                    return True
            except (ValueError, TypeError):
                pass
    return False


def passes_hard_filters(attrs: dict, hard: dict, config_path: str = "") -> bool:
    """
    Apply hard criteria filters. Returns False (drops candidate) if any criterion fails.

    Supported keys in `hard`:
      require_degrees                list[str]  – must have at least one (lowercase)
      require_us_mba                 bool       – MBA from a known US school
      require_us_md                  bool       – doctorate from a top US medical school
      require_us_uk_canada_undergrad bool       – bachelor's from US/UK/Canada institution
      require_doctorate_start_year   int        – doctorate must have started >= this year
      require_fos_keywords           list[str]  – at least one substring must match deg_fos
      require_title_keywords         list[str]  – at least one substring must match exp_titles
      require_min_exp_bucket         int        – max exp bucket must be >= this
      require_max_exp_bucket         int        – max exp bucket must be <= this
      require_currently_employed     bool       – must have a current role
    """
    if "require_degrees" in hard:
        cand_degrees = {d.lower() for d in (attrs.get("deg_degrees") or [])}
        required = {d.lower() for d in hard["require_degrees"]}
        if not cand_degrees & required:
            return False

    if hard.get("require_us_mba"):
        if not _has_us_mba(attrs):
            return False

    if hard.get("require_us_md"):
        if not _has_top_us_md(attrs):
            return False

    if hard.get("require_us_uk_canada_undergrad"):
        if not _has_us_uk_canada_undergrad(attrs):
            return False

    if "require_doctorate_start_year" in hard:
        if not _has_recent_doctorate_start(attrs, hard["require_doctorate_start_year"]):
            return False

    if "require_fos_keywords" in hard:
        fos_blob = " ".join(attrs.get("deg_fos") or []).lower()
        if not any(kw.lower() in fos_blob for kw in hard["require_fos_keywords"]):
            return False

    if "require_title_keywords" in hard:
        titles_blob = " ".join(attrs.get("exp_titles") or []).lower()
        if not any(kw.lower() in titles_blob for kw in hard["require_title_keywords"]):
            return False

    if "require_min_exp_bucket" in hard:
        if _max_exp_bucket(attrs) < hard["require_min_exp_bucket"]:
            return False

    if "require_max_exp_bucket" in hard:
        if _max_exp_bucket(attrs) > hard["require_max_exp_bucket"]:
            return False

    if hard.get("require_currently_employed"):
        if not any(e == "" for e in (attrs.get("exp_end_years") or [])):
            return False

    return True

# ── Re-ranking ────────────────────────────────────────────────────────────────

def rerank(rows: list, query: str, config: dict) -> list:
    """Baseline: return rows in ANN score order (already sorted by TPUF distance)."""
    return rows


LLM_RERANK_POOL = 50   # how many hard-filtered candidates to send to the LLM

def rerank_llm(rows: list, query: str, config: dict, oai_client: openai.OpenAI) -> list:
    """
    Re-rank using a single GPT-4o call that scores up to LLM_RERANK_POOL candidates.
    Candidates are pre-sorted by ANN distance, so slicing to LLM_RERANK_POOL keeps
    the most semantically relevant ones before scoring.
    """
    candidates = rows[:LLM_RERANK_POOL]
    profession = config["config_path"].replace(".yml", "").replace("_", " ").title()

    lines = []
    for i, row in enumerate(candidates):
        summary = (row.model_extra or {}).get("rerankSummary", "") or ""
        lines.append(f"[{i}] {summary[:800]}")

    prompt = (
        f"You are a senior recruiter evaluating candidates for: {profession}\n"
        f"Role criteria: {query}\n\n"
        "Score each candidate 1-10 on fit for this specific role. "
        "10 = exceptional match, 1 = poor match. "
        "Consider seniority, domain depth, and directly relevant experience.\n\n"
        "Return ONLY a JSON object with key \"scores\" containing an array of integers "
        "in the same order as the candidates below.\n\n"
        + "\n\n".join(lines)
    )

    t0 = time.time()
    response = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    scores = json.loads(response.choices[0].message.content).get("scores", [])
    print(f"  LLM scored {len(candidates)} candidates in {time.time()-t0:.1f}s")

    scored = sorted(
        range(len(candidates)),
        key=lambda i: scores[i] if i < len(scores) else 0,
        reverse=True,
    )
    reranked = [candidates[i] for i in scored]

    # Append any remaining rows (beyond LLM_RERANK_POOL) at the end
    reranked.extend(rows[LLM_RERANK_POOL:])
    return reranked


# ── Helpers: Row list ↔ RankedList conversion ────────────────────────────────

def _reranked_rows(rows: list, ranked: list) -> list:
    """Convert a RankedList [(id, score)] back to an ordered list of Row objects."""
    id_to_row = {str(r.id): r for r in rows}
    result = [id_to_row[cid] for cid, _ in ranked if cid in id_to_row]
    # preserve any rows that didn't appear in ranked (shouldn't happen)
    seen = {cid for cid, _ in ranked}
    result += [r for r in rows if str(r.id) not in seen]
    return result


def rerank_bm25(rows: list, query: str) -> list:
    """BM25 over the hard-filtered candidate pool using all profile fields."""
    attrs = [r.model_extra or {} for r in rows]
    ids   = [str(r.id) for r in rows]
    t0 = time.time()
    bm25 = LocalBM25()
    bm25.build(attrs, ids, text_fn=candidate_to_text)
    ranked = bm25.search(query, top_k=len(rows))
    print(f"  BM25 reranked {len(rows)} candidates in {time.time()-t0:.2f}s")
    return _reranked_rows(rows, ranked)


def rerank_dense(rows: list, query: str, dense_retriever: DenseRetriever) -> list:
    """FAISS bi-encoder over the hard-filtered candidate pool."""
    attrs = [r.model_extra or {} for r in rows]
    ids   = [str(r.id) for r in rows]
    t0 = time.time()
    dense_retriever.build_index(attrs, ids, text_fn=candidate_to_text)
    ranked = dense_retriever.search(query, top_k=len(rows))
    print(f"  Dense reranked {len(rows)} candidates in {time.time()-t0:.2f}s")
    return _reranked_rows(rows, ranked)


def rerank_hybrid(rows: list, query: str, dense_retriever: DenseRetriever) -> list:
    """BM25 + FAISS dense, fused with Reciprocal Rank Fusion."""
    attrs = [r.model_extra or {} for r in rows]
    ids   = [str(r.id) for r in rows]
    t0 = time.time()
    bm25 = LocalBM25()
    bm25.build(attrs, ids, text_fn=candidate_to_text)
    dense_retriever.build_index(attrs, ids, text_fn=candidate_to_text)
    hybrid = HybridRetriever(bm25, dense_retriever)
    ranked = hybrid.search(query, top_k=len(rows))
    print(f"  Hybrid reranked {len(rows)} candidates in {time.time()-t0:.2f}s")
    return _reranked_rows(rows, ranked)


def rerank_crossencoder(
    rows: list, query: str, cross_encoder: CrossEncoderReranker
) -> list:
    """Cross-encoder (query, rerankSummary) joint scoring."""
    candidate_index = {str(r.id): r.model_extra or {} for r in rows}
    candidates = [(str(r.id), 0.0) for r in rows]
    t0 = time.time()
    ranked = cross_encoder.rerank(
        query, candidates, candidate_index,
        text_fn=candidate_to_text,
        top_k=len(rows),
        pool_size=50,
    )
    print(f"  CrossEncoder reranked {min(len(rows), 50)} candidates in {time.time()-t0:.2f}s")
    return _reranked_rows(rows, ranked)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(config_path: str, object_ids: list[str], dry_run: bool = False) -> dict:
    if dry_run:
        print(f"  [dry-run] skipping POST — config={config_path}, ids={object_ids}")
        return {"dry_run": True, "config": config_path, "ids": object_ids}

    resp = requests.post(
        EVAL_ENDPOINT,
        headers={
            "Content-Type": "application/json",
            "Authorization": EVAL_EMAIL,
        },
        json={"config_path": config_path, "object_ids": object_ids},
        timeout=30,
    )
    try:
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [eval error] HTTP {resp.status_code}: {resp.text[:400]}")
        raise

# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    config: dict,
    ns,
    voyage: voyageai.Client,
    retrieval_k: int = 200,
    dry_run: bool = False,
    reranker: str = "baseline",
    oai_client: Optional[openai.OpenAI] = None,
    dense_retriever: Optional[DenseRetriever] = None,
    cross_encoder: Optional[CrossEncoderReranker] = None,
) -> dict:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Config : {config['config_path']}")
    print(f"Query  : {config['query'][:80]}...")

    # 1. Embed query
    t0 = time.time()
    embedding = embed_query(voyage, config["query"])
    print(f"  Embedded in {time.time()-t0:.2f}s")

    # 2. ANN retrieval
    t0 = time.time()
    rows = retrieve(ns, embedding, top_k=retrieval_k)
    print(f"  Retrieved {len(rows)} candidates in {time.time()-t0:.2f}s")

    # 3. Hard filter
    filtered = [
        r for r in rows
        if passes_hard_filters(
            r.model_extra or {},
            config.get("hard", {}),
            config_path=config["config_path"],
        )
    ]
    print(f"  After hard filter: {len(filtered)}/{len(rows)} remain")

    if len(filtered) < 10:
        print(f"  ⚠️  Only {len(filtered)} candidates passed hard filters!")
        if len(filtered) == 0:
            print("  ⚠️  Zero candidates — submitting pre-filter ANN results as fallback")
            filtered = rows  # graceful fallback — still gives eval signal

    # 4. Re-rank
    if reranker == "llm" and oai_client is not None:
        reranked = rerank_llm(filtered, config["query"], config, oai_client)
    elif reranker == "bm25":
        reranked = rerank_bm25(filtered, config["query"])
    elif reranker == "dense" and dense_retriever is not None:
        reranked = rerank_dense(filtered, config["query"], dense_retriever)
    elif reranker == "hybrid" and dense_retriever is not None:
        reranked = rerank_hybrid(filtered, config["query"], dense_retriever)
    elif reranker == "crossencoder" and cross_encoder is not None:
        reranked = rerank_crossencoder(filtered, config["query"], cross_encoder)
    else:
        reranked = rerank(filtered, config["query"], config)

    # 5. Extract top-10 IDs
    top10 = reranked[:10]
    top10_ids = [str(r.id) for r in top10]
    print(f"  Top-10 IDs: {top10_ids}")

    # Print candidate names for quick sanity check
    names = [
        (r.model_extra or {}).get("name", "unknown") for r in top10
    ]
    print(f"  Names     : {names}")

    # 6. Submit to eval
    result = evaluate(config["config_path"], top10_ids, dry_run=dry_run)
    if not dry_run:
        score = result.get("average_final_score", "?")
        hard_rates = {
            h["criteria_name"]: h["pass_rate"]
            for h in result.get("average_hard_scores", [])
        }
        soft_scores = {
            s["criteria_name"]: s["average_score"]
            for s in result.get("average_soft_scores", [])
        }
        print(f"  ✅ Final score  : {score}")
        print(f"  Hard pass rates: {hard_rates}")
        print(f"  Soft scores    : {soft_scores}")

    return {
        "config": config["config_path"],
        "num_filtered": len(filtered),
        "ids": top10_ids,
        "names": names,
        "eval": result,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Search pipeline for 10 profession configs")
    parser.add_argument(
        "--config", default=None,
        help=(
            "Run a single config by name. Options: "
            + ", ".join(CONFIGS_BY_NAME.keys())
        ),
    )
    parser.add_argument(
        "--retrieval-k", type=int, default=200,
        help="How many candidates to fetch from TPUF before filtering (default: 200).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print candidate IDs but skip the eval POST request.",
    )
    parser.add_argument(
        "--rerank",
        choices=["baseline", "llm", "bm25", "dense", "hybrid", "crossencoder"],
        default="baseline",
        help=(
            "Re-ranking strategy:\n"
            "  baseline     — ANN order (no re-ranking)\n"
            "  llm          — GPT-4o pointwise scoring\n"
            "  bm25         — BM25 over all candidate profile fields\n"
            "  dense        — FAISS bi-encoder (BAAI/bge-small-en-v1.5)\n"
            "  hybrid       — BM25 + dense fused with Reciprocal Rank Fusion\n"
            "  crossencoder — cross-encoder (query, rerankSummary) joint scoring"
        ),
    )
    args = parser.parse_args()

    print("Connecting to Turbopuffer and Voyage AI...")
    ns     = make_tpuf_namespace()
    voyage = make_voyage_client()

    oai_client     = None
    dense_retriever = None
    cross_encoder   = None

    if args.rerank == "llm":
        print("Initialising OpenAI client for LLM reranking...")
        oai_client = make_openai_client()
    elif args.rerank in ("dense", "hybrid"):
        print("Initialising DenseRetriever (BAAI/bge-small-en-v1.5)...")
        dense_retriever = DenseRetriever()
        dense_retriever._get_model()   # pre-load so first config isn't slower
    elif args.rerank == "crossencoder":
        print("Initialising CrossEncoderReranker (BAAI/bge-reranker-base)...")
        cross_encoder = CrossEncoderReranker()
        cross_encoder._get_model()     # pre-load

    if args.config:
        key = args.config.replace(".yml", "")
        if key not in CONFIGS_BY_NAME:
            raise ValueError(
                f"Unknown config '{args.config}'. Valid: {list(CONFIGS_BY_NAME.keys())}"
            )
        configs_to_run = [CONFIGS_BY_NAME[key]]
    else:
        configs_to_run = QUERY_CONFIGS

    all_results = []
    for cfg in configs_to_run:
        try:
            result = run_pipeline(
                cfg, ns, voyage,
                retrieval_k=args.retrieval_k,
                dry_run=args.dry_run,
                reranker=args.rerank,
                oai_client=oai_client,
                dense_retriever=dense_retriever,
                cross_encoder=cross_encoder,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ Error on {cfg['config_path']}: {e}")
            all_results.append({"config": cfg["config_path"], "error": str(e)})
        time.sleep(0.5)  # be polite to the eval endpoint

    print(f"\n{'='*60}")
    print(f"Done. Ran {len(all_results)} config(s).")

    # Summary table
    print(f"\n{'Config':<35} {'Filtered':>8} {'Score':>8}")
    print("-" * 55)
    for r in all_results:
        if "error" in r:
            print(f"{r['config']:<35} {'ERROR':>8}")
        else:
            score = r["eval"].get("average_final_score", "dry") if not args.dry_run else "dry"
            print(f"{r['config']:<35} {r.get('num_filtered', '?'):>8} {str(score):>8}")

    # Save full results to JSON
    out_path = f"results_{args.rerank}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    main()
