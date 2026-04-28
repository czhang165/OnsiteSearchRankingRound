"""
enhancements.py  (SearchRankingRound)
--------------------------------------
Self-contained search enhancement utilities for the candidate ranking pipeline.
All classes operate on candidate Row.model_extra dicts from Turbopuffer.

The "doc" for every candidate is built by candidate_to_text(), which combines:
    rerankSummary + exp_titles + deg_fos + deg_schools + exp_companies + deg_degrees

Typical usage in baseline_search.py:
-------------------------------------

    from enhancements import (
        QueryExpander, DenseRetriever, HybridRetriever,
        CrossEncoderReranker, candidate_to_text,
    )

    # Stage 2 — richer query
    expander = QueryExpander()
    rich_query = expander.build_profession_query(
        description="Seasoned attorney with JD...",
        soft_criteria=["corporate_transaction_experience", "irs_audit_experience"],
    )

    # Stage 3a — dense retrieval over hard-filtered candidates
    attrs_list = [r.model_extra for r in filtered_rows]
    ids        = [str(r.id)     for r in filtered_rows]
    dr = DenseRetriever()
    dr.build_index(attrs_list, ids)
    dense_results = dr.search(rich_query, top_k=50)

    # Stage 3b — hybrid (BM25 + Dense) over hard-filtered candidates
    from rank_bm25 import BM25Okapi
    # ... build local BM25 on the filtered pool ...

    # Stage 4a — cross-encoder reranking
    candidate_index = {str(r.id): r.model_extra for r in filtered_rows}
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(rich_query, initial_ranked_list, candidate_index)
"""

from __future__ import annotations

import re
import string
from typing import Callable, List, Dict, Tuple


# ── type alias ─────────────────────────────────────────────────────────────────
RankedList = List[Tuple[str, float]]


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE TEXT CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def candidate_to_text(attrs: dict) -> str:
    """
    Convert a candidate's Row.model_extra dict to a single searchable string.

    Combines rerankSummary (rich prose bio) with structured fields to give
    BM25 and dense models full signal coverage across all profile dimensions.
    """
    parts = []
    if attrs.get("rerankSummary"):
        parts.append(attrs["rerankSummary"])
    if attrs.get("exp_titles"):
        parts.append(" ".join(attrs["exp_titles"]))
    if attrs.get("deg_fos"):
        parts.append(" ".join(attrs["deg_fos"]))
    if attrs.get("deg_schools"):
        parts.append(" ".join(attrs["deg_schools"]))
    if attrs.get("exp_companies"):
        parts.append(" ".join(attrs["exp_companies"][:5]))
    if attrs.get("deg_degrees"):
        parts.append(" ".join(attrs["deg_degrees"]))
    return " ".join(filter(None, parts))


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — QUERY EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

class QueryExpander:
    """
    Rule-based query expansion for candidate search.

    Two main entry points:

    1. expand(query)
       Appends synonyms for any recognized tokens in the query.
       e.g. "gp radiology" → "gp radiology general practitioner family medicine
                              primary care radiologist radiology diagnostic imaging"

    2. build_profession_query(description, soft_criteria)
       Builds a rich query from the full natural language description and
       soft criteria names — much richer signal than the short ANN query.
       Use this as the query for BM25 and DenseRetriever.
    """

    SYNONYM_MAP: Dict[str, List[str]] = {
        # ── legal ──────────────────────────────────────────────────────────
        "lawyer":       ["attorney", "counsel", "solicitor", "esquire"],
        "attorney":     ["lawyer", "counsel", "legal professional"],
        "jd":           ["juris doctor", "law degree", "law school"],
        "irs":          ["internal revenue service", "tax authority", "federal tax"],
        "m&a":          ["mergers and acquisitions", "merger", "acquisition", "deal"],
        "corporate law":["business law", "transactional law", "commercial law"],
        "litigation":   ["legal dispute", "trial", "court", "lawsuit"],
        "compliance":   ["regulatory", "legal compliance", "regulatory compliance"],

        # ── medicine ───────────────────────────────────────────────────────
        "md":           ["doctor of medicine", "physician", "medical doctor"],
        "mbbs":         ["bachelor of medicine", "medical degree", "physician"],
        "gp":           ["general practitioner", "family medicine", "primary care"],
        "radiologist":  ["radiology", "diagnostic imaging", "medical imaging"],
        "radiology":    ["radiologist", "diagnostic imaging", "CT", "MRI", "X-ray"],
        "ehr":          ["electronic health record", "emr", "epic", "cerner"],
        "residency":    ["medical training", "clinical training", "intern"],

        # ── finance / banking ──────────────────────────────────────────────
        "ib":           ["investment banking", "investment bank", "bulge bracket"],
        "mba":          ["master of business administration", "business school"],
        "pe":           ["private equity", "buyout", "portfolio company"],
        "quant":        ["quantitative analyst", "quantitative finance", "algorithmic trading"],
        "derivatives":  ["options", "futures", "swaps", "fixed income"],
        "lbo":          ["leveraged buyout", "private equity", "acquisition finance"],
        "dcf":          ["discounted cash flow", "valuation", "financial modeling"],
        "banker":       ["investment banker", "corporate finance", "advisory"],

        # ── academia / research ────────────────────────────────────────────
        "phd":          ["doctor of philosophy", "doctorate", "doctoral degree"],
        "postdoc":      ["postdoctoral", "research fellow", "post-doctoral"],
        "crispr":       ["gene editing", "genome editing", "molecular biology"],
        "ethnography":  ["fieldwork", "qualitative research", "cultural study"],
        "stochastic":   ["probability", "stochastic calculus", "brownian motion"],

        # ── engineering ────────────────────────────────────────────────────
        "cad":          ["SolidWorks", "AutoCAD", "design software", "3D modeling"],
        "fea":          ["finite element analysis", "ANSYS", "structural simulation"],
        "cfd":          ["computational fluid dynamics", "COMSOL", "fluid simulation"],
        "bsme":         ["mechanical engineering", "mechanical engineer", "BSME degree"],
    }

    # Maps soft-criteria names → human-readable search terms
    SOFT_CRITERIA_MAP: Dict[str, str] = {
        "corporate_transaction_experience":
            "corporate transactions mergers acquisitions deal structuring",
        "irs_audit_experience":
            "IRS audit tax dispute federal tax compliance",
        "legal_writing_expertise":
            "legal writing opinions briefs contracts drafting",
        "ma_contracts_exposure":
            "M&A contracts due diligence legal documentation",
        "international_law_familiarity":
            "international law cross-border regulatory compliance",
        "healthcare_investment_banking_experience":
            "healthcare investment banking biotech pharma provider networks",
        "healthcare_ma_transactions":
            "healthcare mergers acquisitions hospital deal recapitalization",
        "healthcare_metrics_knowledge":
            "healthcare metrics revenue cycle RCM payer provider integration",
        "diagnostic_imaging_expertise":
            "diagnostic imaging CT MRI X-ray ultrasound interpretation reporting",
        "board_certification":
            "board certified ABR FRCR radiology certification credential",
        "molecular_biology_research":
            "molecular biology genetics cell biology CRISPR PCR sequencing publications",
        "experimental_design_skills":
            "experimental design data analysis lab techniques research methods",
        "teaching_mentoring_experience":
            "teaching mentoring undergraduate courses student supervision",
        "ethnographic_methods_expertise":
            "ethnography fieldwork qualitative research participant observation",
        "academic_output_quality":
            "peer reviewed publications journal papers conference presentations",
        "applied_anthropology_experience":
            "applied anthropology real-world interdisciplinary migration labor",
        "mathematical_modeling_proficiency":
            "mathematical modeling numerical analysis proof theorem computation",
        "research_expertise_math":
            "mathematics research publications postdoc academia pure applied",
        "financial_modeling_experience":
            "financial modeling DCF valuation portfolio optimization derivatives pricing",
        "python_quantitative_proficiency":
            "Python QuantLib pandas numpy scipy quantitative programming",
        "high_stakes_environment":
            "global investment firm trading desk hedge fund institutional finance",
        "cad_simulation_tools_experience":
            "CAD SolidWorks AutoCAD ANSYS COMSOL FEA CFD simulation tools",
        "product_lifecycle_involvement":
            "product lifecycle prototyping manufacturing testing end-to-end design",
        "domain_specialization_mechanical":
            "thermal systems fluid dynamics structural analysis mechatronics",
        "ehr_systems_familiarity":
            "EHR EMR Epic Cerner electronic health record patient management",
        "telemedicine_comfort":
            "telemedicine telehealth virtual care remote patient consultation",
    }

    def expand(self, query: str) -> str:
        """Return query with synonyms appended for any recognised tokens."""
        tokens = query.lower().split()
        extra: List[str] = []
        for tok in tokens:
            if tok in self.SYNONYM_MAP:
                extra.extend(self.SYNONYM_MAP[tok])
        return query + " " + " ".join(extra) if extra else query

    def build_profession_query(
        self,
        description: str,
        soft_criteria: List[str],
    ) -> str:
        """
        Build a rich query from a profession's natural language description
        and its soft criteria names.

        This should be used as the query for BM25 and DenseRetriever instead
        of the short ANN query string, since it contains much richer signal.

        Example:
            expander.build_profession_query(
                description="Seasoned attorney with JD from top US law school...",
                soft_criteria=["corporate_transaction_experience", "irs_audit_experience"],
            )
        """
        parts = [description]
        for criterion in soft_criteria:
            expansion = self.SOFT_CRITERIA_MAP.get(
                criterion, criterion.replace("_", " ")
            )
            parts.append(expansion)
        combined = " ".join(parts)
        return self.expand(combined)

    def expand_all(self, queries: List[dict]) -> List[dict]:
        """Return new query dicts with expanded text (JobListSearch interface)."""
        return [
            {**q, "text": self.expand(q["text"]), "original_text": q["text"]}
            for q in queries
        ]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — DENSE + HYBRID RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

# BM25 tokenizer (mirrors bm25_retriever.py so token sets are compatible)
STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "in", "on", "at", "to", "of",
    "with", "is", "are", "was", "be", "has", "have", "had", "this", "that",
}

def _tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    text = text.lower()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class LocalBM25:
    """
    Lightweight BM25 over a small in-memory pool of candidate documents.

    Intended for re-ranking the hard-filtered candidate pool (typically
    20–200 docs) rather than the full 200k corpus.  Use this inside
    HybridRetriever to fuse BM25 and dense signals on the local pool.

    Usage:
        attrs_list = [r.model_extra for r in filtered_rows]
        ids        = [str(r.id)     for r in filtered_rows]

        bm25 = LocalBM25()
        bm25.build(attrs_list, ids, text_fn=candidate_to_text)
        results = bm25.search(query, top_k=50)
    """

    def __init__(self):
        self._bm25 = None
        self._ids: List[str] = []

    def build(
        self,
        attrs_list: List[dict],
        ids: List[str],
        text_fn: Callable[[dict], str] = candidate_to_text,
    ) -> None:
        from rank_bm25 import BM25Okapi
        corpus = [_tokenize(text_fn(a)) for a in attrs_list]
        self._bm25 = BM25Okapi(corpus)
        self._ids = list(ids)

    def search(self, query: str, top_k: int = 50) -> RankedList:
        if self._bm25 is None:
            raise RuntimeError("Call build() before search().")
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._ids, scores), key=lambda x: -x[1])
        return [(id_, float(s)) for id_, s in ranked[:top_k]]


class DenseRetriever:
    """
    Bi-encoder dense retrieval using sentence-transformers + FAISS.

    Encodes candidate docs into a shared vector space, retrieves by
    cosine similarity. Designed to re-rank the hard-filtered pool
    (not the full 200k corpus — that's already handled by Turbopuffer ANN).

    Usage:
        attrs_list = [r.model_extra for r in filtered_rows]
        ids        = [str(r.id)     for r in filtered_rows]

        dr = DenseRetriever()
        dr.build_index(attrs_list, ids)              # encodes & builds FAISS
        results = dr.search(query, top_k=50)         # returns [(id, score), ...]

    Model options (all downloaded on first use):
        "BAAI/bge-small-en-v1.5"          — 33M params, 384-dim, fast (~10ms/50 docs)
        "BAAI/bge-base-en-v1.5"           — 109M params, 768-dim, better quality
        "sentence-transformers/all-MiniLM-L6-v2" — 22M params, 384-dim, tiny/fast
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self._index = None      # faiss.IndexFlatIP
        self._ids: List[str] = []

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[Dense] Loading {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_index(
        self,
        attrs_list: List[dict],
        ids: List[str],
        text_fn: Callable[[dict], str] = candidate_to_text,
    ) -> None:
        """
        Encode candidate docs and build a FAISS inner-product index.
        Cosine similarity = inner product on L2-normalised vectors.
        """
        import faiss

        model = self._get_model()
        texts = [text_fn(a) for a in attrs_list]
        print(f"[Dense] Encoding {len(texts)} candidates ...")
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        ).astype("float32")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._ids = list(ids)
        print(f"[Dense] Index ready: {len(self._ids)} docs, dim={dim}.")

    def search(self, query: str, top_k: int = 50) -> RankedList:
        """Encode query and retrieve top-k by cosine similarity."""
        if self._index is None:
            raise RuntimeError("Call build_index() before search().")
        model = self._get_model()
        q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(q_emb, top_k)
        return [
            (self._ids[i], float(scores[0][j]))
            for j, i in enumerate(indices[0])
            if i >= 0
        ]


class HybridRetriever:
    """
    Hybrid BM25 + Dense retrieval with Reciprocal Rank Fusion (RRF).

    Fuses the ranked lists from a local BM25 and a DenseRetriever using:
        score(d) = Σ_i  1 / (k + rank_i(d))    [Cormack et al. 2009]

    Operates on the hard-filtered candidate pool, not the full corpus.

    Usage:
        attrs_list = [r.model_extra for r in filtered_rows]
        ids        = [str(r.id)     for r in filtered_rows]

        bm25 = LocalBM25()
        bm25.build(attrs_list, ids)

        dr = DenseRetriever()
        dr.build_index(attrs_list, ids)

        hybrid = HybridRetriever(bm25, dr)
        results = hybrid.search(query, top_k=10)
    """

    def __init__(self, bm25: LocalBM25, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25  = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    @staticmethod
    def reciprocal_rank_fusion(
        lists: List[RankedList],
        k: int = 60,
    ) -> RankedList:
        """Fuse multiple ranked lists using RRF."""
        scores: Dict[str, float] = {}
        for ranked_list in lists:
            for rank, (doc_id, _) in enumerate(ranked_list, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: -x[1])

    def search(self, query: str, top_k: int = 50) -> RankedList:
        """Retrieve, fuse with RRF, return top-k."""
        bm25_results  = self.bm25.search(query, top_k=top_k)
        dense_results = self.dense.search(query, top_k=top_k)
        fused = self.reciprocal_rank_fusion(
            [bm25_results, dense_results], k=self.rrf_k
        )
        return [(doc_id, score) for doc_id, score in fused[:top_k]]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — RE-RANKING
# ══════════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """
    Cross-encoder re-ranking for candidate search.

    Jointly encodes (query, candidate_doc) pairs for fine-grained relevance
    scoring. Much more accurate than bi-encoders for nuanced matching, but
    O(n) at inference — run only on the hard-filtered pool, not the full corpus.

    Usage:
        candidate_index = {str(r.id): r.model_extra for r in filtered_rows}
        initial = [(str(r.id), 0.0) for r in filtered_rows]

        reranker = CrossEncoderReranker()
        top10 = reranker.rerank(query, initial, candidate_index, top_k=10)

    Model options:
        "BAAI/bge-reranker-base"                  — general purpose, strong on bio text
        "cross-encoder/ms-marco-MiniLM-L-6-v2"    — fastest, web-search trained
        "mixedbread-ai/mxbai-rerank-base-v1"       — recent, strong general reranker
        "cross-encoder/ms-marco-electra-base"      — highest accuracy, slower

    When to use cross-encoder vs GPT-4o (rerank_llm in baseline_search.py):
        Cross-encoder wins: semantic similarity is the main signal
                            (radiology, mechanical_engineers, biology_expert)
        GPT-4o wins:        nuanced credential matching needed
                            (tax_lawyer IRS depth, bankers healthcare M&A,
                             quantitative_finance M7 MBA)
        Hybrid:             cross-encoder for fast pre-filtering to top-20,
                            GPT-4o for final top-10 selection
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"[CrossEncoder] Loading {self.model_name} ...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: RankedList,
        candidate_index: Dict[str, dict],
        text_fn: Callable[[dict], str] = candidate_to_text,
        top_k: int = 10,
        pool_size: int = 50,
    ) -> RankedList:
        """
        Re-score top-pool_size candidates with the cross-encoder.

        Args:
            query:           profession query string (or richer build_profession_query output)
            candidates:      initial ranked list [(candidate_id, score), ...]
            candidate_index: {candidate_id: row.model_extra} for text extraction
            text_fn:         converts model_extra to doc string (default: candidate_to_text)
            top_k:           number of results to return
            pool_size:       candidates to re-score — trade latency vs quality
                             (50 candidates ≈ 100–300ms with bge-reranker-base)
        """
        model = self._get_model()

        pool = candidates[:pool_size]
        pairs: List[Tuple[str, str]] = []
        valid_ids: List[str] = []

        for cand_id, _ in pool:
            attrs = candidate_index.get(cand_id)
            if attrs is None:
                continue
            doc = text_fn(attrs)[:2048]     # truncate to keep latency bounded
            pairs.append((query, doc))
            valid_ids.append(cand_id)

        if not pairs:
            return candidates[:top_k]

        scores = self._model.predict(pairs, show_progress_bar=False)

        reranked = sorted(
            zip(valid_ids, scores.tolist()),
            key=lambda x: -x[1],
        )

        # Append overflow candidates (beyond pool_size) at the end
        reranked_set = {cid for cid, _ in reranked}
        tail = [
            (cid, s) for cid, s in candidates[pool_size:]
            if cid not in reranked_set
        ]

        return [(cid, float(s)) for cid, s in reranked[:top_k]] + tail
