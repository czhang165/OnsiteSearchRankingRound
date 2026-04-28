# Search & Ranking — Results, Analysis & Future Work

## Reranker Comparison

All scores are `average_final_score` (0–100) from the live eval API.
Bold = top performer per profession.

| Profession | Baseline | LLM | BM25 | Dense | Hybrid | Cross-Enc | **Top** |
|------------|----------|-----|------|-------|--------|-----------|---------|
| `tax_lawyer` | 86.3 | 86.7 | **87.7** | 69.3 | 86.3 | 71.0 | BM25 |
| `junior_corporate_lawyer` | 38.7 | 44.7 | 41.5 | **63.8** | 49.2 | 39.7 | Dense |
| `radiology` | 59.7 | **69.3** | 58.0 | 56.0 | 50.3 | 64.7 | LLM |
| `doctors_md` | 4.5 | 13.0 | 0.0 | 4.5 | 4.5 | **25.5** | Cross-Enc |
| `biology_expert` | 45.3 | **69.7** | 64.7 | 56.7 | 62.3 | 54.0 | LLM |
| `anthropology` | 12.7 | 8.3 | **20.3** | 19.0 | 12.7 | 4.3 | BM25 |
| `mathematics_phd` | 77.8 | 72.0 | 70.0 | **79.0** | 69.8 | 78.3 | Dense |
| `quantitative_finance` | 33.3 | **65.0** | 57.3 | 32.7 | 47.7 | 48.3 | LLM |
| `bankers` | 76.7 | 72.0 | 75.7 | **78.0** | 76.7 | 75.3 | Dense |
| `mechanical_engineers` | 90.7 | **93.3** | 93.0 | 89.0 | 83.0 | 91.3 | LLM |
| **Average** | **52.6** | **59.4** | **56.8** | **54.8** | **54.2** | **55.2** | **LLM** |

**Overall ranking: LLM (59.4) > BM25 (56.8) > Cross-Enc (55.2) > Dense (54.8) > Hybrid (54.2) > Baseline (52.6)**

---

## Per-Profession Analysis

**`tax_lawyer` — BM25 wins (87.7)**
Tax law has highly specific, non-ambiguous terminology: "IRS", "corporate tax", "federal tax compliance", "legal opinions", "tax litigation". BM25 excels at exact term matching — a candidate who used these exact words in their bio scores highest. The margin over LLM (86.7) and baseline (86.3) is tiny, confirming the pool is already well-filtered. BM25's edge: it differentiates candidates with dense IRS/tax vocabulary from those with broader legal backgrounds.

**`junior_corporate_lawyer` — Dense wins (63.8, +14 over LLM)**
Classic vocabulary mismatch: European and Canadian corporate lawyers describe their work differently ("cross-border transactions", "transactional counsel", "M&A mandates") vs the query terms. The bi-encoder's shared vector space understands that "international transactional attorney" ≈ "junior corporate lawyer" without exact keyword overlap. BM25 and baseline both need exact term hits, which is why they trail badly.

**`radiology` — LLM wins (69.3)**
The soft criteria require nuanced clinical judgment: `diagnostic_imaging_expertise` and `radiology_expertise` are best assessed by reading the full `rerankSummary` and reasoning about clinical depth (board certification level, imaging modality breadth, AI-assisted diagnosis). BM25 matches "CT" and "MRI" but can't distinguish a research physicist who uses scanners from a clinical radiologist who reads diagnostic reports. Cross-encoder is close (64.7) for the same reason.

**`doctors_md` — Cross-Encoder wins (25.5), all methods score low**
`top_us_md_degree` is nearly impossible to filter locally — we can't reliably detect "top US medical school" from schema fields. Cross-encoder's win comes from joint (query, doc) scoring of the full `rerankSummary`: top US medical school graduates tend to mention prestigious affiliated hospitals (Johns Hopkins, Mayo, UCSF) in their career narratives, and the cross-encoder captures this contextual signal even when the school name isn't literally stated. The fundamental bottleneck is retrieval, not ranking — the candidate pool doesn't contain enough top-US-MD graduates.

**`biology_expert` — LLM wins (69.7)**
The soft criteria are research-specific: peer-reviewed publications, CRISPR/PCR/sequencing lab techniques, mentoring. GPT-4o reads the `rerankSummary` holistically and assesses whether a candidate has substantial research output vs just a biology degree. BM25 is competitive (64.7) because top-US biology researchers use distinctive vocabulary ("Nature", "Cell", "CRISPR"), but it can't evaluate publication depth — LLM can.

**`anthropology` — BM25 wins (20.3), all methods score low**
The `recent_phd_program` criterion (PhD started 2022+) is extremely restrictive and eliminates most candidates. BM25 edges ahead because recent PhD students write dissertation-focused summaries with very specific domain vocabulary ("ethnographic fieldwork", "labor migration", "participant observation") that exactly matches the expanded query. Their summaries are more keyword-dense than senior researchers who write more broadly.

**`mathematics_phd` — Dense wins (79.0)**
Mathematical subdiscipline vocabulary has rich semantic structure but inconsistent surface forms. A candidate researching "stochastic differential equations" is a good match for "applied mathematics" even though neither term appears in the other. The bi-encoder understands this equivalence space. BM25 and baseline also do well (77.8/70.0) because math PhDs use precise technical terms, but dense's ability to map across mathematical subdisciplines gives it the final edge. Cross-encoder is very close (78.3).

**`quantitative_finance` — LLM wins (65.0, +8 over BM25, +32 over baseline)**
Widest single-method margin in the dataset. The M7 MBA + quant finance path requires understanding two credentials simultaneously: business school prestige tier AND finance career relevance. GPT-4o reads "MBA from Columbia, VP at Goldman risk management, derivatives portfolio" and understands this as a strong M7 quant finance profile. Dense and baseline fail badly (32–33) because MBA + quant finance is semantically distant in embedding space — MBAs don't write "quantitative finance" the way math PhDs write "applied mathematics".

**`bankers` — Dense wins (78.0)**
Healthcare investment banking occupies a specific semantic subspace: "provider network M&A", "digital health equity", "payer-provider integration", "RCM optimization". These healthcare-specific finance terms cluster together in embedding space. A candidate with "healthcare private equity" in their bio is semantically close to "healthcare M&A" even without exact overlap. Dense finds them; BM25 misses unless the exact terms appear.

**`mechanical_engineers` — LLM wins (93.3), all methods score well**
All methods perform well because mechanical engineering credentials are explicit and well-structured in profiles (CAD tools named, FEA/CFD mentioned). LLM's small edge (+0.3 over BM25) comes from better assessment of "end-to-end product lifecycle involvement" and "domain specialization" — reading whether a career arc shows product ownership vs component-level work requires narrative understanding.

---

## Cross-Cutting Insights

| Pattern | Explanation |
|---------|-------------|
| **LLM best overall (59.4)** | Soft criteria in every config require reading comprehension — publication depth, clinical nuance, deal specificity. GPT-4o is the only method that can reason about these from the full `rerankSummary`. |
| **BM25 surprisingly strong (56.8)** | Domain-specific professions (tax law, anthropology) have distinctive, low-ambiguity vocabulary. When the right candidates use the right words, exact-match scoring is hard to beat. |
| **Dense wins vocabulary-mismatch cases** | `junior_corporate_lawyer`, `math_phd`, `bankers` — professions where equivalent concepts have multiple surface forms across regions or career stages. |
| **CrossEncoder best for credential inference** | `doctors_md` — credential signal is implicit in the career narrative (hospital affiliations, residency programs) rather than stated directly. Joint (query, doc) scoring extracts it better than retrieval methods. |
| **Hybrid consistently underperforms** | RRF fusion averages strengths away rather than combining them. When BM25 is strong, diluting with dense hurts (`tax_lawyer`: hybrid 86.3 vs BM25 87.7). When dense is strong, BM25 noise brings it down (`junior_lawyer`: hybrid 49.2 vs dense 63.8). Hybrid only helps when both signals are independently strong — rare in this dataset. |
| **`doctors_md` and `anthropology` bottleneck is retrieval, not ranking** | `top_us_md_degree` and `recent_phd_program` can't be locally approximated from schema fields. No re-ranker can fix a pool that doesn't contain the right candidates. |

---

## Hard Filter Learnings

| Finding | Impact |
|---------|--------|
| **MD degrees stored as `"md"` not `"doctorate"`** | Fixed `doctors_md` filter from 0 to 168 candidates. Critical schema correction. |
| **MBBS (Indian/Pakistani MD) stored as `"bachelor's"`** | Changed radiology to use `deg_fos` medicine keywords instead of `deg_degrees`. Raised score from 15 → 59.7. |
| **US/UK/Canada undergrad not filterable from normalized fields** | Added `deg_schools` keyword matching for `biology_expert` and `mathematics_phd`. |
| **`deg_degrees` and `deg_schools` are index-aligned** | Enabled per-degree school checking (e.g. is the doctorate from a US med school?). |
| **`require_us_mba` school matching too strict for quant_finance** | Removed it — school names in TPUF don't reliably match short keywords. Rely on ANN query bias instead. |
| **Hard filter pass rate is the primary score driver** | A single wrong hard filter zeros out entire candidates. Fixing filters gave bigger score gains than any re-ranker improvement. |

---

## Future Work

- **Dynamic soft-criteria weighting**: learn weights for each criterion's contribution to `final_score` across configs, then use weighted scoring in the LLM prompt.
- **Hybrid re-ranking with learned alpha**: instead of fixed RRF, learn per-profession blending weights (BM25 α + dense β) using the eval API scores as the signal.
- **Better M7 MBA detection for quant_finance**: TPUF attribute filtering (`filters` param in the query) could pre-select MBA holders before ANN ranking, improving the candidate pool quality.
- **Doctors_md school detection**: build a larger US medical school name list with full aliases, or use the `rerankSummary` text to search for hospital/program mentions as a proxy for school prestige.
- **Anthropology recency**: `deg_start_years` filter at 2022+ is correct but harsh — explore relaxing to 2020+ and accepting "ABD" (all-but-dissertation) status described in `rerankSummary`.
- **QueryExpander for ANN retrieval**: currently the `build_profession_query()` method in `enhancements.py` is only used locally. Using the richer expanded query for the Turbopuffer ANN call itself could improve top-200 recall.
