# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** EntityResolvers  
**Team Members:** Anmol and Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We developed an industrial-scale, two-stage machine learning system for cross-source Business Entity Resolution across multilingual records (US, India, and France). The pipeline couples a high-recall, multi-key inverted index blocking engine with a precision-calibrated LightGBM pairwise classifier operating over 20 C-accelerated string similarity, token overlap, and geographic/numeric features. Our solution achieves an **F_0.5 score of 0.9515** with an average precision of **96.65%**, strictly adhering to the competition's 2× precision weighting penalty and guaranteeing zero memory overflow through country-partitioned streaming.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across ~24.2 million records in 7 files, we identified four fundamental challenges:
1. **Extreme Combinatorial Complexity:** Matching 1.73M Source 1 reference entities against ~10M Source 2/3 candidates yields $\sim 1.7 \times 10^{13}$ pairwise comparisons, making exhaustive matching computationally impossible.
2. **Asymmetric Precision Weighting (F_0.5 Metric):** Under macro $F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$, a false merge (false positive) penalizes the score twice as heavily as a missed link. Moreover, **5.58%** of S1 records are singletons (zero matches); incorrectly predicting any match for a singleton earns a strict 0.0.
3. **Out-of-Distribution Language Shift (France):** While training data strictly covers `US` and `India`, the test set introduces `France` (~1.7M records) containing French accents (`é`, `è`, `ç`, `ô`), French legal suffixes (`SARL`, `SAS`, `EURL`), and French thoroughfare naming conventions (`Rue`, `Boulevard`, `Impasse`).
4. **Heavy Syntactic and Address Noise:** Inconsistencies include missing postal codes (3.3% null addresses in S2/3), landmark references ("Near SBI ATM"), abbreviated street suffixes ("St" vs "Street", "Rd" vs "Road"), and legal abbreviations ("Pvt Ltd" vs "Private Limited").

### 2.2 Solution Strategy
We architected a strictly decoupled two-stage framework:
- **Stage 1 (Candidate Generation / Blocking):** An embarrassingly parallel, country-partitioned inverted indexing mechanism that prunes >99.999% of non-matching pairs while maintaining $\ge 94.0\%$ recall on true matches.
- **Stage 2 (Pairwise Scoring & Decision Boundary):** A gradient boosted decision tree (LightGBM) trained on hard negatives derived directly from blocking output. The decision threshold was calibrated on held-out validation data to explicitly maximize macro-averaged $F_{0.5}$.

**Approach Type:** Country-Partitioned Multi-Key Inverted Index Blocking + Calibrated LightGBM Pairwise Classifier  
**Core Innovation:** 
- Frequency-capped multi-signal inverted indexing with inverse-document-frequency (IDF) weighting that prevents token explosion while capturing rare, highly discriminative tokens.
- Language-agnostic feature engineering combining Unicode NFKD normalization, character n-gram jaccard similarities, and numeric/postal containment, allowing zero-shot generalization from US/India to France.

---

## 3. Candidate Generation (Blocking)

### Blocking Strategy & Keys Used
To reduce the $17.3 \text{ trillion}$ comparison space to under 50 million pairs, we enforced:
1. **Strict Country Partitioning:** Matches never cross national boundaries ($P(\text{cross-country match}) = 0.0$). Records are partitioned by country before indexing.
2. **Frequency-Capped Name Inverted Index:** Document frequency ratio capped at $df \le 0.005$ to suppress ultra-frequent noise words ("Services", "Consulting", "Store"). Rare tokens receive a 3.0× score multiplier.
3. **First Significant Word Index:** Exact matching on the first non-stopword token (weight 4.0×), capturing leading brand/trade names despite differing legal forms.
4. **Address Number Index:** Inverted index on digit sequences of length $\ge 3$ (PIN codes, plot/door numbers). PIN codes ($\ge 5$ digits) receive a 3.0× precision boost.
5. **Character Shingle Fallback (k=4):** Character 4-gram shingles capture typographical variations and transliterations.

### Candidate Set Size & Recall Preservation
- Total candidate pairs generated on full test set (1.73M entities): **47,943,544 pairs** (average 27.7 candidates per entity).
- Total predicted match links: **6,450,140 links** (average 4.34 links per matched entity).
- Predicted singletons: **247,831 entities** (14.30%).
- Entities with predicted matches: **1,484,713 entities** (85.70%).
- Blocking recall on ground truth evaluation: **94.00%** (107,532 / 114,396 true links captured).
- Candidate reduction ratio: **99.9997%** reduction in total comparison pairs.

---

## 4. Matching Model

## 4. Matching Model

### Features Used (25 Dimensions)
All features are extracted using C-accelerated primitives (`rapidfuzz`, vectorised numpy) without heavy model inference latency:
1. **Name Similarity (10 features):**
   - Jaro-Winkler similarity (heavily weights prefix matches)
   - Token Sort Ratio & Token Set Ratio (invariant to word order and extra tokens)
   - Partial Ratio (substring matching)
   - Token Jaccard & Character 3-gram Jaccard
   - Normalized Levenshtein edit distance
   - Absolute string length difference ratio
   - Exact first-token indicator
   - Exact normalized name match indicator
2. **Address Similarity (10 features):**
   - Jaro-Winkler & Normalized Levenshtein on normalized addresses
   - Token Sort Ratio & Token Set Ratio
   - Token Jaccard & Character 3-gram Jaccard
   - Numeric sequence overlap ratio & S1 number containment ratio
   - Address string length difference ratio
   - Exact normalized address match indicator
3. **Semantic Conflict Resolution (2 features):**
   - `postal_code_status`: Explicit PIN/ZIP semantic validation. Yields `+1.0` if codes match, and a severe `-1.0` penalty if recognized 5/6 digit postal codes conflict (preventing cross-city false merges on franchise chains like Domino's, Subway, or Apollo Pharmacy).
   - `door_number_status`: Door/plot/suite number conflict detection (`+1.0` on match, `-1.0` on conflict).
4. **Cross & Meta Features (2 features):**
   - Name-in-address containment indicator
   - Country match indicator
5. **Script & Normalization Fallback (1 feature):**
   - `name_dropped_by_normalization`: Binary indicator (`1.0` if either entity's clean name collapsed to empty despite having non-empty raw text). Empirically signals the gradient booster to shift decision weight to address, PIN, and door-number features rather than penalizing for zero name similarity.

### Model Architecture & Training
- **Model Type:** LightGBM Binary Classifier (MIT License, $\approx 500$ estimators, learning rate 0.05, num_leaves 63, colsample_bytree 0.8, subsample 0.8).
- **Training Scale:** 5,480,548 pairs generated by the blocking stage (178,935 positive match links, 5,301,613 hard negative distractors).
- **Threshold Selection & Adaptive Rank Pruning:** 
  - Decision threshold $\tau^* = 0.25$ calibrated directly on held-out validation data for macro $F_{0.5}$.
  - Adaptive rank pruning: Caps matches per entity to a maximum of **8** (matching the 99.9th percentile of ground truth).
  - Margin filtering: Requires candidate probability to be within $0.15$ of the top-ranked candidate, strictly suppressing low-confidence false positives.

---

## 5. Results & Error Analysis

### Cross-Script Devanagari & Indic Script Discovery
During error analysis on Indian records, we discovered a silent script-normalization gap:
- **The Issue:** The raw Source 1 dataset is 100% Latin-transliterated, whereas **13.35%** of Source 2 Indian business names and **7.60%** of Source 3 names are written in native Indic scripts (primarily Devanagari, but also Gujarati, Bengali, Tamil, Kannada, and Telugu).
- Standard ASCII-stripping regexes (`[^a-z0-9\s\-]`) erased all non-Latin characters, collapsing 21.90% of Indian Source 2 names to completely empty strings (`""`) and zeroing out all 10 name similarity features.
- **The Fix:** We implemented a local, rule-based transliteration stage (`indic-transliteration` ITRANS, MIT licensed) supporting Devanagari, Gujarati, Bengali, Tamil, Kannada, Telugu, Malayalam, Gurmukhi, and Oriya, alongside modern loanword vowel mapping (`\u0949`, `\u0911`, `\u093c`).
- **Fair Play Compliance:** This is a deterministic, rule-based script converter (0 parameters, executed 100% locally on CPU with zero network requests and zero business identity lookups), adhering strictly to the competition integrity rules.
- **Diagnostic Result:** Dropped Indian name rate collapsed from **21.90% $\rightarrow$ 0.00%**.
- **Blocking Gain:** Blocking recall rose from **93.40% $\rightarrow$ 93.92% (+0.52% lift)**, capturing **+974 true match links** that were previously impossible to retrieve.

### Validation Performance & Country Breakdown
- **Macro F_0.5 Score (Overall):** **0.9546** (improved from 0.9515 baseline)
- **Average Precision:** **98.75%** (exceptional precision, strictly penalizing false merges)
- **Average Recall:** **89.09%**
- **Logloss:** **0.00403**

#### Per-Country Performance Breakdown
| Country | Macro $F_{0.5}$ | Precision | Recall | Validation Entities |
|---|---|---|---|---|
| **India** | **0.9356** | 97.93% | 85.80% | 2,385 |
| **US** | **0.9672** | 99.29% | 91.24% | 3,615 |
| **Overall** | **0.9546** | **98.75%** | **89.09%** | **6,000** |

### Error Analysis & Mitigation
- **False Positives (Wrong Merges):** Primarily caused by franchise businesses sharing an identical name but located at different branches within the same city where address fields are partially missing. Addressed by `postal_code_status` and `door_number_status`, which strictly penalize matches when postal codes or building numbers conflict.
- **False Negatives (Missed Matches):** Primarily caused by extreme address omission in Source 2/3 (e.g., address containing only a state name while Source 1 has a detailed street address). The token set ratio and partial ratio features prevent over-penalization when one source contains a subset of the other's address.

---

## 6. Conclusion

Our solution demonstrates that combining domain-specific, high-recall blocking with C-accelerated string similarity features and gradient boosted trees outperforms monolithic neural approaches in both accuracy and operational feasibility. By adhering to memory-safe streaming, deterministic formatting, and strict metric alignment with macro $F_{0.5}$, the system delivers competitive entity resolution at commercial scale within minutes.

---

## Appendix

### A. Code Artefacts
All code resides under `code/business_entity_resolution/`:
- `src/preprocessing.py`: Vectorised Unicode NFKD normalization, legal suffix stripping, and address expansion.
- `src/blocking.py`: Country-partitioned multi-key inverted index candidate generation.
- `src/features.py`: RapidFuzz C-accelerated 20-dimensional feature extractor.
- `src/model.py`: LightGBM training, threshold tuning, and batch prediction.
- `src/train.py`: End-to-end training entry point.
- `src/predict.py`: Memory-safe streaming inference producing `matching_results.tsv` and `candidate_pairs.tsv`.
- `requirements.txt`: Pinned open-source dependencies (all MIT / Apache 2.0).

### B. Verification & Compliance
Outputs are validated using the official contest validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Result: `PASS` — zero formatting errors, exact schema match, zero cross-source violations.
