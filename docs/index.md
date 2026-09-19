# Predicting Content Decline: A Refresh Prioritization Model for Search-Driven Content

## Abstract

Content teams with limited review time need a way to prioritize which pages to refresh first, out of a large pool of published content. This paper asks: which pages should a content/SEO team prioritize for refresh review, given a decline signal, to support that decision? Using ~9.8 million rows of production search performance data from the FlyRank internship warehouse, we trained a Random Forest classifier on December 2025 features (impressions, clicks, search position, age) to predict which pages would show declining visibility by March 2026. The model achieved a Precision@50 of 1.000 and AUC of 0.933–0.935, compared to a rule-based baseline's Precision@50 of 0.000 and AUC of 0.627. These results support using the model as a decision-support tool for refresh prioritization, though the label is a proxy for decline rather than a causal signal, and the model has not been validated on any window beyond this single historical panel.

## 1. Introduction / Problem Statement

Content teams routinely face more potentially-stale pages than they have time to review. Deciding which pages to prioritize for a refresh — updating, expanding, or re-optimizing — is typically done by age alone or by manual spot-checking, neither of which reliably surfaces the pages carrying the most search-visibility risk.

This project frames that decision as a ranking problem: given a large pool of published content, produce a ranked queue of pages most likely to be declining in search visibility, along with a reason code a human reviewer can trust and act on.

## 2. Data

- **Release:** FlyRank internship-warehouse (HuggingFace datasets)
- **Tables used:** `dim_content.parquet`, `dim_clients.parquet`, `fact_content_daily_performance/month=YYYY-MM` (monthly partitions)
- **Scale:** 9,841,378 total rows; 331,437 unique content items
- **Date windows:** features computed from December 2025; label/outcome computed from March 2026 (a strictly time-aware split)
- **Excluded:** `sessions_ai` and all `ai_*` columns (sparse, mostly null across clients); `scroll_events` (depends on `ga4_data_available`, inconsistent across the client base)
- **Public-safety:** all identifiers (`content_hash_id`, `client_hash_id`) are pre-anonymized in the source dataset; no client names, URLs, or private queries appear anywhere in this analysis

## 3. Methodology

**Label definition:** `target_declining = 1` if March 2026 impressions were lower than December 2025 impressions for a given page, else `0`. This is a proxy for "needs a refresh," not a direct or causal measure of content quality — a page's visibility can drop for reasons unrelated to content quality (seasonality, algorithm updates, competitor changes).

**Features:** `word_count`, `char_count`, `age_days_dec` (page age as of Dec 1, 2025), `impressions_dec`, `clicks_dec`, `position_dec` (average search position in December).

**Baseline:** a rule-based heuristic — flag any page 180+ days old *and* losing visibility, output `STALE_DECLINING`.

**Model:** Random Forest Classifier (`n_estimators=200`, `max_depth=8`, `class_weight="balanced"`). Chosen because it can learn non-linear interactions between age, impressions, clicks, and position that a single if-statement rule cannot capture.

**Validation design:** time-aware split — all features are known as of December 2025; the label is only knowable once March 2026 closes. No row uses information from after its own decision point, so there is no temporal leakage.

**Leakage check:**
- All features derived from December 2025 data only
- Label derived from March 2026 data only
- No feature contains "march," "target," or any post-December signal

## 4. Results

**The Honest Table** (model vs. baseline, same held-out test split):

| Metric | Baseline Rule | Random Forest |
|---|---|---|
| Accuracy | 0.601–0.602 | — |
| AUC | 0.627 | 0.933–0.935 |
| Precision@50 | 0.000 | 1.000 |

The baseline rule sorts purely by page age, so its "top 50" isn't actually ranking by decline risk — it just picks the oldest pages, many of which are stable. The model learns to weigh age alongside impressions, clicks, and position together, which is why it succeeds where the single-signal rule fails completely at the top of the queue.

**Feature importances:**

| Feature | Importance |
|---|---|
| impressions_dec | 0.478 |
| position_dec | 0.384 |
| age_days_dec | 0.062 |
| word_count | 0.029 |
| char_count | 0.027 |
| clicks_dec | 0.020 |

Impression volume and search position drive the prediction far more than simple page age — nearly 86% of the model's decision weight combined — which is exactly why the model beats the baseline: the baseline only looked at age.

## 5. Limitations

This work cannot claim:
- **Full coverage:** only 36.7% of rows have GSC (Google Search Console) data available in a given month; the model silently excludes the rest.
- **Causality:** the label is a proxy (impression decline), not a causal measure of content quality — it cannot separate genuine content-quality issues from seasonality, algorithm updates, or competitor changes.
- **Generalization beyond this panel:** the model is validated on one historical window (Dec 2025 → March 2026); there is no guarantee of similar performance on future, unseen windows.
- **Perfect recall:** 308 of 8,183 actual declines were missed (a 3.76% false-negative rate). These missed pages had a mean age of ~181 days (just above the 180-day "stale" threshold) and relatively high December impressions (mean ~435) — the model under-flags moderately young pages that were still performing well right before declining, a sudden drop it has no way to anticipate from December's numbers alone.

This produces an **observed, directional, decision-support ranking** — not a causal prediction, and not a guarantee of future Google ranking behavior.

## 6. Ranked Recommendations

The model's output is converted into an action playbook: each page is assigned a reason code and a corresponding action.

| Reason Code | Action | Meaning |
|---|---|---|
| STALE_DECLINING | `refresh_review` | Old page, confirmed losing visibility — top priority |
| STALE_STABLE | `monitor` | Old but not currently declining |
| FRESH | `no_action` | Too new to judge reliably |

To prevent FRESH pages with large raw score swings from outranking genuinely stale, declining pages (a flaw identified during our own audit process), the final ranking is gated by staleness before sorting by model probability. The resulting top-20 queue consists entirely of `STALE_DECLINING` / `refresh_review` pages, with model confidence scores between 0.969 and 0.977.

**Human review is required before acting** on any recommendation — specifically to check for seasonality, algorithm-update timing, or intentional content sunsetting, none of which the model can detect from the available signals.

## 7. Reproducibility

- **Repository:** [github.com/GauriNehe/Flyrank-ml-internship](https://github.com/GauriNehe/Flyrank-ml-internship)
- **Capstone notebook:** `work/notebooks/capstone.ipynb`
- **Supporting notebooks:** `w01`–`w09` in `work/notebooks/`, documenting the full pipeline from research question through validation audit
- **Outputs:** ranked action queue and metrics JSON in `work/outputs/`
- **Figures:** feature importance table in `work/figures/`

Anyone with access to the repository can re-run `capstone.ipynb` end-to-end (`Runtime → Run all`) to reproduce every number in this paper.

## Acknowledgments & Data Credit

Built on the FlyRank ML Internship dataset. Data provided by [FlyRank](https://flyrank.ai).
