# Community-Aware Aspect-Based Sentiment Analysis

Do different groups of reviewers feel differently about the same product aspects (battery, price, screen...)?
This project finds reviewer communities from review histories and tests whether adding community information helps aspect-level sentiment prediction.

## Pipeline
1. **Data:** Amazon Reviews 2023, Cell Phones & Accessories (5-core), sampled to about 79k reviews. SemEval-2014 (Laptops, Restaurants) is used to validate the sentiment model.
2. **Splits (no leakage):** train, val, test_unseen_reviews, test_unseen_items, test_unseen_reviewers, plus reviewer_history for new reviewers.
3. **Aspects and sentiment:** spaCy plus a 13-aspect phone lexicon for aspects, and a pretrained DeBERTa ABSA model as the "teacher" for sentiment (pseudo-labels).
4. **Graph:** heterogeneous graph (user, review, item, aspect) built from the train split only.
5. **Communities:** Leiden on a reviewer graph (shared products plus aspect-mention similarity, never sentiment). K = 8. New reviewers are assigned using their history only.
6. **Models:** text-only, user-aware and community-aware (all DistilRoBERTa), plus a GCN graph baseline.
7. **Interpretation:** community x aspect heatmap and an item-controlled significance test.

## Results (macro-F1, mean over 3 seeds)
| Model | Unseen reviews | Unseen items | Unseen reviewers |
|---|---|---|---|
| GCN (graph only) | 0.3890 | 0.4469 | 0.3755 |
| Text-only | 0.7814 | 0.7738 | 0.7910 |
| User-aware | 0.7817 | 0.7752 | 0.7912 |
| Community-aware | 0.7852 | 0.7762 | 0.7882 |

Teacher on SemEval-2014: Laptops acc 0.828 / macro-F1 0.790; Restaurants acc 0.837 / macro-F1 0.793.

Heatmap: `results/heatmap.png`

## Key findings
- Text-only, user-aware and community-aware models perform almost the same (macro-F1 about 0.77 to 0.79). Community information gives no consistent gain: it is slightly ahead on unseen reviews and unseen items, and slightly behind on unseen reviewers. The differences are within seed-to-seed variation.
- Community differences in aspect sentiment are significant for price, quality, screen and charging, but not after removing each product's average sentiment. They are largely explained by product type.
- The graph-only GCN is far weaker (macro-F1 about 0.38 to 0.45), so review text carries most of the signal.
- Community assignment for unseen reviewers works without leakage (it uses only their history reviews), but it did not improve prediction for them in this dataset.

## Limitations
- Sentiment labels are teacher pseudo-labels, not human annotations.
- One product category; about 63% of reviews have at least one detected aspect.
- The teacher may have seen SemEval-2014 data in pretraining.
- The unseen-reviewers test set is small (1,335 rows).

## Reproduce
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/download_amazon.py
python scripts/make_splits.py
python -m scripts.run_aspects
python -m src.graph.build
python -m src.graph.communities --gamma 1.0 --min_size 300
python -m src.models.absa --mode text      # also: user, community, with --seed 0/1/2
python scripts/gcn_baseline.py
python src/models/make_table.py
python scripts/interpret_communities.py
```
Data is not stored in the repo because it is large; the scripts download and rebuild it.

## Authors
Shalini V K - 24MID0308
Ananya H Narasimha - 24MID0174
