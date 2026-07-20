# Retrieval Evaluation Results

Corpus: **35 chunks** from 4 resumes · Eval set: **16 labeled queries**

| Method | Hit@1 | Hit@3 | MRR |
|--------|------:|------:|----:|
| Vector only | 62.5% | 93.8% | 0.776 |
| BM25 only | 68.8% | 100.0% | 0.844 |
| Hybrid (RRF) | 68.8% | 100.0% | 0.844 |
| Hybrid + Rerank | 81.2% | 100.0% | 0.896 |
