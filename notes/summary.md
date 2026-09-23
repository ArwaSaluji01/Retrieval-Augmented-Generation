# Retrieval-Augmented Generation — Miniature Reimplementation

## Project Summary

This project is a miniature implementation of Retrieval-Augmented Generation (RAG), based on Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.

The system combines:

1. A dense sentence-embedding retriever
2. A FAISS vector index
3. Retrieved SQuAD contexts as non-parametric memory
4. A FLAN-T5 sequence-to-sequence generator
5. RAG-Sequence-style marginal likelihood training
6. Exact Match and token-level F1 evaluation
7. Retrieval recall and ablation analysis

The implementation is intentionally smaller than the original paper's system so that the core RAG mechanism can be studied and trained in Google Colab.

The current results demonstrate that adding retrieved external context substantially improves performance over the generator-only baseline.


## Key Components

* Dense document embeddings
* FAISS vector indexing
* Top-K dense retrieval
* Sequence-to-sequence generation
* RAG-Sequence marginal likelihood
* RAG-Token formulation
* SQuAD-based training and evaluation
* Retrieval recall
* Exact Match and F1 evaluation
* Retrieval and Top-K ablation studies

## Main Research Question

How can external non-parametric memory improve a pretrained generative model on knowledge-intensive question answering?

## Current Results

Evaluation was performed on 500 SQuAD validation examples.

| Metric                   |     Result |
| ------------------------ | ---------: |
| RAG Exact Match          | **40.20%** |
| RAG Token F1             | **52.33%** |
| Top-3 Retrieval Recall   | **74.00%** |
| No-Retrieval Exact Match |  **1.00%** |
| No-Retrieval Token F1    |  **5.33%** |

The miniature RAG system substantially improves answer quality compared with the generator-only baseline.

However, the current Top-1/Top-3/Top-5 ablation should not yet be interpreted because the inference function currently returns the answer generated from the highest-ranked retrieved document. A proper K-dependent decoding implementation remains a future improvement.

## Research Value

The primary goal of this project is not to reproduce the original benchmark scores, but to understand and implement the architectural and mathematical principles behind RAG.

The project demonstrates the complete progression from:

**Dense Retrieval → External Knowledge → Conditional Generation → Marginalized RAG Training → Evaluation**

and provides a foundation for future work involving DPR, larger Wikipedia indexes, BART-large, full RAG-Token training, and larger-scale benchmark reproduction.
