# Retrieval-Augmented Generation — Miniature Reimplementation

## Summary

This project implements the core ideas of **Retrieval-Augmented Generation (RAG)** introduced by Lewis et al. in *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.

The implementation combines a pretrained sequence-to-sequence generator with an external dense retrieval system. Given a question, the system converts the query into a dense representation, retrieves the most relevant passages from a FAISS index, and conditions the generator on the retrieved evidence.

The project implements a miniature **RAG-Sequence** training pipeline using SQuAD 1.1, a Sentence-Transformer dense retriever, FAISS, and FLAN-T5-base. It also demonstrates the RAG-Token formulation and evaluates retrieval and generation performance using retrieval recall, Exact Match, and token-level F1.

The implementation intentionally differs from the original paper in scale and model selection. The paper uses DPR, BART-large, and a roughly 21-million-passage Wikipedia index, whereas this project uses smaller components suitable for experimentation in Google Colab.

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

## Results

Final evaluation results will be added after completion of model training.

* Top-3 Retrieval Recall: `<TO_UPDATE>`
* RAG Exact Match: `<TO_UPDATE>`
* RAG F1: `<TO_UPDATE>`
* No-Retrieval Exact Match: `<TO_UPDATE>`
* No-Retrieval F1: `<TO_UPDATE>`

## Research Value

The primary goal of this project is not to reproduce the original benchmark scores, but to understand and implement the architectural and mathematical principles behind RAG.

The project demonstrates the complete progression from:

**Dense Retrieval → External Knowledge → Conditional Generation → Marginalized RAG Training → Evaluation**

and provides a foundation for future work involving DPR, larger Wikipedia indexes, BART-large, full RAG-Token training, and larger-scale benchmark reproduction.
