# Retrieval-Augmented Generation — Miniature Reimplementation

A research-oriented miniature implementation of **Retrieval-Augmented Generation (RAG)** based on the paper *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* by Patrick Lewis et al.

The project reproduces the core RAG pipeline:

**Question → Dense Retrieval → Top-K Documents → Sequence-to-Sequence Generation → Marginalization → Evaluation**

The implementation is designed as an educational and research exercise that demonstrates the main architectural and mathematical ideas of RAG while remaining practical to train in Google Colab.

---

## Project Overview

Large language models store substantial amounts of knowledge in their parameters, but accessing, updating, and inspecting this knowledge can be difficult.

RAG addresses this by combining two forms of memory:

* **Parametric memory:** a pretrained sequence-to-sequence generator.
* **Non-parametric memory:** an external dense vector index containing knowledge passages.

For a given question, a neural retriever searches the external knowledge base for relevant passages. The retrieved passages are then provided to the generator, allowing generation to be conditioned on external evidence.

This project implements a miniature version of this architecture using:

* Sentence-Transformer dense embeddings
* FAISS similarity search
* SQuAD 1.1 passages and questions
* FLAN-T5-base as the sequence-to-sequence generator
* RAG-Sequence-style marginal likelihood
* RAG-Token formulation demonstration
* Exact Match and token-level F1 evaluation
* Retrieval ablation experiments

---

## Original Paper

**Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D.**

*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*

arXiv:2005.11401, 2021.

Official paper: see the arXiv link provided with this repository documentation.

---

## Dataset

### SQuAD 1.1

The final training experiment uses **SQuAD 1.1**, a question-answering dataset containing questions paired with Wikipedia-derived passages.

For this miniature implementation:

* Training examples: **5,000**
* Validation examples: **500**
* Retrieval corpus: unique contexts extracted from the selected training examples
* Retrieval depth: **Top-K = 3** for the main RAG configuration

The original RAG paper instead uses a large Wikipedia knowledge base consisting of approximately 21 million passages. Therefore, the retrieval corpus in this project is intentionally much smaller.

Dataset source:

**Stanford Question Answering Dataset (SQuAD 1.1)**

---

## Repository Structure

```text
retrieval-augmented-generation-reimplementation/
│
├── rag_reimplementation.ipynb
├── README.md
├── summary.md
└── requirements.txt
```

The notebook contains the complete implementation from data preparation through retrieval, generation, training, evaluation, and ablation experiments.

---

## Implementation Pipeline

### 1. Data Preparation

A subset of SQuAD 1.1 is loaded and divided into training and validation examples.

Each example contains:

* Question
* Context
* Answer

### 2. Dense Document Representation

The retrieval corpus is converted into dense vector representations using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are normalized before indexing.

### 3. FAISS Retrieval

The document embeddings are stored in a FAISS inner-product index.

For every question:

```text
Question
   ↓
Query Embedding
   ↓
FAISS Search
   ↓
Top-K Documents
```

### 4. Sequence-to-Sequence Generator

The retrieved passage is concatenated with the question and passed to:

```text
google/flan-t5-base
```

The original paper uses BART-large; FLAN-T5-base is used here to make the implementation practical on Colab.

### 5. RAG-Sequence Training

For each question, the model retrieves multiple candidate documents and evaluates the probability of the target answer conditioned on each document.

The document-level probabilities are marginalized to obtain the RAG-Sequence likelihood.

### 6. Evaluation

The model is evaluated using:

* Exact Match
* Token-level F1
* Top-K retrieval recall

### 7. Ablation Study

The following configurations are compared:

| Configuration | Description                          |
| ------------- | ------------------------------------ |
| No Retrieval  | Generator receives only the question |
| Top-1         | One retrieved document               |
| Top-3         | Three retrieved documents            |
| Top-5         | Five retrieved documents             |

---

## Results

### Retrieval

| Metric                 |        Result |
| ---------------------- | ------------: |
| Top-1 Retrieval Recall | `<TO_UPDATE>` |
| Top-3 Retrieval Recall | `<TO_UPDATE>` |
| Top-5 Retrieval Recall | `<TO_UPDATE>` |

### Answer Generation

| Configuration |   Exact Match |            F1 |
| ------------- | ------------: | ------------: |
| No Retrieval  | `<TO_UPDATE>` | `<TO_UPDATE>` |
| RAG Top-1     | `<TO_UPDATE>` | `<TO_UPDATE>` |
| RAG Top-3     | `<TO_UPDATE>` | `<TO_UPDATE>` |
| RAG Top-5     | `<TO_UPDATE>` | `<TO_UPDATE>` |

### Training

| Parameter           |                Value |
| ------------------- | -------------------: |
| Training examples   |                5,000 |
| Validation examples |                  500 |
| Epochs              |                    1 |
| Top-K               |                    3 |
| Generator           |         FLAN-T5-base |
| Retriever           | Sentence-Transformer |
| Index               |    FAISS IndexFlatIP |
| Learning rate       |                 2e-5 |

> **Note:** The result placeholders above will be replaced after the final training and evaluation runs.

---

## Comparison with the Original Paper

This project reproduces the **core concepts** of the RAG architecture rather than the exact experimental setup.

| Component      | Original Paper                          | This Implementation                      |
| -------------- | --------------------------------------- | ---------------------------------------- |
| Generator      | BART-large                              | FLAN-T5-base                             |
| Retriever      | DPR                                     | Sentence-Transformer                     |
| Knowledge base | ~21M Wikipedia passages                 | SQuAD-derived corpus                     |
| Index          | FAISS                                   | FAISS                                    |
| Retrieval      | Dense MIPS                              | Dense inner-product search               |
| RAG-Sequence   | Implemented                             | Implemented                              |
| RAG-Token      | Implemented conceptually                | Demonstrated                             |
| Training       | End-to-end RAG training                 | Miniature generator-focused training     |
| Dataset        | Multiple knowledge-intensive benchmarks | SQuAD 1.1                                |
| Scale          | Research-scale                          | Colab-scale                              |
| Evaluation     | Multiple benchmark-specific metrics     | EM, F1, retrieval recall                 |
| Ablations      | Extensive paper experiments             | Top-K and retrieval baseline experiments |

The original paper introduces both RAG-Sequence and RAG-Token and uses a pretrained seq2seq generator together with a dense Wikipedia index accessed through a neural retriever.

This implementation preserves that central design while reducing model and dataset scale to make experimentation feasible in a Colab environment.

---

## What I Learned

This project provided hands-on understanding of several components of modern retrieval-augmented NLP systems:

* How dense document representations are created and indexed.
* How FAISS performs efficient similarity search over dense vectors.
* How retrieval and generation can be combined into a single pipeline.
* How external non-parametric memory complements a pretrained language model.
* How latent retrieved documents can be marginalized during training.
* The difference between RAG-Sequence and RAG-Token.
* How retrieval quality directly affects downstream generation.
* How to evaluate both retrieval and answer generation independently.
* How ablation studies can be used to understand the contribution of individual components.
* The practical trade-offs between reproducing a research paper exactly and building a computationally feasible miniature implementation.

Most importantly, the project helped bridge the gap between reading a research paper and implementing its core algorithmic ideas from scratch.

---

## Future Improvements

Several improvements would move this implementation closer to the original research system:

1. **Use a DPR retriever**

   Replace the Sentence-Transformer retriever with the DPR question and document encoders used by the original RAG architecture.

2. **Use a larger knowledge base**

   Build a substantially larger Wikipedia passage index instead of the current SQuAD-derived corpus.

3. **Use BART-large**

   Replace FLAN-T5-base with the BART-large generator used in the original paper.

4. **Jointly fine-tune the query encoder**

   The original RAG architecture fine-tunes the query-side retriever together with the generator while keeping the document encoder and index fixed.

5. **Implement full RAG-Token training**

   Extend the current RAG-Token demonstration into a complete token-level marginal likelihood training procedure.

6. **Improve batching and efficiency**

   Batch retrieval and generator computation to reduce training time and GPU memory usage.

7. **Expand evaluation**

   Evaluate on additional knowledge-intensive benchmarks and compare against stronger parametric-only and retrieval-based baselines.

8. **Reproduce paper-scale experiments**

   With sufficient computational resources, reproduce the original Wikipedia index, retrieval settings, decoding strategies, and benchmark experiments.

---

## References

Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, arXiv:2005.11401.

Rajpurkar et al., *SQuAD: 100,000+ Questions for Machine Comprehension of Text*, EMNLP 2016.

---

## Disclaimer

This repository is an independent educational/research reimplementation of the core ideas presented in the RAG paper. It is intentionally smaller than the original system and should not be interpreted as an exact reproduction of the reported paper results.
