# Retrieval-Augmented Generation — Miniature Reimplementation

A research-oriented miniature implementation of **Retrieval-Augmented Generation (RAG)** based on the paper *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* by Patrick Lewis et al.

The project reproduces the core RAG pipeline:

**Question → Dense Retrieval → Top-K Documents → Sequence-to-Sequence Generation → Marginalization → Evaluation**

The implementation is designed as an educational and research exercise that demonstrates the main architectural and mathematical ideas of RAG while remaining practical to train in Google Colab.

---

## Project Overview

This project implements a miniature research-oriented version of Retrieval-Augmented Generation (RAG), based on Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.

The implementation demonstrates the core RAG pipeline:

**Question → Dense Retrieval → FAISS Document Index → Retrieved Context → Seq2Seq Generation → RAG-Sequence Training → Evaluation**

The original RAG architecture combines a pretrained neural retriever, a non-parametric document index, and a pretrained sequence-to-sequence generator, with retrieved documents treated as latent variables during generation.

Because the original experiments use large-scale Wikipedia indexes, pretrained DPR retrieval, BART-large, and substantially larger computational resources, this project uses a smaller SQuAD-based dataset and lightweight pretrained models suitable for Google Colab.

The goal is to reproduce and understand the **core algorithmic ideas**, rather than reproduce the original paper's exact experimental scale or reported scores.

---

## Original Paper

**Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D.**

*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*

arXiv:2005.11401, 2021.

Official paper: see the arXiv link provided with this repository documentation.

---

## Dataset

The implementation uses the **SQuAD dataset** for a miniature open-domain question-answering experiment.

The dataset is divided into:

* **Training:** 2,000 examples used for miniature RAG training
* **Validation:** 500 examples used for evaluation
* **Retrieval corpus:** unique contexts extracted from the training split

Each retrieval document contains:

* document ID
* title
* context text

The retrieval corpus is embedded using `sentence-transformers/all-MiniLM-L6-v2` and indexed with FAISS using inner-product similarity.

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

Evaluation was performed on **500 SQuAD validation examples**.

| System       | Exact Match |   Token F1 |
| ------------ | ----------: | ---------: |
| No Retrieval |       1.00% |      5.33% |
| RAG          |  **40.20%** | **52.33%** |

### Retrieval Performance

| Metric                 |     Result |
| ---------------------- | ---------: |
| Top-3 Retrieval Recall | **74.00%** |

The results show a substantial improvement when retrieved context is provided to the generator. In this miniature implementation, Exact Match increases from **1.00% without retrieval to 40.20% with retrieval**, while token-level F1 increases from **5.33% to 52.33%**.

These results are specific to this implementation, dataset split, pretrained models, and training configuration. They should not be compared directly with the original paper's reported results.

### Top-K Ablation

A Top-1/Top-3/Top-5 ablation is implemented, but the current generation function selects the answer produced from the highest-ranked retrieved document. Consequently, the current Top-K scores are identical and should **not** be interpreted as evidence that K has no effect.

A proper Top-K decoding comparison is a planned improvement.

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

## Differences from the Original Paper

This implementation intentionally uses a smaller setup suitable for Google Colab.

| Original RAG                          | This implementation                      |
| ------------------------------------- | ---------------------------------------- |
| Wikipedia-scale non-parametric memory | SQuAD-derived retrieval corpus           |
| DPR retriever                         | MiniLM sentence embeddings               |
| FAISS Wikipedia index                 | FAISS `IndexFlatIP`                      |
| BART-large generator                  | FLAN-T5-base                             |
| Joint retriever + generator training  | Generator-focused miniature training     |
| Large-scale training                  | 2,000 training examples                  |
| Full RAG-Sequence decoding            | Simplified retrieved-document generation |
| Large-scale experiments               | 500-example validation evaluation        |

The original paper describes RAG-Sequence as marginalizing generation probabilities across retrieved documents and uses document-wise decoding for RAG-Sequence inference.

Therefore, this project should be viewed as a **conceptual and algorithmic reimplementation**, not an exact reproduction of the original training setup.

---

## What I Learned

This implementation helped demonstrate several key RAG concepts:

* How dense vector representations can be used for semantic retrieval.
* How FAISS performs efficient similarity search over document embeddings.
* How retrieved documents can be incorporated into a sequence-to-sequence generator.
* How RAG-Sequence models marginalize over multiple retrieved documents during training.
* How retrieval quality affects downstream answer generation.
* How to evaluate a retrieval-augmented system using both retrieval metrics and answer-generation metrics.
* Why retrieval and generation should be evaluated separately rather than relying only on final answer accuracy.
* The practical differences between a research paper's full-scale implementation and a resource-constrained educational reproduction.

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
