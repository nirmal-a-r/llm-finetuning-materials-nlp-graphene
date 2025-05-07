# 🧪 LLM Fine-Tuning for Material Informatics: Graphene-Focused Scientific Q&A

This project showcases the fine-tuning and benchmarking of large language models (LLMs) for domain-specific **question answering** and **summarization** in **materials science**, with a primary focus on **graphene-based research**. It explores how Natural Language Processing (NLP) can accelerate scientific discovery in the 2D materials space.

---

## 📌 Project Overview

Graphene is a revolutionary two-dimensional material with remarkable strength and conductivity. Extracting insights from the vast literature on graphene is a challenge. This project proposes a scalable **NLP-based pipeline** to automate knowledge extraction from scientific abstracts, focusing on:

- ✅ Scientific Q&A generation
- ✅ Summarization of domain-specific content
- ✅ Fine-tuning of transformer-based LLMs on real research data

---

## 🧠 Models Used

Four transformer-based models were evaluated for domain adaptation:

| Model            | Task Type            | Role              |
|------------------|----------------------|-------------------|
| **DistilBART-CNN** | Abstractive Summarization | Decoder |
| **BART-Base**      | Summarization + Q&A  | Decoder |
| **Pegasus-XSUM**   | Summarization        | Decoder |
| **T5-Small**       | Multi-task (Q&A + Summary) | Decoder |

---

## 🧾 Dataset Construction

- ✅ **Source**: Graphene-based abstracts from **Scopus**
- ✅ **Size**: 1000+ manually verified abstract-QA pairs
- ✅ **Preprocessing**:
  - Stopword removal
  - Scientific term normalization
  - Tokenization and vector embedding
- ✅ **Split**: 80% Train / 20% Test

---

## ⚙️ Methodology

1. **Text Embedding** with domain-specific encoder (BERT / SciBERT / MatBERT)
2. **Contextual Retrieval** of relevant abstracts per question
3. **Decoder Fine-Tuning** using encoder outputs and generated answers
4. **Evaluation** using:
   - 🔹 **ROUGE**: Lexical overlap
   - 🔹 **BERTScore**: Semantic similarity
   - 🔹 **Cosine Similarity**: Embedding match
   - 🔹 **Perplexity**: Language fluency

---

## 📈 Visualizations

- 🔸 Cumulative metric trends for 1000 Q&A pairs
- 🔸 Model convergence and error analysis
- 🔸 Domain-specific vocabulary influence

---

## 📍 Highlights

- ✅ Built entirely from **real scientific literature**
- ✅ Evaluated **BLEU, ROUGE, Perplexity**, and **BERTScore**
- ✅ Compared performance of **domain-specific embeddings** (SciBERT, MatBERT) vs generic (BERT)
- ✅ Developed a **reproducible pipeline** using Hugging Face + PyTorch

---

## 📚 Future Work

- Integrate graph-based models for structure–property relation mining  
- Use **knowledge graphs** for abstract linking  
- Extend to **battery materials**, **perovskites**, and other 2D compounds  

---

## 🧑‍🎓 Author

- **Nirmal A.R.**  
  B.Tech, Artificial Intelligence & Data Science  
  [Amrita Vishwa Vidyapeetham – Coimbatore](https://www.amrita.edu/)

**Supervisor**: Dr. Kritesh K. Gupta  
Assistant Professor, School of AI  
Amrita Center for Computational Engineering & Networking

---

## 📁 File Structure

fine_tuning_project/

    Group_No._8/
        PROJECT FINAL PRESENTATION.pptx
        PROJECT REPORT.docx
        COLAB CODES/
            20% TEST_DATA EVALUATION GRAPH CODE.ipynb
            FINAL GRAPH CODE-4 MODELS.ipynb
            FT_LLM_BARTBASE.ipynb
            FT_LLM_DISTILBART.ipynb
            FT_LLM_PEGASUS-XSUM.ipynb
            FT_LLM_T5-SMALL.ipynb
            MODELS_COMBINED_FT_GRAPH.ipynb
        DATASETS/
            abstract_summaries_gptstyle.csv
            qanda.csv
        OUTPUT GRAPHS/
            20% TEST_DATA ALL 4 MODEL OUTPUT.jpg
            ALL MODEL COMPARISON GRAPH-MAJOR.png
            ALL MODEL EVALUATION TABLE.png
            BARTBASE OUTPUT.png
            DISTILBART OUTPUT.png
            PEGASUS XSUM OUTPUT.png
            T5-SMALL OUTPUT.png

## 🧠 Models Fine-Tuned

We used Hugging Face models pre-trained on large corpora and fine-tuned them on domain-specific scientific abstracts:

- `facebook/bart-base`
- `sshleifer/distilbart-cnn-12-6`
- `google/pegasus-xsum`
- `t5-small`

---

## 🧪 Evaluation Metrics

- **BLEU Score**: To measure overlap in generated vs reference answers.
- **Perplexity**: To evaluate fluency of generated text.
- **Graphical Comparisons**: Plots for all models across 80% and 20% test slices and exclusive evaluation with 1000 Q&A pair dataset.

---

## 📊 Key Outcomes

- DistilBART outperformed others in BLEU while T5-small maintained the lowest perplexity.
- All models were evaluated on the **1000 Q&A pair dataset**, offering comprehensive insights into their performance across multiple metrics.
- Model performance was visualized across different test sizes and query types.
- A combined comparative analysis offered insights into trade-offs between model size and accuracy.
