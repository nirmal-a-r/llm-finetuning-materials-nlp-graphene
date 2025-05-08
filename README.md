# 🧪 LLM Fine-Tuning for Material Informatics: Graphene-Focused Scientific Q&A

This project explores fine-tuning and benchmarking large language models (LLMs) for domain-specific question answering and summarization in materials science, with a special focus on **graphene-based research**. It demonstrates how NLP can accelerate scientific discovery in the 2D materials domain.

---

## 📌 Project Overview

Graphene, a two-dimensional material known for its exceptional strength and conductivity, has a vast and complex body of research literature. This project builds a scalable NLP pipeline to automate the extraction of insights from scientific abstracts.

### Key Goals:
- ✅ Scientific Q&A generation  
- ✅ Abstractive summarization of research literature  
- ✅ Fine-tuning transformer-based LLMs on real-world graphene data  

---

## 🧠 Models Used

| Model              | Task Type                   | Role     |
|--------------------|-----------------------------|----------|
| DistilBART-CNN     | Abstractive Summarization   | Decoder  |
| BART-Base          | Summarization + Q&A         | Decoder  |
| Pegasus-XSUM       | Abstractive Summarization   | Decoder  |
| T5-Small           | Multi-task (Q&A + Summary)  | Decoder  |

---

## 🧾 Dataset Construction

- **Source**: Scopus-indexed graphene abstracts  
- **Size**: ~1200 entries  
- **Format**: `abstract_summaries_gptstyle.csv`, `qanda.csv`  

---

## ⚙️ Methodology

- **Tokenization**: Max input length = 128 tokens, output length = 32 tokens  
- **Fine-Tuning**:  
  - Epochs: 5  
  - Batch Size: 2 (Effective: 4 with gradient accumulation)  
  - Learning Rate: `5e-5`  
  - Weight Decay: `0.01`  
  - Optimizations: FP16 precision, gradient checkpointing  
  - Inference: Beam search with 4 beams  

- **Evaluation Metrics**:
  - 🔹 ROUGE-1, ROUGE-2, ROUGE-L (lexical overlap)  
  - 🔹 BERTScore (semantic similarity - F1)  
  - 🔹 Cosine Similarity (embedding alignment)  
  - 🔹 Perplexity (fluency)  

---

## 📈 Visualizations

- 📊 Cumulative performance plots (1000 Q&A pairs):  
  - `cumulative_rouge1.png`  
  - `cumulative_bertscore.png`  
  - `cumulative_cosine_similarity.png`  
  - `cumulative_perplexity.png`  

- 📉 Training convergence plots and model-specific performance trends  
- 📁 Stored under: `plots/` directory  

---

## 📍 Highlights

- ✅ Real scientific literature from Scopus  
- ✅ Evaluated across 4 NLP metrics  
- ✅ ROUGE-1 improved by **+214%** for DistilBART-CNN  
- ✅ BERT-F1 **0.9068** achieved with Pegasus-XSUM  
- ✅ Built using Hugging Face 🤗 Transformers + PyTorch  

---

## 📚 Future Work

- Integrate graph-based models for structure–property extraction  
- Link abstracts via knowledge graphs  
- Expand to other 2D materials (e.g., perovskites, battery compounds)  

---

## 🧑‍🎓 Author

**Nirmal A.R.**  
B.Tech – Artificial Intelligence & Data Science  
Amrita Vishwa Vidyapeetham, Coimbatore  

**Supervisor**:  
Dr. Kritesh K. Gupta  
Assistant Professor, School of AI  
Amrita Center for Computational Engineering & Networking  

---

## 📁 Project Structure

```

llm-finetuning-materials-nlp-graphene/
├── data/
│   ├── abstract\_summaries\_gptstyle.csv
│   ├── qanda.csv
├── plots/
│   ├── cumulative\_rouge1.png
│   ├── cumulative\_bertscore.png
│   └── ...
├── preprocess\_data.py
├── train\_bart.py
├── train\_distilbart.py
├── train\_pegasus.py
├── train\_t5.py
├── evaluate\_models.py
├── requirements.txt

````

---

## 🧠 Models Fine-Tuned

| Hugging Face Model Name             | Purpose                  |
|------------------------------------|--------------------------|
| `facebook/bart-base`               | Q&A + Summarization      |
| `sshleifer/distilbart-cnn-12-6`    | Summarization            |
| `google/pegasus-xsum`              | Summarization            |
| `t5-small`                         | Multi-task               |

---

## 🧪 Evaluation Summary

| Metric              | Best Model        | Value      |
|---------------------|------------------|------------|
| ROUGE-1             | DistilBART-CNN    | 0.6952     |
| BERT-F1             | Pegasus-XSUM      | 0.9068     |
| ROUGE-L             | Pegasus-XSUM      | 0.3206     |
| Perplexity          | T5-Small          | 5.7769     |

✅ All models outperformed baselines like vanilla BERT (BERT-F1 gain: +12–38%).

---

## 🚀 Setup & Usage

### ✅ Prerequisites

- Python 3.8+  
- GPU recommended  
- Dependencies: See `requirements.txt`

### 🔧 Installation

```bash
git clone https://github.com/nirmal-a-r/llm-finetuning-materials-nlp-graphene.git
cd llm-finetuning-materials-nlp-graphene
pip install -r requirements.txt
````

### 🚀 Quick Start

```bash
# Step 1: Preprocess the data
python preprocess_data.py

# Step 2: Train a model (example: T5)
python train_t5.py

# Step 3: Evaluate the model
python evaluate_models.py --model t5-small

# Step 4: Visualize results
open plots/cumulative_rouge1.png  # or check other images
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full details.

---

## 📬 Contact

For questions, collaborations, or issues, please open an issue on GitHub or reach out via email (available upon request).

```

