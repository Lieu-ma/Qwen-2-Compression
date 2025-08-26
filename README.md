# Qwen-2-Compression
# Project: Math Problem Classification and Model Optimization with Qwen2-0.5B

## Overview

This project explores the fine-tuning of the `Qwen/Qwen2-0.5B-Instruct` model for a multi-class classification task involving mathematical problems. It is divided into two main parts:

1.  **Fine-Tuning for Classification:** A series of experiments were conducted to fine-tune the model on the "Classification of Math Problems by KASUT Academy" dataset. The goal was to achieve the highest classification accuracy using techniques like Low-Rank Adaptation (LoRA) and quantization (4-bit and 8-bit).

2.  **Model Optimization Analysis:** A systematic investigation into the effects of pruning and quantization on the base model's performance. This analysis uses perplexity on the WikiText dataset as a benchmark to measure performance degradation against model size reduction. Several strategies, including global, single-layer, and ranked-sparsity pruning, were evaluated.

The project utilizes a multi-GPU setup with PyTorch's Distributed Data Parallel (DDP) for efficient training.

---

## Part 1: Math Problem Classification

### Objective
To classify math problems into one of eight distinct categories using the `Qwen/Qwen2-0.5B-Instruct` model.

### Methodology
- **Dataset:** "Classification of Math Problems by KASUT Academy". The data was split into a training set (9170 samples) and a validation set (1019 samples).
- **Fine-Tuning:** Low-Rank Adaptation (LoRA) was used to efficiently fine-tune the model.
- **Quantization:** Experiments were run using 4-bit (NF4) and 8-bit quantization via the `bitsandbytes` library to assess the performance-efficiency trade-off.
- **Training:** The model was trained for 5 epochs using the AdamW optimizer, a batch size of 4, and 4 gradient accumulation steps on a 4x GPU setup.

### Classification Results Summary
The model was fine-tuned under various configurations on a 90/10 train/validation split. The primary evaluation metric was the F1-micro score.

| Experiment Description                               | Quantization | LoRA Config | Best F1-Micro | Best F1-Macro |
| ---------------------------------------------------- | :----------: | :---------: | :-----------: | :-----------: |
| **Standard LoRA (Baseline)** |    **8-bit** | `r=16, α=32`  |   **0.8822** |   **0.8357** |
| Standard LoRA (Baseline)                             |    4-bit     | `r=16, α=32`  |    0.8763     |    0.8262     |
| "Low and Slow" (High Capacity, Low LR)               |    8-bit     | `r=32, α=64`  |    0.8665     |    0.8305     |
| "Low and Slow" (High Capacity, Low LR)               |    4-bit     | `r=32, α=64`  |    0.8724     |    0.8311     |
| Max Performance (Unquantized, High Capacity)         |     None     | `r=32, α=64`  |    0.8695     |    0.8295     |
| **Dynamic Adaptive Hybrid (Novelty)** | **4/8-bit** | `r=16, α=32`  |   **0.8813** |      N/A      |

**Conclusion:** The standard **8-bit quantized model** with a LoRA rank of 16 and alpha of 32 achieved the highest F1 scores, providing an excellent balance of high performance and resource efficiency. The novel "Dynamic Adaptive Hybrid" approach, which mixed 4-bit and 8-bit layers, also showed very promising results.

---

## Part 2: Pruning & Quantization Analysis

### Objective
To measure the impact of unstructured pruning and post-pruning quantization on the language modeling capabilities of the `Qwen/Qwen2-0.5B-Instruct` model.

### Methodology
- **Benchmark:** Perplexity was evaluated on the `wikitext-2-raw-v1` dataset. A lower perplexity score indicates better performance. The baseline perplexity of the original float16 model was **12.60**.
- **Pruning:** Unstructured L1 pruning was applied.
- **Quantization:** 4-bit (NF4) and 8-bit quantization were applied after pruning.

### Key Findings & Perplexity Results

| Pruning Strategy                               | Sparsity          | Quantization | Perplexity |
| ---------------------------------------------- | :---------------: | :----------: | :--------: |
| **Baseline (No Pruning)** |      **0%** |   **None** | **12.60** |
| Baseline (No Pruning)                          |        0%         |    8-bit     |   12.65    |
| Baseline (No Pruning)                          |        0%         |    4-bit     |   13.72    |
| **Global Pruning (All Layers)** |      **25%** |   **None** | **15.21** |
| Global Pruning (All Layers)                    |        50%        |     None     |   184.59   |
| **Ranked-Sparsity Pruning (Top 4 Layers)** | **25% -> 50%** |   **8-bit** | **13.85** |
| Ranked-Sparsity Pruning (Top 4 Layers)         |   25% -> 50%      |    4-bit     |   15.13    |
| **Mixed-Sparsity Pruning (2 Groups)** | **50% / 25%** |   **8-bit** | **21.63** |
| Mixed-Sparsity Pruning (2 Groups)              |     50% / 25%     |    4-bit     |   24.32    |

**Analysis Conclusion:**
- **Quantization:** 8-bit quantization has a negligible impact on model perplexity, whereas 4-bit quantization causes a minor increase.
- **Pruning:** Pruning all layers equally (Global Pruning) beyond 25% sparsity leads to a significant and rapid degradation in performance.
- **Advanced Pruning:** The **Ranked-Sparsity** approach, where larger layers are pruned more aggressively, proved to be the most effective strategy, maintaining a low perplexity even after quantization. The 8-bit ranked-pruned model had a perplexity of **13.85**, very close to the 4-bit unpruned baseline.

---

## How to Run the Experiments

The project is structured within three Jupyter notebooks. The core logic is encapsulated in Python scripts that are written and executed from within the notebooks.

### Dependencies
Install all required packages using pip:
```bash
pip install -U -q torch transformers datasets evaluate bitsandbytes peft pandas scikit-learn numpy tqdm accelerate
