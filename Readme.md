# VLM Art Grader

This repository contains a parameter-efficient fine-tuning (PEFT) pipeline that trains a Vision-Language Model to automatically grade children's artwork. It evaluates images across three dimensions: Clarity, Detail, and Creativity.

I built this to solve a specific scaling problem: proprietary APIs like Gemini cost ₹2 to ₹5 per evaluation. By fine-tuning a quantized 2B parameter model, this pipeline brings the inference cost down to under ₹0.10 per image, allowing educational platforms to scale evaluations without breaking the bank.

## Tech Stack
* **Base Model:** Qwen2-VL-2B-Instruct
* **Frameworks:** PyTorch, HuggingFace (Transformers, PEFT, TRL)
* **Optimization:** 4-bit NF4 Quantization, LoRA adapters

## The "Triple-Masking" Fix
If you've tried fine-tuning VLMs, you've probably seen training loss flatline around 2.35. The model usually ends up wasting its gradient budget trying to predict hundreds of meaningless image padding tokens (token ID 151655) instead of learning the actual task.

To fix this, I wrote a custom data collation function that implements "triple-masking." It initializes all labels to `-100` (ignore), searches for the `<|im_start|>assistant` boundary, and strictly unmasks *only* the assistant's generated JSON response. This forces the model to only calculate loss on the actual grading output. After implementing this, the loss dropped from 2.35 to below 0.3 almost immediately.

## Performance
Trained on a dataset of 4,000 images (enriched with Chain-of-Thought reasoning), the model achieves:
* **Mean Absolute Error (MAE):** 0.247
* **Classification Accuracy:** 87.3%
* **JSON Parse Success Rate:** 100%
* **Hardware:** Fits comfortably on a single free-tier Tesla T4 GPU (8GB VRAM during inference).

## Repository Structure
* `synthetic_data_generation.ipynb`: Script using the free Groq API to generate ground-truth CoT grades for the raw KIDO dataset.
* `VLM_Training_and_Inference.ipynb`: The complete training loop, custom collate function, and evaluation metrics.
* `sample_data/`: A few sample images and their expected JSON outputs.

## How to Run
1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
