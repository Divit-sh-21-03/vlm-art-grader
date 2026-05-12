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
Trained on a dataset of 4,000 images (enriched with Chain-of-Thought reasoning) and evaluated on 406 test samples, the model achieves:
* **Mean Absolute Error (MAE):** 0.247
* **Root Mean Squared Error (RMSE):** 0.556
* **Classification Accuracy:** 87.3% (Overall F1 Score: 0.751)
* **JSON Parse Success Rate:** 100% (0 parse failures)
* **Inference Speed:** ~10.69 seconds per image 
* **Hardware:** Fits comfortably on a single free-tier Tesla T4 GPU (8GB VRAM during inference).

##  Live Model Weights
I have hosted the trained LoRA adapters on Hugging Face so you can download and test the model without retraining it from scratch:
**[View Model on Hugging Face](https://huggingface.co/Divit56/VLM_grader)**

```python
from peft import PeftModel
# Load the live weights directly
model = PeftModel.from_pretrained(base_model, "Divit56/VLM_grader")
```

## Repository Structure
* `Codes/Synthetic_data_generator.ipynb`: Script using the free Groq API to generate ground-truth Chain-of-Thought grades for the raw KIDO dataset.
* `Codes/vlm_art_grader.ipynb`: The complete training loop, custom collate function, and evaluation metrics.
* `dataset/graded_dataset.json`: A sample of the dataset containing images graded using Groq's Llama 3.2 Vision API.
* `output.png`: A sample visual output demonstrating the model's grading capabilities and predictions.

## How to Run
1. Clone the repo and install the required dependencies:
   ```bash
   git clone [https://github.com/Divit-sh-21-03/vlm-art-grader.git](https://github.com/Divit-sh-21-03/vlm-art-grader.git)
   cd vlm-art-grader
   pip install -r requirements.txt
   ```
2. (Optional) Run `Codes/Synthetic_data_generator.ipynb` if you want to rebuild the dataset from scratch using the Groq API.
3. Open `Codes/vlm_art_grader.ipynb` in Jupyter or Kaggle. Ensure your runtime has a GPU with at least 14GB VRAM (like a Tesla T4) and run the cells sequentially to train and evaluate the model.
