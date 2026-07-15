# 👁️ Vision-Language Model Fine-Tuning on ScienceQA

A practical implementation of a Vision-Language Model (VLM) pipeline using **Qwen3.5-0.8B** for multimodal question answering on the **ScienceQA** benchmark. The project explores zero-shot inference, the impact of image resolution on performance, and parameter-efficient fine-tuning with LoRA.

## Features

- 🖼️ Zero-shot multimodal question answering
- 📚 Evaluation on the ScienceQA dataset
- ⚡ Image resolution vs. inference speed analysis
- 🎯 Accuracy comparison across different resolutions
- 🔧 Parameter-Efficient Fine-Tuning (LoRA)
- 📊 Performance comparison before and after fine-tuning

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- Qwen3.5-0.8B
- ScienceQA
- Matplotlib

## Project Workflow

```text
        ScienceQA Dataset
               │
      ┌────────┴────────┐
      │                 │
      ▼                 ▼
 Image Processing   Text Processing
      │                 │
      └────────┬────────┘
               ▼
     Vision-Language Model
      (Qwen3.5-0.8B)
               │
      ┌────────┼────────┐
      │        │        │
      ▼        ▼        ▼
 Zero-shot  Resolution  LoRA
 Evaluation   Analysis  Fine-Tuning
      │
      ▼
 Performance Comparison
```

## Experiments

### 1. Zero-shot Inference

- Evaluated the base VLM on ScienceQA without additional training.
- Measured prediction accuracy on multimodal questions.

### 2. Resolution Analysis

Compared:

- High-resolution images
- Low-resolution images

Metrics:

- Inference time
- Prediction accuracy

### 3. LoRA Fine-Tuning

- Fine-tuned the model using PEFT (LoRA).
- Compared the fine-tuned model against the baseline.
- Evaluated improvements in ScienceQA performance.

## Example

**Question**

 Select the fish below.
Choices:
- piranha
- zebra
- Answer concisely with the correct choice.

Ground Truth: piranha

<img width="328" height="328" alt="image" src="https://github.com/user-attachments/assets/b4be2a91-d739-4e6b-b023-1d80272957d0" />

**Prediction**

piranha


## Results

The model was evaluated on the ScienceQA benchmark under three different settings.

| Model | Resolution | Inference Time | Accuracy |
| :--- | :--- | ---: | ---: |
| Baseline | High Resolution (Default) | **572.10 s** | **65%** |
| Baseline | Low Resolution (32 × 28 × 28) | **157.27 s** | **51%** |
| LoRA Fine-Tuned | Low Resolution (32 × 28 × 28) | **114.00 s** | **80%** |

### Key Findings

- Reducing the image resolution decreased the inference time from **572.10 s** to **157.27 s** (≈3.6× faster), but reduced accuracy from **65%** to **51%**.
- After LoRA fine-tuning, the model achieved **80% accuracy** while maintaining a low inference time (**114 s**).
- LoRA fine-tuning not only recovered the performance loss caused by lower image resolution but also outperformed the high-resolution baseline by **15 percentage points**.

## Future Improvements

- QLoRA fine-tuning
- Larger Vision-Language Models
- Multi-GPU training
- More comprehensive evaluation metrics
- Support for additional multimodal benchmarks
