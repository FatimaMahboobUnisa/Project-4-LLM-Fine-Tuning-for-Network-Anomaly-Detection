# LLM Fine-Tuning for Network Anomaly Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-4.34.0-yellow)](https://huggingface.co/)
[![AWS](https://img.shields.io/badge/AWS-EC2%2C%20S3-orange)](https://aws.amazon.com/)

Fine-tuned LLaMA-2 to detect network anomalies using Hugging Face Transformers and FAISS for similarity search. Deployed on AWS with Docker.

## 📋 Project Overview
- **Objective**: Improve anomaly detection accuracy with LLMs.
- **Tools**: LLaMA-2, PyTorch, Hugging Face, FAISS, AWS.
- **Key Results**: 35% accuracy boost, 20% latency reduction.

## 🚀 Features
- **LLM Fine-Tuning**: Adapted LLaMA-2 for network log classification.
- **Vector Database**: FAISS for efficient similarity search.
- **AWS Deployment**: Dockerized inference API on EC2.

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/llm-anomaly-detection.git
   cd llm-anomaly-detection
