FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data_preprocessing.py .
COPY train_llm.py .
COPY faiss_integration.py .

# Download model weights (replace with your fine-tuned model)
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-2-7b-chat-hf')"

CMD ["python", "train_llm.py"]
