import pandas as pd
from transformers import AutoTokenizer

def preprocess_logs(log_file, model_name="meta-llama/Llama-2-7b-chat-hf"):
    """Preprocess network logs for LLM fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load and clean logs (example format: timestamp, source_ip, action, protocol)
    df = pd.read_csv(log_file)
    df["text"] = df.apply(lambda row: f"Source IP: {row.source_ip}, Action: {row.action}, Protocol: {row.protocol}", axis=1)
    
    # Tokenize logs
    tokens = tokenizer(df["text"].tolist(), padding="max_length", truncation=True, return_tensors="pt")
    return tokens

# Example usage
if __name__ == "__main__":
    tokens = preprocess_logs("network_logs.csv")
    print(f"Tokenized logs shape: {tokens['input_ids'].shape}")
