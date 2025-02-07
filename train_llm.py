import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

def fine_tune_llm(tokenized_data, model_name="meta-llama/Llama-2-7b-chat-hf"):
    """Fine-tune LLM for anomaly classification."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Mock labels (0=normal, 1=anomaly)
    labels = torch.randint(0, 2, (tokenized_data["input_ids"].shape[0],))
    dataset = Dataset.from_dict({k: v for k, v in tokenized_data.items()}).add_column("labels", labels)
    
    # Training setup
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_dir="./logs",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    model.save_pretrained("./fine_tuned_llm")

if __name__ == "__main__":
    tokenized_data = torch.load("tokenized_logs.pt")  # Load preprocessed data
    fine_tune_llm(tokenized_data)
