import faiss
import numpy as np
from transformers import AutoModel

def build_faiss_index(model_name="fine_tuned_llm"):
    """Create a FAISS index for anomaly embeddings."""
    model = AutoModel.from_pretrained(model_name)
    dummy_data = np.random.rand(1000, 768).astype("float32")  # Mock embeddings
    
    index = faiss.IndexFlatL2(768)
    index.add(dummy_data)
    faiss.write_index(index, "anomaly_index.faiss")
    print("FAISS index saved.")

if __name__ == "__main__":
    build_faiss_index()
