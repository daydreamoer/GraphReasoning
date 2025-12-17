from sentence_transformers import SentenceTransformer

model_path = "/home/amax/public/lyh/GraphOTTER/gte-base"
try:
    model = SentenceTransformer(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
