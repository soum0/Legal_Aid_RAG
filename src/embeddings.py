from sentence_transformers import SentenceTransformer

class LocalEmbeddingModel:
    
    def __init__(self,model_name: str = "all-MiniLM-L6-v2"):
        print(f'Loading Embedding Model {model_name}')

        self.model = SentenceTransformer(model_name)

    def embed_texts(self,texts):

        return self.model.encode(texts,
                                 show_progress_bar= True,
                                 convert_to_numpy= True,
                                 normalize_embeddings= True)
    
