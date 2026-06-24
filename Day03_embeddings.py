
"""
#Pretrained model for generating sentence embeddings.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#WE created our own sample sentence to test.
sentence = "Machine learning is a subset of AI."

#.encode() will generate the embedding of the input sentence.
embeddings = model.encode(sentence)
print(f"Sentence: {sentence}")
print(f"Embedding: {embeddings}")
print(f"Embedding: {len(embeddings)}")
"""



#---------Main Execution-----------------
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#function to open the chunk.pkl file
def load_chunks():
    with open("Chunks.pkl", "rb") as file:
        return pickle.load(file)

#A function block which will generate the embedings for the chunks.
def embed_all_chunks(chunks):
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)
    return embeddings


"""
#A function to check similarity between chunks
def check_similarity(embeddings):
    similarities = model.similarity(embeddings, embeddings)
    return similarities
similar = check_similarity(embedded)
"""


def save_embeddings(embeddings):
    np.save("Embeddings.npy", embeddings)
    print("Embeddings saved successfully")



chunks = load_chunks()
embedded = embed_all_chunks(chunks)
save_embeddings(embedded)