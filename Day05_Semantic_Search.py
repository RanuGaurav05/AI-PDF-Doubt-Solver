from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np


#Function to load index and chunk data.
def load_index():
    with open('faiss_index.pkl','rb') as file:
        index = faiss.deserialize_index(pickle.load(file))

    with open('chunks_texts.pkl','rb') as file:
        chunks_texts = pickle.load(file)
    
    return index , chunks_texts

#load_model
model = SentenceTransformer('all-MiniLM-L6-v2')

#semantic search
def semantic_search(question, index, chunks_texts):
    """
    1. user question.
    2. embed the question in form of vectors.
    3. search FAISS for the top_k chunk vector.
    4. return the matching chunks text.
    """

    #step1: Embed the question
    question_vector = model.encode([question]).astype(np.float32)

    #Step2: Search Faiss
    #1. distances -> how far each result is (lower = more similar)
    #2. indices   -> position of matchig chunk in our chunks.
    distance, indices = index.search(question_vector, k=3)

    #Step3: Reterive chunk texts
    results = [chunks_texts[i] for i in indices[0]]
    
    #we will return results
    return results


#display results
def display_results(results):
    print("\nTop Matching Chunks:")
    for i, chunk in enumerate(results, 1):
        print(f"{i}. {chunk}\n")

if __name__ == "__main__":
    index, chunks_texts = load_index()
    user_question = input("Enter your question: ")
    semantic_search = semantic_search(user_question, index, chunks_texts)
    display_results(semantic_search)