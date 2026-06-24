import pickle
import numpy as np
import faiss


#1. Load chunk data.
def load_chunk():
    with open('Chunks.pkl','rb') as file:
        data = pickle.load(file)
    return data

#2. load embeddings.
def load_embeddings():
    return np.load('Embeddings.npy')

#3. Build Faiss index.
def build_faiss(embeddings):
    #Seprate number of chunks and dimension
    num_chunks, dimension = embeddings.shape

    #FAISS requires float32
    embeddings_32 = embeddings.astype('float32')

    #create index for vectors of size embeddings.
    index = faiss.IndexFlatL2(dimension)

    #add all vectors to the index
    index.add(embeddings_32)
    print(f"Total indexed chunks: {index.ntotal}")

    return index    


#4. save index to folder.
def save_index(index,chunk):
    with open('faiss_index.pkl','wb') as file:
        pickle.dump(faiss.serialize_index(index),file)
        print('File faiss_index.pkl saved successfully.')

    with open('chunks_texts.pkl','wb') as file:
        pickle.dump(chunk,file)
        print('File chunks_texts.pkl saved successfully.')

#5. load index from folder.
def load_index():
    with open('faiss_index.pkl','rb') as file:
        index = faiss.deserialize_index(pickle.load(file))
        print('File faiss_index.pkl loaded successfully.')

    with open('chunks_texts.pkl','rb') as file:
        chunks = pickle.load(file)
        print('File chunks_texts.pkl loaded successfully.')
    return index,chunks


if __name__ == "__main__":
    chunks = load_chunk()
    index = build_faiss(load_embeddings())
    save_index(index, chunks)
    load_index()