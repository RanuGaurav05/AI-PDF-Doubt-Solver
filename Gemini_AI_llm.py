#All final steps.
#We will load all the files and all the functions then we will move forward with LLM.

from multiprocessing import process

from Day01_pdf_utlis import extract_text,save_file
from Day02_chunking import load_file, split_text, save_chunks_pickle
from Day03_embeddings import save_embeddings 
from Day4_Vector_Store import load_chunk, load_embeddings, build_faiss, save_index, load_index
# from Day05_Semantic_Search import load_index

from sentence_transformers import SentenceTransformer
import faiss
import os

#-------------------------------------------#-------------------------------------#
import google.generativeai as genai
genai.configure(api_key=process.env.get("API_KEY"))
#-------------------------------------------#-------------------------------------#

model = SentenceTransformer('all-MiniLM-L6-v2')

#Semantic Search question with chunks.
def semantic_search_with_score(question, index, chunks):
    question_vector = model.encode([question])
    distance, indices = index.search(question_vector, k=3)
    results = [chunks[i] for i in indices[0]]
    score = distance[0].tolist()

    return results, score

def build_pipeline(pdf_path="sample.pdf"):
    files_needed = [
        "extracted_text.txt",
        "Chunks.pkl",
        "Embeddings.npy",
        "faiss_index.pkl",
        "chunks_texts.pkl",
    ]

    all_exists = all(os.path.exists(i) for i in files_needed)

    if all_exists:
        print("All files exist - loading from disk.")
        return
    
    print('Some files are missing - Creating those files.')
    
#Step 1: Extract text from PDF
    if not os.path.exists("extracted_text.txt"):
        print('extracted_text does not exist creating it.')
        text = extract_text(pdf_path)
        save_file(text)
    else:
        print("extracted_text.txt already exists - skipping extraction.")

#Step 2: Chunking
    if not os.path.exists("Chunks.pkl"):
        print('Chunks.pkl does not exist creating it.')
        text   = load_file('extracted_text.txt')
        chunks = split_text(text)
        save_chunks_pickle(chunks)
    
    else:
        print("Chunks.pkl already exists - skipping chunking.")

#Step 3: Embeddings
    if not os.path.exists("Embeddings.npy"):
        print("Embeddings.npy does not exist creating it.")
        chunks     = load_chunk()
        embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)
        save_embeddings(embeddings)
    else:
        print("Embeddings.npy already exists - skipping embeddings.")

#Step 4: Vector Store
    if not os.path.exists("faiss_index.pkl"):
        print("faiss_index.pkl does not exist creating it.")
        chunks     = load_chunk()
        embeddings = load_embeddings()
        index      = build_faiss(embeddings)
        save_index(index, chunks)
    else:
        print("faiss_index.pkl already exists - skipping vector store creation.")
    
    print('All files are created and pipeline is ready.')

# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch

#-------------------------------------------------------------------------#
import google.generativeai as genai

def load_llm():
    return genai.GenerativeModel("gemini-2.5-flash")

#-------------------------------------------------------------------------#

def get_answer(llm, question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt  = f"""<|system|>
You are a strict PDF assistant. You ONLY answer from the context below.
RULES you must follow:
1. Use ONLY the context provided. No outside knowledge allowed.
2. If the question cannot be answered from the context, you MUST respond with exactly: "This information is not available in the PDF."
3. Do NOT guess. Do NOT use your training knowledge. Do NOT make up facts.
</s>
<|user|>
CONTEXT FROM PDF:
{context}

QUESTION: {question}

Remember: Answer ONLY using the context above. If not found, say "This information is not available in the PDF."
</s>
<|assistant|>
"""
    response = llm.generate_content(prompt)
    return response.text
    
def main():
    #S1: Build index from pdf
    build_pipeline("sample.pdf")

    #S2: Load Save data
    index, chunks = load_index()

#   S3. Load Tinyllama model (LLM)
#   llm = load_tinyllama()

# S3. Load Gemini Model 
    llm = load_llm()

    #S4. Ask question in a loop until user types "Quit"\
    while True:
        question = input("Enter your question or type quit to stop. : ")

        if question.lower() == "quit":
            print("Exiting the program. Goodbye!")
            break

        #S5: Search for relevant chunks
        relevant_chunks, scores = semantic_search_with_score(question, index, chunks)

        Threshold = 1.5
        if scores[0] > Threshold:
            print("Answer not found in PDF.")
            continue

        answer = get_answer(llm, question, relevant_chunks)
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()
    