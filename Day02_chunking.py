"""
Day-02
We are splitting our complete pdf text to a some small pieces.

and these small pieces are known as chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle

#Task 1: Load the extracted.txt file in read mode.
def load_file():
    with open('extracted_text.txt') as file:
        text = file.read()
        return text

#Task 2: Chunking the raw text from extracted.txt file. 
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50)

    chunks = splitter.split_text(text)
    return chunks

# Print top 3 chunks
def view_chunks(chunks):
    for index , value in enumerate(chunks):
        print(f'Chunk {index} : {value}')
        print('-----------------------------')

#Saving all the chunks in a .txt file named as chunks.txt
def save_chunks(chunks):
    with open('Chunks.txt','w') as file:
        for index, value in enumerate(chunks):
            file.write(f'Chunk {index} : {value}\n')
            file.write('-----------------------------\n')

#Task 3: Save the chunks in .pkl format
def save_chunks_pickle(chunks):
    with open('Chunks.pkl','wb') as file: #wb -> write in bytes
        pickle.dump(chunks, file) #dump-> Save the chunks in the file.
    print('File saved successfully in .pkl format')

text = load_file()
chunks = split_text(text)
save_chunks_pickle(chunks)

text = load_file()

chunks = split_text(text)

save_chunks(chunks)

save_chunks_pickle(chunks)