"""
Day - 01
WE will extract all the text from our pdf.
and we will save all the text in a .txt file.
"""

import fitz
import os

def extract_text(pdf): #Pdf open karega aur text extract karega from all the pages 
    if not os.path.exists(pdf):
        print('Folder does not exists!')

    #Step1: Open the PDF file
    doc = fitz.open(pdf)
    all_text = [] #Jiske andar sara pdf ka text aa raha hoga

    #Step2: Loop through every page
    for i in range(len(doc)): #total_page = 5, i->0,1,2,3,4
        page = doc[i] #we are getting each page, 1,2,3,4,5
        
        #Step3: Extract the text from each page
        text = page.get_text()  #Extracting all the text from each pages.
        all_text.append(text)

    #Step4: Join all pages into one big string
    full_text = "\n".join(all_text)
    return full_text


# Step5: Save to extracted_text.txt
def save_file(text):
    with open('extracted_text.txt','w') as file:
        file.write(text)
        print('All the text from each page of PDF is extracted Successfully')

    
#1. extract_text
text = extract_text('sample.pdf')

#2. Save_file
save_file(text)