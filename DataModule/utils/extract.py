import os
from llama_parse import LlamaParse

import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv()

def extract_from_pdf(pdf_path: str, content_prompt: str, high_level_prompt: str, save_dir: str ='Data/extracted'):
    #Get the first page from PDF file (extract high level information from this page such as title, abstract)
    assert pdf_path.endswith('.pdf'), "Requires PDF file as input"

    filename = os.path.join(pdf_path.split('/')[-2], pdf_path.split('/')[-1][:-4])
    filename = os.path.join(save_dir, filename)

    #parser for extract content from PDF file
    content_parser = LlamaParse(
        result_type="markdown",
        parsing_instruction=content_prompt,

    )
    #parser for extract high level information from PDF file
    high_level_parser = LlamaParse(
        result_type="markdown",
        parsing_instruction=high_level_prompt,
        target_pages="0",
    )

    high_level_content = high_level_parser.load_data(pdf_path)[0].text
    with open(f'{filename}_high_level.md', 'w') as f:
        f.write(high_level_content)

    documents = content_parser.load_data(pdf_path)
    result = ''

    for i, doc in enumerate(documents):
        result += doc.text + ' '
    
    with open(f'{filename}.md', 'w') as f:
        f.write(result)