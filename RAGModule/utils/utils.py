import numpy as np
from itertools import islice
from typing import List
from langchain_core.documents import Document

def convert_defaultdict(input: dict):
    new_dict = {"values": [], "indices": []}
    for key, value in input.items():
        new_dict['values'].append(value)
        new_dict['indices'].append(int(key))
    
    new_dict['values'] = np.array(new_dict['values'])
    new_dict['indices'] = np.array(new_dict['indices'])
    return new_dict

#def batch_iterator(data, batch_size: int):
#    iterator = iter(data)
#    for first in iterator:
#        yield [first] + list(islice(iterator, batch_size - 1))



def format_chunks(chunks):
    res = ''
    for i, chunk in enumerate(chunks):
        res += f"---Begin Chunk---\nSource: {chunk.metadata['source']}\ndoc_id:{chunk.metadata['doc_id']}\n{chunk.metadata['title']}\n\nContent:\n{chunk.page_content}---End Chunk---\n\n"
    return res