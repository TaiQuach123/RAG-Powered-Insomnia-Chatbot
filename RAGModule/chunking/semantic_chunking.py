import os
import uuid
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

from RAGModule.chunking import BaseEncoder, JinaV3Encoder
from semantic_chunkers.splitters.regex import RegexSplitter
from semantic_chunkers.schema import Chunk
from semantic_chunkers import StatisticalChunker


def reformat_semantic_chunks_with_overlap(original: Document, semantic_chunks: List[List[Chunk]], overlap=2):
    reformat_chunks = []
    level1 = RegexSplitter()(doc=original.page_content, delimiters=['\n\n'])
    level2 = RegexSplitter()(doc=original.page_content, delimiters=['\n'])
    for i, chunks in enumerate(semantic_chunks[0]):
        if overlap > 0:
            if i == 0:
                prechunks = []
            else:
                prechunks = semantic_chunks[0][i-1].splits
        else:
            prechunks = []

        chunks = chunks.splits
        res = ''

        if len(prechunks) > overlap:
            chunks = prechunks[-overlap:] + chunks
        else:
            chunks = prechunks + chunks

        for chunk in chunks:
            for paragraph in level1:
                if (chunk in paragraph):
                    if (chunk + '\n\n') in (paragraph + '\n\n'):
                        chunk = chunk + '\n\n'
                        break
                    else:
                        for paragraph_level2 in level2:
                            if chunk in paragraph_level2:
                                if (chunk + '\n') in (paragraph_level2 + '\n'):
                                    chunk = chunk + '\n'
                                else:
                                    chunk = chunk + ' '
            res += chunk
        reformat_chunks.append(Document(page_content=res, metadata=original.metadata))
    
    return reformat_chunks


def create_semantic_chunks_from_directory_with_overlap(dir: str, encoder: BaseEncoder, min_split_tokens=150, max_split_tokens = 600, window_size = 15, overlap = 0):
    documents = []
    statistical_chunker = StatisticalChunker(encoder=encoder, min_split_tokens=min_split_tokens, max_split_tokens=max_split_tokens, window_size=window_size)
    
    headers_to_split_on = [('#', 'Title'), ('##', 'High-level Summary')]
    markdown_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

    for path in os.listdir(dir):
        if path.endswith('_high_level.md'):
            continue
        path = os.path.join(dir, path)
        main_content = TextLoader(path).load()[0]

        abstract_path = path[:-3] + '_high_level.md'
        abstract_content = TextLoader(abstract_path).load()[0].page_content

        title_summary = markdown_header_splitter.split_text(abstract_content)
        title, summary = title_summary[0].metadata['Title'], title_summary[0].metadata['High-level Summary']

        main_content.metadata['title'] = title
        main_content.metadata['summary'] = summary

        documents.append(main_content)

    final = []

    for doc in documents:
        semantic_chunks = statistical_chunker(docs = [doc.page_content])
        semantic_chunks = reformat_semantic_chunks_with_overlap(original = doc, semantic_chunks = semantic_chunks, overlap = overlap)

        for i, chunk in enumerate(semantic_chunks):
            chunk.metadata['num_chunks'] = len(semantic_chunks)
            chunk.metadata['doc_id'] = str(uuid.uuid4())
            chunk.metadata['chunk_no'] = i

        final.append(semantic_chunks)
    
    return final