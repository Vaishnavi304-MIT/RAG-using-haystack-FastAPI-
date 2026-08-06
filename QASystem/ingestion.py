import os
from pathlib import Path
from dotenv import load_dotenv

# Core Haystack imports
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import PyPDFToDocument

# Sentence Transformers embedders integration namespace
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack.dataclasses import Document

# Pinecone integration import
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

# Your local configuration utility
from QASystem.utils import pinecone_config

# Load your environment variables (API keys, etc.)
load_dotenv()

def ingest(document_store):
    indexing = Pipeline()

    indexing.add_component("converter", PyPDFToDocument())
    indexing.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=2))
    indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
   
    indexing.add_component("writer", DocumentWriter(document_store))

    # CORRECTED CONNECTIONS FOR HAYSTACK 2.x COMPONENTS
    # PyPDFToDocument outputs a single key named 'documents'
    indexing.connect("converter", "splitter")
    
    # DocumentSplitter outputs a dictionary containing a 'documents' list
    indexing.connect("splitter", "embedder")
    
    # SentenceTransformersDocumentEmbedder processes chunks and outputs 'documents' with vector arrays
    indexing.connect("embedder", "writer")


    indexing.run({"converter": {"sources": [Path("C:\\Users\\Nikita\\EndtoEnd_haystack_fastAPI\\data\\RAG_for_Knowledge_nlp_tasks.pdf")]}})
	
if __name__ == "__main__":
    # Ensure this function correctly initialises and returns a PineconeDocumentStore instance
    document_store = pinecone_config()
    ingest(document_store)
	
