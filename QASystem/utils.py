import os
from dotenv import load_dotenv
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from pinecone import Pinecone


load_dotenv()

# Set environment variables for the core Haystack framework processes
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print("Import Successfully")

def pinecone_config():
    # Configuring the Pinecone connection setup
    document_store = PineconeDocumentStore(
        index ="default",            
        namespace="default",       
        dimension=768,             
        metric="cosine",
        spec={
            "serverless": {
                "region": "us-east-1",  # Matches your aped-4627 AWS cluster
                "cloud": "aws"          
            }
        }
    )
    return document_store

