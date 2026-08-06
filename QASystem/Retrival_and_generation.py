import os
from typing import List
from dotenv import load_dotenv

# Core Haystack imports
from haystack import Pipeline, component
from haystack.utils import Secret
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import ChatMessage

# HuggingFace integration components
from haystack_integrations.components.generators.huggingface_api import HuggingFaceAPIChatGenerator
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder

# Pinecone integration retriever
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

# Your local application utilities
from QASystem.ingestion import ingest
from QASystem.utils import pinecone_config

# Custom component to bridge PromptBuilder (text) to HuggingFaceAPIChatGenerator (ChatMessage)
@component
class TextToChatMessage:
    @component.output_types(messages=List[ChatMessage])
    def run(self, text: str):
        return {"messages": [ChatMessage.from_user(text)]}

# Reads from doc.meta['text'] to align with your Pinecone configuration schema
prompt_template = """Answer the following query based on the provided context. If the context does
                     not include an answer, reply with 'I don't know'.\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.meta['text'] }}
                     {% endfor %}
                     Answer: 
                  """
 
def get_result(query):                  
    query_pipeline = Pipeline()

    # 768-dimension embedding model matching your Pinecone index dimension limits
    query_pipeline.add_component(
        "text_embedder", 
        SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2")
    )
    query_pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=pinecone_config()))
    query_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    query_pipeline.add_component("converter", TextToChatMessage())
    
    query_pipeline.add_component("llm", HuggingFaceAPIChatGenerator(
        api_type="serverless_inference_api",
        api_params={"model": "Qwen/Qwen2.5-7B-Instruct"},  
        token=Secret.from_env_var("HF_TOKEN")
    ))

    # Pipeline Connections
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever.documents", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder.prompt", "converter.text")
    query_pipeline.connect("converter.messages", "llm.messages")

    results = query_pipeline.run(
        {
            "text_embedder": {"text": query},
            "prompt_builder": {"query": query},
        }
    )

    # FIXED: Added [0] to extract from the list, and targeted the correct underlying text property
    chat_message_obj = results['llm']['replies'][0]
    
    # This try/except handles any structural variances gracefully across minor versions
    try:
        return chat_message_obj.text
    except AttributeError:
        return chat_message_obj._content[0].text

if __name__ == '__main__':
    load_dotenv()
    print("All components initialized. Running end-to-end RAG query with text parsing...")
    
    result = get_result("What problem does RAG aim to solve??")
    print("\nResult from LLM:")
    print(result)
