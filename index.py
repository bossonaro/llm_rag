import json, os, warnings
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from elasticsearch import AsyncElasticsearch, ElasticsearchWarning
from elasticsearch.helpers import BulkIndexError
import asyncio

# Ignora avisos de segurança do Elasticsearch
warnings.filterwarnings("ignore", category=ElasticsearchWarning)

def get_documents_from_file(file):
    """Reads a json file and returns list of Documents"""
    with open(file=file, mode='rt') as f:
        conversations_dict = json.loads(f.read())

    # Build Document objects using fields of interest.
    documents = [Document(text=item['conversation'],
                          metadata={"conversation_id": item['conversation_id']})
                 for item in conversations_dict]
    return documents

async def main():
    index_name = "calls_br"
    es_url = "http://192.168.0.40:9200"

    # Cria um cliente Elasticsearch assíncrono
    es_client = AsyncElasticsearch(es_url)

    # ElasticsearchStore que gerencia o armazenamento de vetores
    es_vector_store = ElasticsearchStore(index_name=index_name,
                                         es_url=es_url,
                                         vector_field='conversation_vector',
                                         text_field='conversation',
                                         es_client=es_client)  # Passa o cliente aqui

    # Embedding Model to do local embedding using Ollama.
    ollama_embedding = OllamaEmbedding(model_name="llama3.2:1b", base_url="http://192.168.0.190:11434") 

    # LlamaIndex Pipeline configurado para gerenciar a transformação e o armazenamento de embeddings
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=350, chunk_overlap=50),
            ollama_embedding,
        ],
        vector_store=es_vector_store
    )

    # Carrega dados de um arquivo json para uma lista de Documentos LlamaIndex
    documents = get_documents_from_file(file="conversations_br.json")

    try:
        pipeline.run(documents=documents)
    except BulkIndexError as e:
        print(f"BulkIndexError: {e.errors}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Fecha o cliente Elasticsearch
        await es_client.close()
        print(".....Done running pipeline.....\n")

if __name__ == "__main__":
    asyncio.run(main())
