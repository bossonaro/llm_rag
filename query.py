from llama_index.core import VectorStoreIndex, QueryBundle, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from elasticsearch import AsyncElasticsearch
from llama_index.llms.ollama import Ollama
import asyncio
import warnings
from elasticsearch import ElasticsearchWarning

# Ignore Elasticsearch warnings
warnings.filterwarnings("ignore", category=ElasticsearchWarning)

async def main():
    # Local LLM configuration
    local_llm = Ollama(model="llama3.2:1b", base_url="http://192.168.0.190:11434")
    
    # Important: Use the same model and settings as in index.py
    embed_model = OllamaEmbedding(
        model_name="llama3.2:1b",  # Make sure this matches index.py
        base_url="http://192.168.0.190:11434"
    )
    Settings.embed_model = embed_model

    # Elasticsearch configuration
    index_name = "calls_br"
    es_url = "http://192.168.0.40:9200"
    es_client = AsyncElasticsearch(es_url)

    try:
        # Create vector store
        es_vector_store = ElasticsearchStore(
            index_name=index_name,
            es_url=es_url,
            vector_field='conversation_vector',
            text_field='conversation',
            es_client=es_client
        )

        # Create index and query engine
        index = VectorStoreIndex.from_vector_store(es_vector_store)
        query_engine = index.as_query_engine(
            llm=local_llm,
            similarity_top_k=10
        )

        # Perform query
        query = "Dê-me um resumo das problemas relacionados à água"
        response = await query_engine.aquery(query)
        print(response)

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Clean up
        await es_client.close()

if __name__ == "__main__":
    asyncio.run(main())