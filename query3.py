from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from elasticsearch import AsyncElasticsearch
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor
import asyncio
import warnings
from elasticsearch import ElasticsearchWarning

# Ignore Elasticsearch warnings
warnings.filterwarnings("ignore", category=ElasticsearchWarning)

async def main():
    # Local LLM configuration
    local_llm = Ollama(model="llama3.2:1b", base_url="http://192.168.0.190:11434")
    
    # Embedding model configuration
    embed_model = OllamaEmbedding(
        model_name="llama3.2:1b",
        base_url="http://192.168.0.190:11434"
    )
    Settings.embed_model = embed_model

    # Elasticsearch configuration
    index_name = "calls_br"
    es_url = "http://192.168.0.40:9200"
    es_client = AsyncElasticsearch(es_url)

    try:
        # Primeiro, vamos verificar se o índice existe e quantos documentos ele tem
        if await es_client.indices.exists(index=index_name):
            stats = await es_client.indices.stats(index=index_name)
            doc_count = stats['indices'][index_name]['total']['docs']['count']
            print(f"\nÍndice '{index_name}' encontrado com {doc_count} documentos.")
            
            # Vamos buscar alguns documentos para exemplo
            sample_docs = await es_client.search(
                index=index_name,
                body={
                    "size": 2,
                    "_source": ["conversation"]
                }
            )
            
            print("\nExemplo de documentos no índice:")
            for hit in sample_docs['hits']['hits']:
                print(f"\nDocumento: {hit['_source'].get('conversation', 'Sem conteúdo')[:200]}...")
        else:
            print(f"\nÍndice '{index_name}' não encontrado!")
            return

        # Create vector store
        es_vector_store = ElasticsearchStore(
            index_name=index_name,
            es_url=es_url,
            vector_field='conversation_vector',
            text_field='conversation',
            es_client=es_client
        )

        # Create index
        index = VectorStoreIndex.from_vector_store(es_vector_store)
        
        # Define custom template
        custom_template = PromptTemplate("""
        Você é um assistente especializado em análise de análise de relatos.
        Baseie-se apenas no contexto fornecido abaixo para responder à pergunta.
        Se não houver informações suficientes no contexto, diga que não há dados suficientes.

        Contexto: {context}
        
        Pergunta: {question}
        
        Por favor, estruture sua resposta assim:
        1. Resumo dos fatos relevantes encontrados
        2. Resposta específica à pergunta
        3. Exemplos do contexto (se houver)
        
        Resposta:""")

        # Create query engine with custom template and debugging
        query_engine = index.as_query_engine(
            llm=local_llm,
            text_qa_template=custom_template,
            similarity_top_k=5,  # Reduzido para debug
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]  # Reduzido threshold
        )

        # Teste com uma query específica
        query = "Faça um resumo dos problemas relacionados à água"
        print(f"\nExecutando query: {query}")
        
        response = await query_engine.aquery(query)
        print("\nResposta:")
        print(response)

    except Exception as e:
        print(f"\nErro encontrado: {e}")
        import traceback
        print(traceback.format_exc())
    
    finally:
        await es_client.close()

if __name__ == "__main__":
    asyncio.run(main())