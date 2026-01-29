import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    get_response_synthesizer,
    Settings
)

from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    TokenTextSplitter
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.response_synthesizers import ResponseMode
import shutil



def load_rag_index(rag_config):
    rag_data_dir = rag_config["source_data_dir"]
    rag_index_dir = rag_config["index_dir"]
    embedding_model = rag_config["embedding_model"]
    reset_index = rag_config["reset_index"]
    #if "nvidia" in embedding_model:
    #    Settings.embed_model = NVIDIAEmbedding(model_name=embedding_model)
    #else:
    Settings.embed_model = OpenAIEmbedding(model=embedding_model)
    if reset_index:
        if os.path.exists(rag_index_dir):
            shutil.rmtree(rag_index_dir)
    if not os.path.exists(rag_index_dir):
        documents = SimpleDirectoryReader(rag_data_dir).load_data()
        #node_parser = SemanticSplitterNodeParser(
        #    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model
        #)

        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=rag_index_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=rag_index_dir)
        index = load_index_from_storage(storage_context)
    return index

def init_rag_query_engine(rag_config):
    index = load_rag_index(rag_config)
    retriever = VectorIndexRetriever(index, similarity_top_k=10, max_top_k=10)
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.NO_TEXT
    )
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
    return query_engine


def retrieve_from_index(query_engine, search_query):
    query_result = query_engine.query(search_query)
    return query_result.source_nodes