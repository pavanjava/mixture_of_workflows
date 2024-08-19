from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core import set_global_handler
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
import os
import qdrant_client
import logging
import phoenix as px
import pickle

px.launch_app()
set_global_handler("arize_phoenix")

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class RAGOperations:

    def __init__(self):
        # load the local data directory and chunk the data for further processing
        self.docs = SimpleDirectoryReader(input_dir="./data", required_exts=[".pdf"]).load_data(show_progress=True)
        self.text_parser = SentenceSplitter(chunk_size=128, chunk_overlap=100)

        # Create a local Qdrant vector store
        logger.info("initializing the vector store related objects")
        self.client = qdrant_client.QdrantClient(url="http://localhost:6333/", api_key="th3s3cr3tk3y")
        self.vector_store = QdrantVectorStore(client=self.client, collection_name="financial_stmts")

        # local vector embeddings model
        logger.info("initializing the OllamaEmbedding")
        embed_model = OllamaEmbedding(model_name='nomic-embed-text:latest', base_url='http://localhost:11434')

        logger.info("initializing the global settings")
        Settings.embed_model = embed_model
        Settings.llm = Ollama(model="llama3.1", base_url='http://localhost:11434', request_timeout=300)
        Settings.transformations = [self.text_parser]
        self.vector_index = None
        self.vector_retriever: VectorIndexRetriever = None
        self.base_nodes_2023 = None
        self._pre_process()

    def _pre_process(self):
        logger.info("enumerating docs")
        node_parser = UnstructuredElementNodeParser()
        pickle_file = "./financial_report.pkl"
        if not os.path.exists(pickle_file):
            raw_nodes_2023 = node_parser.get_nodes_from_documents(self.docs)
            pickle.dump(raw_nodes_2023, open(pickle_file, "wb"))
        else:
            raw_nodes_2023 = pickle.load(open(pickle_file, "rb"))

        self.base_nodes_2023, self.node_mappings_2023 = node_parser.get_base_nodes_and_mappings(
            raw_nodes_2023
        )

        self._index_in_vector_store()

    def _index_in_vector_store(self):
        logger.info("initializing the storage context")
        # Create a local Qdrant vector store
        client = qdrant_client.QdrantClient(url="http://localhost:6333/", api_key="th3s3cr3tk3y")
        vector_store = QdrantVectorStore(client=client, collection_name="financial_stmts")

        # construct top-level vector index + query engine
        if not self.client.collection_exists(collection_name='financial_stmts'):
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.vector_index = VectorStoreIndex(nodes=self.base_nodes_2023, storage_context=storage_context,
                                                 transformations=Settings.transformations,
                                                 embed_model=Settings.embed_model)
        else:
            self.vector_index: VectorStoreIndex = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

    def create_retriever(self) -> VectorIndexRetriever:
        logger.info("initializing the VectorIndexRetriever with top_k as 5")
        vector_retriever = VectorIndexRetriever(index=self.vector_index, similarity_top_k=5)
        return vector_retriever
