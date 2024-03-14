from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import json
from dotenv import load_dotenv


class LlamaIndexModule:
    def __init__(self, data_directory, db_path, persist_dir, embedding_model=None):
        self.data_directory = data_directory
        self.db_path = db_path
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model or OpenAIEmbedding()

    def serialize_results_explained(self, results):
        """
        Serializes retrieved results into a list of dictionaries with explanations.

        Args:
            results: A list of NodeWithScore objects retrieved from the index.

        Returns:
            A list of dictionaries, each representing a retrieved document with its score and explained content.
        """
        serialized_results = []
        for result in results:
            base_node = result.node

            document_id = base_node.id_
            document_text = base_node.get_text()
            document_embedding = base_node.get_embedding()[:10] if base_node.embedding else None
            document_metadata = base_node.metadata
            document_score = result.get_score()

            serialized_results.append({
                "id": document_id,
                "score": document_score,
                "content": {
                    "text": document_text,
                    "embedding": document_embedding,
                    "metadata": document_metadata,
                }
            })
        return serialized_results

    def initialize_index(self):
        documents = SimpleDirectoryReader(self.data_directory).load_data()
        semantic_splitter = SemanticSplitterNodeParser(embed_model=self.embedding_model)
        nodes = semantic_splitter.get_nodes_from_documents(documents=documents)

        print(f"Number of documents: {len(documents)}")
        print(f"Number of nodes: {len(nodes)}")

        chroma_client = chromadb.PersistentClient(path=self.db_path)
        chroma_collection = chroma_client.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=self.embedding_model)
        index.storage_context.persist(persist_dir=self.persist_dir)
        return index


if __name__ == '__main__':
    load_dotenv()

    ### example usage
    # from your_module import LlamaIndexModule
    module = LlamaIndexModule(data_directory="data", db_path="./data_db", persist_dir="./data_db/persist")
    index = module.initialize_index()

    # Test retrieval
    # retriever = index.as_retriever()
    # results = retriever.retrieve("What are the fundamental database indexing concepts?")
    # explained_results = module.serialize_results_explained(results)
    # print(json.dumps(explained_results, indent=2))


