from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import json
from dotenv import load_dotenv

class LlamaIndexHandler:
    def __init__(self, db_path, persist_dir, embedding_model=None):
        self.db_path = db_path
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model or OpenAIEmbedding()
        self.chroma_collection = self.initialize_chroma_collection()

    def initialize_chroma_collection(self):
        chroma_client = chromadb.PersistentClient(path=self.db_path)
        return chroma_client.get_or_create_collection("quickstart")

    def serialize_results_explained(self, results):
        serialized_results = []
        for result in results:
            base_node = result.node
            document_id = base_node.id_
            document_text = base_node.get_text()
            document_embedding = base_node.get_embedding()[:10] if base_node.embedding else None
            document_metadata = base_node.metadata
            document_score = result.get_score()
            document_relationships = self.get_relationships(base_node.relationships)

            serialized_results.append({
                "id": document_id,
                "score": document_score,
                "content": {
                    "text": document_text,
                    "embedding": document_embedding,
                    "metadata": document_metadata,
                    "relationships": document_relationships
                }
            })
        return serialized_results

    def get_relationships(self, relationships):
        document_relationships = {}
        for relationship, related_node_info in relationships.items():
            relationship_name = relationship_name_mapping.get(relationship, relationship)
            if isinstance(related_node_info, list):
                document_relationships[relationship_name] = [info.dict() for info in related_node_info]
            else:
                document_relationships[relationship_name] = related_node_info.dict()
        return document_relationships

    def get_next_nodes(self, base_node, n):
        next_nodes = [(0, base_node.node.get_text())]
        current_node_id = base_node.node.node_id
        for i in range(1, n + 1):
            current_node = self.chroma_collection.get(current_node_id)
            if not current_node or 'metadatas' not in current_node or not current_node['metadatas']:
                break
            metadata = current_node['metadatas'][0]
            if '_node_content' in metadata:
                metadata['_node_content'] = json.loads(metadata['_node_content'])
            relationships = metadata.get('_node_content', {}).get('relationships', {})
            next_node_id = self.find_next_node_id(relationships)
            if next_node_id is None:
                break
            next_node = self.chroma_collection.get(next_node_id)
            if not next_node or 'documents' not in next_node or not next_node['documents']:
                break
            next_nodes.append((i, next_node['documents'][0]))
            current_node_id = next_node_id
        return next_nodes

    def find_next_node_id(self, relationships):
        for relationship_type, related_node_info in relationships.items():
            if relationship_type == relationship_name_mapping["NEXT"]:
                if isinstance(related_node_info, list):
                    related_node_info = related_node_info[0]
                return related_node_info.get('node_id')
        return None

if __name__ == '__main__':
    load_dotenv()

    # include the line below to load the module...
    # from semantic_retrieve import LlamaIndexHandler
    handler = LlamaIndexHandler(db_path="./data_db", persist_dir="./data_db/persist")
    vector_store = ChromaVectorStore(chroma_collection=handler.chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=handler.persist_dir)
    index = VectorStoreIndex([], storage_context=storage_context, embed_model=OpenAIEmbedding())
    retriever = index.as_retriever()
    results = retriever.retrieve("What are the main storage types along with their parameters?")
    explained_results = handler.serialize_results_explained(results)
    print(json.dumps(explained_results, indent=2))

    next_nodes = handler.get_next_nodes(results[0], 3)
    print(json.dumps(next_nodes, indent=2))
