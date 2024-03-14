from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import json
from dotenv import load_dotenv

relationship_name_mapping = {
    "CHILD": '5',
    "NEXT": '3',
    "PARENT": '4',
    "PREVIOUS": '2',
    "SOURCE": '1'
}


def serialize_results_explained(results):
    """
    Serializes retrieved results into a list of dictionaries with explanations.

    Args:
        results: A list of NodeWithScore objects retrieved from the index.
        get_node: A function that retrieves a node from the index or storage using its ID.

    Returns:
        A list of dictionaries, each representing a retrieved document with its score and explained content.
    """
    serialized_results = []
    for result in results:
        base_node = result.node  # Access the BaseNode object

        # Extract relevant information from BaseNode
        document_id = base_node.id_
        document_text = base_node.get_text()
        # Check if the embedding is set and handle accordingly
        if base_node.embedding:
            document_embedding = base_node.get_embedding()[:10]  # Truncate embedding for readability
        else:
            document_embedding = None  # Handle cases where embedding is not set

        document_metadata = base_node.metadata  # Consider selecting specific keys
        document_score = result.get_score()  # Get the score

        # Extract relationship details
        document_relationships = {}
        for relationship, related_node_info in base_node.relationships.items():
            # Get the relationship name from the mapping
            relationship_name = relationship_name_mapping.get(relationship, relationship)

            # Convert related_node_info to a dictionary if it's a single object, or a list of dictionaries if it's a list
            if isinstance(related_node_info, list):
                document_relationships[relationship_name] = [info.dict() for info in related_node_info]
            else:
                document_relationships[relationship_name] = related_node_info.dict()

        # Combine information for serialization
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

def get_next_nodes(base_node, n):
    next_nodes = [(0, base_node.node.get_text())]  # Include the text of the base node
    current_node = base_node.node
    current_node_id = current_node.node_id
    for i in range(1, n + 1):  # Start from 1 since the base node is already included
        current_node = chroma_collection.get(current_node_id)
        if current_node is None:
            print(f'Error: Node with ID {current_node_id} not found in chroma_collection.')
            break

        next_node_id = None
        if 'metadatas' not in current_node or not current_node['metadatas']:
            print(f'Error: "metadatas" field is missing or empty in node with ID {current_node_id}.')
            break

        metadata = current_node['metadatas'][0]
        if '_node_content' in metadata:
            metadata['_node_content'] = json.loads(metadata['_node_content'])
        relationships = metadata.get('_node_content', {}).get('relationships', {})
        if not relationships:
            print(f'Warning: "relationships" field is empty for node with ID {current_node_id}.')


        for relationship_type, related_node_info in relationships.items():
            if relationship_type != relationship_name_mapping["NEXT"]:
                continue  # Skip non-"NEXT" relationships

            if isinstance(related_node_info, list):
                # Assuming the first related node is the "next" node if there are multiple
                related_node_info = related_node_info[0]

            if 'node_id' not in related_node_info:
                print(f'Error: "node_id" is missing in relationships for node with ID {current_node_id}.')
                continue  # Skip this relationship

            print(f'Found next node after {i} loops out of {n + 1}')
            next_node_id = related_node_info['node_id']
            break

        if next_node_id is None:
            print(f'Exiting next node after {i} loops out of {n + 1}')
            break  # No more "next" nodes

        next_node = chroma_collection.get(next_node_id)
        if next_node is None:
            print(f'Error: Next node with ID {next_node_id} not found in chroma_collection.')
            break

        if 'documents' not in next_node or not next_node['documents']:
            print(f'Error: "documents" field is missing or empty in next node with ID {next_node_id}.')
            break

        next_nodes.append((i, next_node['documents'][0]))  # Include the index and text of the next node
        current_node_id = next_node_id

    return next_nodes

# Load environment variables from a .env file
load_dotenv()



if __name__ == '__main__':

# Initialize the persistent client for ChromaDB at the specified path
chroma_client = chromadb.PersistentClient(path="./data_db")

# Get or create a collection named "quickstart" in ChromaDB
chroma_collection = chroma_client.get_or_create_collection("quickstart")

# Initialize the vector store using the ChromaVectorStore with the specified collection
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create a storage context
# LlamaIndex supports dozens of vector stores. You can specify which one to use by passing in a StorageContext, on which in turn you specify the vector_store argument
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./data_db/persist")

# Initialize the VectorStoreIndex with an empty dataset, the storage context, and the OpenAI embedding model
index = VectorStoreIndex([], storage_context=storage_context, embed_model=OpenAIEmbedding())

# configure retriever
retriever = index.as_retriever()

#results = retriever.retrieve("What are the fundamental database indexing concepts?")
results = retriever.retrieve("What are the main storage types along with their parameters?")
explained_results = serialize_results_explained(results)
print(json.dumps(explained_results, indent=2))


next_nodes = get_next_nodes(results[0], 3)
print(json.dumps(next_nodes, indent=2))

### debug
# node_id = "4f38c0a6-5b54-4dd0-92ac-55c53c61d47e"  # replace with your node id
# node = chroma_collection.get(node_id)
# for metadata in node['metadatas']:
#     if '_node_content' in metadata:
#         metadata['_node_content'] = json.loads(metadata['_node_content'])
# print(json.dumps(node, indent=4))