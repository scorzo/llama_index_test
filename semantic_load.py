from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv
import json

def serialize_results_explained(results):
    """
    Serializes retrieved results into a list of dictionaries with explanations.

    Args:
        results: A list of NodeWithScore objects retrieved from the index.

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

        # Combine information for serialization
        serialized_results.append({
            "id": document_id,
            "score": document_score,
            "content": {
                "text": document_text,
                "embedding": document_embedding,  # Truncate embedding for readability
                "metadata": document_metadata,
            }
        })
    return serialized_results



load_dotenv()

# Read documents from the "data" directory
documents = SimpleDirectoryReader("data").load_data()

# Print the first document's text for inspection
#print(f"First document text: {documents[0].text}")

# Initialize the OpenAI embedding model
embed_model = OpenAIEmbedding()

# Create a semantic splitter using the OpenAI embedding model
semantic_splitter = SemanticSplitterNodeParser(embed_model=embed_model)

# Use the semantic splitter to get nodes from the documents
nodes = semantic_splitter.get_nodes_from_documents(documents=documents)

# Print the number of documents and nodes for reference
print(f"Number of documents: {len(documents)}")
print(f"Number of nodes: {len(nodes)}")

# Initialize the ChromaDB client and create a collection named "quickstart"
chroma_client = chromadb.PersistentClient(path="./data_db")
chroma_collection = chroma_client.get_or_create_collection("quickstart")

# Initialize the vector store using the ChromaVectorStore with the specified collection
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create a storage context for managing data persistence
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Initialize the VectorStoreIndex with the nodes, storage context, and OpenAI embedding model
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=OpenAIEmbedding())

# Persist the index to the specified directory
index.storage_context.persist(persist_dir="./data_db/persist")

if __name__ == '__main__':

### testing ####
# Convert the index into a retriever for querying
# retriever = index.as_retriever()

#results = retriever.retrieve("What are the fundamental database indexing concepts?")
# results = retriever.retrieve("What are the main storage types along with their parameters?")
# explained_results = serialize_results_explained(results)
# print(json.dumps(explained_results, indent=2))

