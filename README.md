# Llama Index Example

This repository contains a script that demonstrates the usage of the Llama Index library for indexing and retrieving documents, as well as navigating through related nodes in a graph-like structure.

## Features

- Initializes a persistent client for ChromaDB.
- Creates a collection in ChromaDB and initializes a vector store using ChromaVectorStore.
- Creates a storage context and initializes a VectorStoreIndex with the OpenAI embedding model.
- Configures a retriever for querying the index.
- Retrieves documents based on a query and serializes the results with explanations.
- Navigates through related nodes using the `get_next_nodes` function.

## Utility Functions

- `serialize_results_explained(results)`: Serializes retrieved results into a list of dictionaries with explanations.
- `get_next_nodes(base_node, n)`: Retrieves the next `n` nodes from a given base node.

## Installation

Ensure you have Python 3.8 or higher installed. Then, install the required dependencies using pip:

```bash
pip install llama-index chromadb python-dotenv
```

## Usage
Load your environment variables from a .env file.

Run the script:
````
python semantic_load.py
python semantic_retrieve.py
````