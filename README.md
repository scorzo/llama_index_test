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

## Using the Modules

To use the modules in your own projects, you can import the classes from the modules and initialize them with the necessary parameters.

### LlamaIndexModule

```python
from semantic_load import LlamaIndexModule

# Initialize the module with the data directory, database path, and persistence directory
module = LlamaIndexModule(data_directory="data", db_path="./data_db", persist_dir="./data_db/persist")

# Initialize the index
index = module.initialize_index()

# Use the index for retrieval and other operations
````

### LLamaIndexHandler

```python
from semantic_retrieve import LlamaIndexHandler

# Initialize the handler with the database path and persistence directory
handler = LlamaIndexHandler(db_path="./data_db", persist_dir="./data_db/persist")

# Retrieve and serialize results
results = retriever.retrieve("Your query here")
explained_results = handler.serialize_results_explained(results)
print(json.dumps(explained_results, indent=2))

# Get next nodes from a base node
next_nodes = handler.get_next_nodes(results[0], 3)
print(json.dumps(next_nodes, indent=2))

````
