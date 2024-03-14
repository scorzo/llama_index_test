from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from dotenv import load_dotenv

load_dotenv()

documents = SimpleDirectoryReader("data").load_data()
# print(documents)
# print(documents[0].text)
from llama_index.core.node_parser import SentenceSplitter

text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=10)
nodes = text_splitter.get_nodes_from_documents(documents=documents)
# nodes
# print(len(documents))
# print(len(nodes))
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("tes1233t")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=OpenAIEmbedding()
)
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=OpenAIEmbedding())
retriever = index.as_retriever()
retriever.retrieve("How long does it take to prepare a pizza")




# llm = OpenAI(model="gpt-3.5-turbo")

# query_engine = index.as_query_engine(llm=llm)

# Settings.llm = OpenAI(model="gpt-3.5-turbo")
#
# query_engine.query("How long does it take to prepare a pizza")
# prompts_dict = query_engine.get_prompts()
# print(prompts_dict)
# from llama_index.core import PromptTemplate
#
#
# new_summary_tmpl_str = (
#     "You always say 'Hello my friend' at the beginning of your answer. Below you find data from a database\n"
#     "{context_str}\n"
#     "Take that context and try to answer the question with it."
#     "Query: {query_str}\n"
#     "Answer: "
# )
# new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
# query_engine.update_prompts(
#     {"response_synthesizer:text_qa_template": new_summary_tmpl}
# )
# prompts_dict = query_engine.get_prompts()
# print(prompts_dict)
# query_engine.query("How long does it take to prepare a pizza")