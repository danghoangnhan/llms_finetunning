from langchain import FAISS
from langchain.document_loaders import  PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from rag.helper_functions import replace_t_with_space, retrieve_context_per_question


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """
    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

if __name__ == "__main__":
    path = ''
    chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)
    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})
    test_query = "What is the main cause of climate change?"
    context = retrieve_context_per_question(test_query, chunks_query_retriever)
    print(context)