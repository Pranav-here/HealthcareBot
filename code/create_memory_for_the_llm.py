from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

'''
Initial process
    Load raw medical PDFs
    Chunk Documents
    Create Vector Embeddings
    Store Embeddings in FAISS
'''

# Step 1: Load the raw medical pdf files

path = "data/"


# Function to load PDF files from data directory
def load_pdf_files(data):
    loader = DirectoryLoader(data,  # all files in a directory
                             glob='*pdf',  # Read all pdf
                             loader_cls=PyPDFLoader)  # use PyPDFLoader to load
                                                    # each pdf file, PyPDFLoader
                                                    # loads one Document per page

    doc = loader.load()  # load all of the matching pdf's into a list
    return doc


docs = load_pdf_files(data=path)
# print("Number of pages in the PDF:", len(docs))  # Should show the number of pages


# Step 2: Create Chunks
def create_chunks(extracted_data):  # pass the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,  # how many letters is each chunk
                                                   chunk_overlap=50)  # overlapping letters (increases the context window)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


text_chunks = create_chunks(extracted_data=docs)
# print("Length of text chunks:", len(text_chunks))


# Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    return embedding_model


embedding_model = get_embedding_model()


# Step 4: Store the embdeeing in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)