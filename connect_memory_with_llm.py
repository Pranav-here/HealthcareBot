import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import warnings
warnings.filterwarnings("ignore")

# Step 1: Setup LLM(Mistral with HF)
HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"


def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id = huggingface_repo_id,
        task="text-generation",
        temperature = 0.5,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs ={"max_length": 512}
    )
    return llm


# Step 2: Connet LLM with Faiss and Create a chain
DF_FAISS_PATH = "vectorstore/db_faiss"
custom_prompt_template = """
You are a highly intelligent and helpful AI medical assistant. Your job is to answer user queries based strictly on the context provided below.

Follow these rules:
- Use only the information provided in the context to form your answer.
- If the answer cannot be determined from the context, respond with: "I'm not sure based on the provided information."
- Do NOT fabricate or infer facts that are not explicitly present.
- Be clear, direct, and avoid unnecessary small talk.
- Format your answer in complete, professional sentences.

Context: {context}

Question: {question}

Answer:
"""


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(DF_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create a Question-Answer Chain
qa_chain = RetrievalQA.from_chain_type(
    llm= load_llm(huggingface_repo_id),
    chain_type= "stuff",
    retriever = db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents= True,
    chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
)


# INvoke the chain with a single query
user_query = input("Write Query here:")
response = qa_chain.invoke({'query': user_query})
print("Result:", response["result"])
print("Source Documents", response["source_documents"])

