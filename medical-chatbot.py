import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = 'vectorstore/db_faiss'
HF_TOKEN = os.environ.get("HF_TOKEN")


@st.cache_resource
def get_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"max_length": 512}
    )
    return llm


def format_source_documents(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source_path = doc.metadata.get('source', 'Unknown Source')
        book_name = source_path.split('\\')[-1].split('.pdf')[0].replace('_', ' ')
        page_number = doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))

        # Clean and format content
        content = ' '.join(doc.page_content.split()).strip()  # Remove extra whitespace
        if len(content) > 300:  # Increased from 150 to 300 characters
            content = content[:300] + '...'  # Show more context before truncating

        formatted.append(
            f"**📚 Source {i}**\n"
            f"- **Book:** {book_name}\n"
            f"- **Page:** {page_number}\n"
            f"- **Content:** {content}\n"
        )
    return '\n'.join(formatted)



def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input('Pass your prompt here: ')
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        custom_prompt_template = """You are a highly intelligent and helpful AI medical assistant. Your job is to answer user queries based strictly on the context provided below.

        Follow these rules:
        - Use only the information provided in the context to form your answer.
        - If the answer cannot be determined from the context, respond with: "I'm not sure based on the provided information."
        - Do NOT fabricate or infer facts that are not explicitly present.
        - Be clear, direct, and avoid unnecessary small talk.
        - Format your answer in complete, professional sentences.

        Context: {context}

        Question: {question}

        Answer:"""

        huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

        try:
            vectorstore = get_vector_store()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=huggingface_repo_id, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
            )

            response = qa_chain.invoke({'query': prompt})

            # Format the response
            answer = response['result']
            sources = format_source_documents(response["source_documents"])

            formatted_response = (
                f"**Answer:**\n{answer}\n\n"
                f"---\n"
                f"⚠️ *Disclaimer: This response is based on the provided material and should not be taken as medical advice. Please consult a licensed healthcare professional for real-world concerns.*\n\n"
                f"---\n"  # Horizontal line separator
                f"**Supporting Sources:**\n"
                f"{sources}"
            )

            st.chat_message('assistant').markdown(formatted_response)
            st.session_state.messages.append({'role': 'assistant', 'content': formatted_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()