import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

# 1. Coleta e extração do texto do site
def coletar_texto(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.find_all(class_="post-content cf entry-content content-spacious")
        return " ".join(t.text.strip() for t in text)
    else:
        print(f"Erro ao acessar a página: {response.status_code}")
        return ""

# 2. Split do texto em chunks
def splitar_texto(texto):
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    return splitter.split_text(texto)

# 3. Criação da base vetorial (FAISS)
def criar_vectorstore(chunks):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_texts(chunks, embedding=embeddings)

# 4. Interface do usuário com Streamlit
def main():
    st.set_page_config(page_title="Chatbot Jurídico")
    st.title("🧑‍⚖️ Chatbot Jurídico (RAG)")

    url = "https://itrabalhistas.com.br/departamento-pessoal/legislacao-trabalhista/"
    texto = coletar_texto(url)
    chunks = splitar_texto(texto)
    vectorstore = criar_vectorstore(chunks)
    retriever = vectorstore.as_retriever()

    llm = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Pergunte sobre leis trabalhistas ou direitos do consumidor:")

    if user_input:
        result = qa({"question": user_input})
        st.session_state.chat_history.append((user_input, result["answer"]))

        for q, a in st.session_state.chat_history:
            st.markdown(f"**Você:** {q}")
            st.markdown(f"**Bot:** {a}")

        if st.button("🔍 Citar fontes da resposta"):
            st.markdown("### Fontes:")
            for doc in result.get("source_documents", []):
                st.markdown(f"- Trecho: `{doc.page_content[:300]}...`")

if __name__ == "__main__":
    main()
