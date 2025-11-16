import requests
from bs4 import BeautifulSoup
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import boto3
import json
from langchain.retrievers import MergerRetriever
from langchain.memory import ConversationBufferMemory
import streamlit as st

bedrock = boto3.client('bedrock-runtime', 'us-east-1')


def researchAgent(query, llm):
    tools = load_tools(["ddg-search", "wikipedia"], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt)
    webContext = agent_executor.invoke({ "input": query })
    return webContext['output']

# 1.0 Coleta e extração do texto do site(leis trabalhistas)
def carregar_dados_leis_trabalhistas(urls,chunks):
    textos = []
    for url in urls:
        response = requests.get(urls[0]) 
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
        if "itrabalhistas.com.br" in url:
            text = soup.find_all(class_="post-content cf entry-content content-spacious")
            return " ".join(t.text.strip() for t in text)
        if "planalto.gov.br" in url:
            body = soup.find("body")
            if body:
                text = body.get_text(separator="\n", strip=True)
                return text
        else:
            print(f"Erro ao acessar a página: {response.status_code}")
    textos.append(text,body)
    texto_final = " ".join(textos)
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    splitter.split_text(texto_final)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_texts(chunks, embedding=embeddings)

# 1.1 Coleta e extração do texto do site(direitos eleitorais)
def carregar_dados_direitos_eleitorais(urls1, chunks):
    textos1 = []
    for url in urls1:
       response = requests.get(urls1[0])
       if response.status_code == 200:
           soup = BeautifulSoup(response.content, "html.parser")
           if "https://www.planalto.gov.br/ccivil_03/Leis/L9504" in url:
               body = soup.body("body")
               if body:
                   text = body.get_text(separator="\n", strip=True)
                   return text
           elif "https://www.planalto.gov.br/ccivil_03/Leis/L4737" in url:
               body = soup.find("body")
               if body:
                   text = body.get_text(separator="\n", strip=True)
                   return text
           if "https://www.planalto.gov.br/ccivil_03/Leis/L9096" in url:
               body = soup.find("body")
               if body:
                   text = body.get_text(separator="\n", strip=True)
                   return text
       else:
           print(f"Erro ao acessar a página: {response.status_code}")
    textos1.append(text,body)
    texto_final1 = " ".join(textos1)
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    splitter.split_text(texto_final1)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_texts(chunks, embedding=embeddings)


# Colocar o retriever_unificado aqui
def getRelevantDocs(user_input):
  retriever = carregar_dados_leis_trabalhistas(), carregar_dados_direitos_eleitorais()
  relevant_documents = retriever.invoke(user_input)
  print(relevant_documents)
  return relevant_documents



# 4. Interface do usuário com Streamlit
def main():
    st.set_page_config(page_title="Chatbot Jurídico")
    st.title("🧑‍⚖️ Chatbot Jurídico (RAG)")

    urls = ["https://itrabalhistas.com.br/departamento-pessoal/legislacao-trabalhista/",
           "https://www.planalto.gov.br/ccivil_03/decreto-lei/Del5452compilado.htm"
           ]
    
    urls1 = ["https://www.planalto.gov.br/ccivil_03/Leis/L9504compilado.htm",
            "https://www.planalto.gov.br/ccivil_03/Leis/L4737compilado.htm",
            "https://www.planalto.gov.br/ccivil_03/Leis/L9096compilado.htm"
            ]
    vectorstore = carregar_dados_leis_trabalhistas(chunks)
    vectorstore1 = carregar_dados_direitos_eleitorais(chunks)


    retriever = vectorstore.as_retriever()
    retriever1 = vectorstore1.as_retriever()
    retrievers = [retriever, retriever1]

    retriever_unificado = MergerRetriever(retrievers=retrievers)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever_unificado, memory=memory)

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
    user_input = st.text_input("Pergunte sobre leis trabalhistas ou direitos do consumidor:")

    if user_input:
        body = {
            "prompt": user_input,
            "temperature": 0.7,
            "top_k": 250,
            "max_tokens_to_sample": 150
        }

        llm = bedrock.invoke_model(
            modelId = "mistral.mistral-7b-instruct-v0:2",
            json = json.dumps(body),
            contentType = "application/json",
            accept = "application/json"
        )

        result = qa({"question": user_input})
        st.session_state.chat_history.append((user_input, result["answer"]))

        for q, a in st.session_state.chat_history:
            st.markdown(f"**Você:** {q}")
            st.markdown(f"**Bot:** {a}")

        if st.button("🔍 Citar fontes da resposta"):
            st.markdown("### Fontes:")
            for doc in result.get("source_documents", []):
                st.markdown(f"- Trecho: `{doc.page_content[:300]}...`")
