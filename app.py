import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import streamlit as st

st.set_page_config(layout = "wide")

user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

os.environ["OPENAI_API_KEY"] = user_api_key

data_path = "data/"

files = os.listdir(data_path)

def document_data(query, chat_history):
   all_documents = []
   for file in files:
       file_path = os.path.join(data_path, file)
       loader = TextLoader(file_path)
       documents = loader.load()
       all_documents.append(documents)

   final_document = [item for sublist in all_documents for item in sublist]
   text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 50, separators = ["\n\n","\n"," ",""]) 
   text = text_splitter.split_documents(documents = final_document) 

   # creating embeddings using OPENAI

   embeddings = OpenAIEmbeddings()

   vectorstore = FAISS.from_documents(text, embeddings)
   vectorstore.save_local("vectors")
   print("Embeddings successfully saved in vector Database and saved locally")

   # Loading the saved embeddings 
   loaded_vectors=FAISS.load_local("vectors", embeddings)

   template = """You are an assistant for question-answering tasks on FAQs. 
   Use the following pieces of retrieved context to answer the question. 
   If you don't know the answer, just say that you don't know. The answers should be as detailed as possible.
   Use three sentences maximum and keep the answer concise.
   Question: {question} 
   Context: {context} 
   Answer:
   """

   # Create a PromptTemplate instance with your custom template
   custom_prompt = PromptTemplate(
       template = template,
       input_variables = ["context", "question"],
   )


   # ConversationalRetrievalChain 
   qa = ConversationalRetrievalChain.from_llm(
       llm = OpenAI(), 
       retriever =  loaded_vectors.as_retriever(),
       combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    
   return qa({"question":query, "chat_history":chat_history})

if __name__ == '__main__':

    st.header("FAQs chatbot")
    # ChatInput
    prompt = st.chat_input("Enter your questions here")

    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]
    if "chat_history" not in st.session_state:
       st.session_state["chat_history"]=[]

    if prompt:
       with st.spinner("Generating......"):
           output=document_data(query=prompt, chat_history = st.session_state["chat_history"])

          # Storing the questions, answers and chat history

           st.session_state["chat_answers_history"].append(output['answer'])
           st.session_state["user_prompt_history"].append(prompt)
           st.session_state["chat_history"].append((prompt,output['answer']))

    # Displaying the chat history

    if st.session_state["chat_answers_history"]:
       for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
          message1 = st.chat_message("user")
          message1.write(j)
          message2 = st.chat_message("assistant")
          message2.write(i)
