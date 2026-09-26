import os

import streamlit as st
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# הגדרת מפתח ה-API של Google
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# פונקציה שטוענת את מאגר המידע פעם אחת ושומרת בזיכרון (Cache)
@st.cache_resource
def load_knowledge_base():
    loader = TextLoader("flight_policies.txt", encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # מעבר למודל ההטמעה של גוגל
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()


retriever = load_knowledge_base()

# הגדרת המודל והשרשרת - שימוש ב-Gemini של גוגל
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
system_prompt = (
    "You are a professional and courteous customer service representative for a flight app."
    "Answer solely based on the attached information. If the information is not available, state that you do not know."
    "\n\n{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# ----------------- ממשק המשתמש (Streamlit UI) ----------------- #
st.title("Personal Assistant – Flights ✈️")
st.write("Hi! I'm Sun's smart assistant. How can I help?")

# שורת הקלט של המשתמש

st.markdown("### 💡 Example questions you can try:")
st.markdown("- How can I cancel or update an existing order?")
st.markdown("- What is the baggage allowance for carry-on luggage?")
st.markdown("- How much does it cost to add an extra checked bag?")
st.markdown("- Can I change my flight date and is there a fee?")
st.markdown("- How can I cancel or update an existing order?")

st.write("---")

user_question = st.text_input("Ask something:")

if user_question:
    # הצגת הודעת טעינה בזמן שהמודל חושב
    with st.spinner("Looking for an answer..."):
        response = rag_chain.invoke({"input": user_question})

        # הצגת התשובה על המסך
        st.success("answer:")
        st.write(response["answer"])
