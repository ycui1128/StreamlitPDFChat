"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
import re
from io import BytesIO
from typing import List
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from langchain import  OpenAI
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Define a function to parse a PDF file and extract its text content
@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


# Define a function for the embeddings
@st.cache_resource
def test_embed(uploadFileName):
    with st.spinner(f"It's indexing {uploadFileName}.."):
        index = VectorstoreIndexCreator().from_documents(pages)
    st.success("Embeddings done. ", icon="✅")
    return index



# From here down is all the StreamLit UI.
st.set_page_config(page_title="PDF Chatbot", page_icon=":robot:")
st.header("PDF Chatbot")

# Set up the sidebar
st.sidebar.markdown(
    """
    ### Steps:
    1. Upload PDF File
    3. Perform Q&A

    """
)
st.sidebar.divider()

# api = st.sidebar.text_input(
#                 "**Enter OpenAI API Key**",
#                 type="password",
#                 placeholder="sk-",
#                 help="https://platform.openai.com/account/api-keys",
#             )
# st.session_state.apikey = api
# Allow the user to upload a PDF file
with st.expander("Upload PDF file", expanded=False):
    uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])
    if uploaded_file:
        name_of_file = uploaded_file.name
        doc = parse_pdf(uploaded_file)
        pages = text_to_docs(doc)
        if pages:
            # Allow the user to select a page and view its content
            page_sel = st.number_input(
                label="Select Page", min_value=1, max_value=len(pages), step=1
            )
            pages[page_sel - 1]
            index = test_embed(name_of_file)
               
    
if uploaded_file:
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
            st.session_state["past"] = []


    def get_text():
        input_text = st.text_input("Talk with PDF: ", "Hello, how are you?", key="input")
        return input_text

    col1, col2 = st.columns([9, 1])
    with col1:
        user_input = get_text()
    with col2:
        clear = st.button('Clear')
    
    if clear:
        st.session_state["generated"] = []
        st.session_state["past"] = []
        user_input =""
    
    if user_input:
        output = index.query(question=user_input)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(message=st.session_state["generated"][i], avatar_style="icons", key=str(i))
            message(message=st.session_state["past"][i], is_user=True, avatar_style="big-smile", key=str(i) + "_user")
