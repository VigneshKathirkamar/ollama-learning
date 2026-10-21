# Streamlit UI on top of cleaner-pdf-rag.py
# Run with: streamlit run pdf-rag-app.py

import importlib

import streamlit as st

# The file name has hyphens, so it can't be imported with a normal `import` statement
rag = importlib.import_module("cleaner-pdf-rag")


@st.cache_resource(show_spinner="Loading the PDF and models. This takes a minute on the first run...")
def get_chain():
    """Build the RAG chain once and reuse it across questions."""
    data = rag.load_pdf(rag.DOC_PATH)
    chunks = rag.split_documents(data)
    vector_db = rag.create_vector_db(chunks)

    llm = rag.load_llm()
    retriever = rag.create_retriever(vector_db, llm)
    return rag.create_chain(retriever, llm)


def main():
    st.set_page_config(page_title="PDF RAG", page_icon="📄")
    st.title("📄 Chat with your PDF")
    st.caption(f"Document: `{rag.DOC_PATH}` · Model: `{rag.MODEL}`")

    chain = get_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the document")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(question)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
