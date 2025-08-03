import os
import streamlit as st
import pickle
import time
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
import torch

# Configure Streamlit page
st.set_page_config(
    page_title="RockyBot News Research Tool",
    page_icon="📈",
    layout="wide"
)

# Main title
st.title("RockyBot: News Research Tool 📈")
st.markdown("*Powered by HuggingFace - Completely Free & Local*")

# Sidebar for inputs
st.sidebar.title("News Article URLs")

# Check system capabilities
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"🖥️ Running on: {device.upper()}")

# URL inputs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_huggingface.pkl"

main_placeholder = st.empty()


# Load models with caching to avoid reloading
@st.cache_resource
def load_llm():
    """Load HuggingFace language model with caching"""
    try:
        st.info("🔄 Loading language model... (first time may take a few minutes)")

        # Using FLAN-T5 base model - good balance of speed and quality
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
            temperature=0.7,
            do_sample=True,
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        st.success("✅ Language model loaded successfully!")
        return HuggingFacePipeline(pipeline=pipe)

    except Exception as e:
        st.warning(f"⚠️ Error loading main model: {str(e)}")
        st.info("🔄 Falling back to smaller model...")

        # Fallback to smaller, more compatible model
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            max_length=256,
            device=-1  # Force CPU for compatibility
        )

        st.success("✅ Fallback model loaded!")
        return HuggingFacePipeline(pipeline=pipe)


@st.cache_resource
def load_embeddings():
    """Load HuggingFace embeddings with caching"""
    try:
        st.info("🔄 Loading embeddings model...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )

        st.success("✅ Embeddings model loaded!")
        return embeddings

    except Exception as e:
        st.error(f"❌ Error loading embeddings: {str(e)}")
        raise e


# Initialize models
try:
    llm = load_llm()
    embeddings = load_embeddings()
    st.sidebar.success("🎉 All models loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load models: {str(e)}")
    st.stop()

# Process URLs
if process_url_clicked:
    # Check if any URLs are provided
    valid_urls = [url.strip() for url in urls if url.strip()]

    if not valid_urls:
        st.error("❌ Please enter at least one valid URL")
    else:
        try:
            # Load data from URLs
            loader = UnstructuredURLLoader(urls=valid_urls)
            main_placeholder.text("📥 Loading data from URLs...")

            data = loader.load()

            if not data:
                st.error("❌ No data could be loaded from the provided URLs")
                st.info("💡 Make sure the URLs are accessible and contain readable text content")
            else:
                st.success(f"✅ Successfully loaded {len(data)} documents")

                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000,
                    chunk_overlap=200
                )

                main_placeholder.text("✂️ Splitting text into manageable chunks...")
                docs = text_splitter.split_documents(data)
                st.info(f"📄 Created {len(docs)} text chunks for processing")

                # Create embeddings and build FAISS index
                main_placeholder.text("🔍 Building search index... (this may take a moment)")

                vectorstore = FAISS.from_documents(docs, embeddings)

                # Save the vectorstore for future use
                main_placeholder.text("💾 Saving search index...")
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore, f)

                # Show completion message
                main_placeholder.success("🎉 Processing complete! You can now ask questions about the articles.")
                time.sleep(3)
                main_placeholder.empty()

        except Exception as e:
            st.error(f"❌ Error processing URLs: {str(e)}")
            st.info("💡 Please check that the URLs are valid and accessible")

# Query interface
st.markdown("---")
st.markdown("## 💬 Ask Questions About Your Articles")

query = st.text_input(
    "Question:",
    placeholder="What is the main topic discussed in the articles?"
)

if query:
    if os.path.exists(file_path):
        try:
            # Load the saved vectorstore
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Get top 4 most relevant chunks
            )

            # Create the QA chain
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )

            # Generate answer
            with st.spinner("🤔 Analyzing articles and generating answer..."):
                result = chain({"question": query}, return_only_outputs=True)

            # Display the answer
            st.markdown("### 📝 Answer:")
            st.write(result["answer"])

            # Display sources if available
            sources = result.get("sources", "")
            if sources and sources.strip():
                st.markdown("### 🔗 Sources:")
                sources_list = [s.strip() for s in sources.split("\n") if s.strip()]
                for source in sources_list:
                    st.markdown(f"- {source}")

            # Optional: Show source documents for transparency
            if st.checkbox("🔍 Show source excerpts"):
                if "source_documents" in result and result["source_documents"]:
                    st.markdown("### 📄 Relevant Text Excerpts:")
                    for i, doc in enumerate(result["source_documents"][:3]):
                        with st.expander(f"📄 Excerpt {i + 1} from {doc.metadata.get('source', 'Unknown source')}"):
                            # Show first 800 characters of the document
                            content = doc.page_content
                            if len(content) > 800:
                                content = content[:800] + "..."
                            st.write(content)
                else:
                    st.info("No source documents available")

        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")
            st.info("💡 Try rephrasing your question or reprocessing the URLs")
    else:
        st.warning("⚠️ Please process some URLs first using the 'Process URLs' button in the sidebar")

# Sidebar instructions and info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 How to Use:")
st.sidebar.markdown("""
1. **Enter URLs** of news articles in the text boxes above
2. **Click 'Process URLs'** to analyze the content
3. **Wait** for processing to complete (may take 1-2 minutes)
4. **Ask questions** about the articles in the main area
5. **Get answers** with source citations
""")

st.sidebar.markdown("### 💡 Tips for Best Results:")
st.sidebar.markdown("""
- Use clear, specific questions
- Make sure URLs contain readable text
- Wait for processing to fully complete
- First run downloads models (~500MB)
- Subsequent runs are much faster
""")

st.sidebar.markdown("### ⚙️ Technical Details:")
st.sidebar.markdown(f"""
- **Device:** {device.upper()}
- **Language Model:** FLAN-T5 Base
- **Embeddings:** all-MiniLM-L6-v2
- **Vector Store:** FAISS
- **Framework:** LangChain + HuggingFace
""")

