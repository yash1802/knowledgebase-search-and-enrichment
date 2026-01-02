import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from config.settings import UPLOADS_DIR, SUPPORTED_EXTENSIONS
from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.manual_input import ManualInputProcessor
from src.storage.embeddings import EmbeddingGenerator
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_store import SQLiteStore
from src.rag.rag_pipeline import RAGPipeline
from src.llm.llm_client import LLMClient

st.set_page_config(
    page_title="Knowledge Base Chat",
    layout="wide"
)


def initialize_session_state():
    if "sqlite_store" not in st.session_state:
        st.session_state.sqlite_store = SQLiteStore()

    if "llm_client" not in st.session_state:
        st.session_state.llm_client = LLMClient()

    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()

    if "current_chat_id" not in st.session_state:
        default_chat_id = st.session_state.sqlite_store.get_default_chat_id()
        if not default_chat_id:
            default_chat_id = create_chat_with_auto_name("Chat 1")
        st.session_state.current_chat_id = default_chat_id



def get_file_size(file_path):
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return 0


def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_date(timestamp):
    if isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp)
    else:
        dt = timestamp
    return dt.strftime("%Y-%m-%d %H:%M")


def get_file_icon(ext):
    icons = {
        '.pdf': '📕',
        '.txt': '📄',
        '.md': '📝',
        '.docx': '📘'
    }
    return icons.get(ext, '📄')


def format_message_for_llm(message):
    """Format a single message for LLM consumption."""
    role = message.get("role")
    content = message.get("content", "")
    
    # Handle file attachments in user messages
    if role == "user" and message.get("files"):
        files = message.get("files", [])
        if isinstance(files, str):
            files = json.loads(files)
        if files:
            file_list = ", ".join(files)
            content = f"{content}\n[Attached files: {file_list}]"
    
    return {
        "role": role,
        "content": content
    }


def get_chat_history_for_llm(chat_id, max_messages=10):
    """Get formatted chat history for LLM context."""
    sqlite_store = st.session_state.sqlite_store
    messages = sqlite_store.get_chat_messages(chat_id)
    
    # Format last N messages
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    formatted = []
    for msg in recent_messages:
        formatted_msg = format_message_for_llm(msg)
        if formatted_msg:
            formatted.append(formatted_msg)
    
    return formatted


def can_create_chat():
    return st.session_state.sqlite_store.get_chat_count() < 10


def create_chat_with_auto_name(name=None):
    if name is None:
        count = st.session_state.sqlite_store.get_chat_count()
        name = f"Chat {count + 1}"
    return st.session_state.sqlite_store.create_chat(name)


def delete_document(doc_id):
    sqlite_store = SQLiteStore()
    chroma_store = ChromaStore()

    chunks = sqlite_store.get_chunks_by_document_id(doc_id)

    chroma_ids = [chunk['chroma_id'] for chunk in chunks if chunk.get('chroma_id')]
    if chroma_ids:
        chroma_store.delete_chunks_by_ids(chroma_ids)

    sqlite_store.delete_document(doc_id)


def display_message_attachments(files):
    if not files:
        return

    if isinstance(files, str):
        files = json.loads(files)

    st.write("📎 **Attachments:**")
    for file in files:
        ext = Path(file).suffix.lower()
        icon = get_file_icon(ext)
        st.write(f"{icon} {file}")


def display_attached_files_preview(chat_id):
    attachments = st.session_state.get(f"attachments_{chat_id}", [])
    if not attachments:
        return

    st.write("**📎 Attached files:**")
    for i, file in enumerate(attachments):
        col1, col2 = st.columns([4, 1])
        with col1:
            ext = Path(file.name).suffix.lower()
            icon = get_file_icon(ext)
            st.write(f"{icon} {file.name}")
        with col2:
            if st.button("✖", key=f"remove_{i}_{chat_id}"):
                attachments.pop(i)
                st.session_state[f"attachments_{chat_id}"] = attachments
                st.rerun()



def process_files_to_knowledge_base(uploaded_files):
    if not uploaded_files:
        return []

    processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()
    chroma_store = ChromaStore()
    sqlite_store = SQLiteStore()

    processed_files = []

    for uploaded_file in uploaded_files:
        try:
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext not in SUPPORTED_EXTENSIONS:
                continue

            # Save file
            file_path = UPLOADS_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process document
            doc_data = processor.process_document(str(file_path))

            # Add to SQLite
            doc_id = sqlite_store.add_document(
                filename=doc_data["filename"],
                file_path=str(file_path),
                file_type=file_ext
            )

            # Generate embeddings
            embeddings = embedding_generator.generate_embeddings_batch(
                doc_data["chunks"]
            )

            # Prepare metadata
            chroma_metadata = []
            for i, chunk in enumerate(doc_data["chunks"]):
                chroma_id = f"doc_{doc_id}_chunk_{i}"
                chunk_id = sqlite_store.add_chunk(
                    document_id=doc_id,
                    chunk_index=i,
                    text=chunk,
                    chroma_id=chroma_id
                )

                chroma_metadata.append({
                    "document_id": str(doc_id),
                    "chunk_id": str(chunk_id),
                    "chunk_index": str(i),
                    "sqlite_id": str(chunk_id),
                    "filename": doc_data["filename"],
                    "chroma_id": chroma_id
                })

            # Add to Chroma
            chroma_store.add_chunks(
                chunks=doc_data["chunks"],
                embeddings=embeddings,
                metadata=chroma_metadata
            )

            processed_files.append(uploaded_file.name)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    return processed_files



def display_message_metadata(metadata):
    if not metadata:
        return

    confidence = metadata.get("confidence", "low")
    if confidence == "high":
         st.success(f"Confidence: {confidence.upper()}")
    elif confidence == "medium":
         st.warning(f"Confidence: {confidence.upper()}")
    else:
         st.error(f"Confidence: {confidence.upper()}")
    
    sources = metadata.get("sources", [])
    if sources:
        with st.expander("📄 Sources"):
            for source in sources:
                st.write(f"- {source}")

    missing_info = metadata.get("missing_info", [])
    if missing_info:
        st.write("**⚠️ Missing Information:**")
        for info in missing_info:
            st.write(f"- {info}")

    suggestions = metadata.get("enrichment_suggestions", [])
    if suggestions:
        st.write("**Where to find this information:**")
        for suggestion in suggestions:
            if isinstance(suggestion, dict):
                suggestion_text = suggestion.get('description', str(suggestion))
            else:
                suggestion_text = str(suggestion)
            st.write(f"• {suggestion_text}")


def display_chat_messages(chat_id):
    messages = st.session_state.sqlite_store.get_chat_messages(chat_id)

    for message in messages:
        role = message["role"]
        content = message["content"]
        metadata = message.get("metadata")
        files = message.get("files")

        if role == "user":
            with st.chat_message("user"):
                st.write(content)
                if files:
                    display_message_attachments(files)

        elif role == "assistant":
            with st.chat_message("assistant"):
                st.write(content)
                if metadata:
                    display_message_metadata(metadata)



def render_sidebar():
    st.sidebar.title("📚 Chats")

    if st.sidebar.button("➕ New Chat", use_container_width=True):
        if can_create_chat():
            new_chat_id = create_chat_with_auto_name()
            st.session_state.current_chat_id = new_chat_id
            st.rerun()
        else:
            st.sidebar.warning("Maximum 10 chats reached. Delete a chat to create a new one.")

    st.sidebar.divider()

    chats = st.session_state.sqlite_store.get_all_chats()
    current_chat_id = st.session_state.current_chat_id

    if not chats:
        st.sidebar.info("No chats yet. Create a new chat to get started.")

    for chat in chats:
        chat_id = chat["id"]
        chat_name = chat["name"]

        col1, col2 = st.sidebar.columns([3, 1])

        with col1:
            if chat_id == current_chat_id:
                st.button(
                    f"● {chat_name}",
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    disabled=True,
                    type="primary"
                )
            else:
                if st.button(
                    chat_name,
                    key=f"chat_btn_{chat_id}",
                    use_container_width=True
                ):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()

        with col2:
            with st.popover("⋮", use_container_width=True):
                if st.button("🗑️ Delete", key=f"delete_{chat_id}"):
                    if len(chats) > 1:
                        st.session_state.sqlite_store.delete_chat(chat_id)
                        remaining_chats = st.session_state.sqlite_store.get_all_chats()
                        if remaining_chats:
                            st.session_state.current_chat_id = remaining_chats[0]["id"]
                        st.rerun()
                    else:
                        st.sidebar.warning("Cannot delete the last chat.")

                if st.button("🧹 Clear History", key=f"clear_{chat_id}"):
                    st.session_state.sqlite_store.clear_chat_history(chat_id)
                    st.rerun()



def render_file_upload_section():
    st.subheader("📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        key="files_tab_uploader"
    )

    if st.button("Upload Files", type="primary"):
        if uploaded_files:
            with st.spinner("Processing files..."):
                processed = process_files_to_knowledge_base(uploaded_files)
            if processed:
                st.success(f"✅ Uploaded {len(processed)} file(s)")
                st.rerun()
        else:
            st.warning("Please select files to upload")


def render_file_list():
    st.subheader("📄 Uploaded Documents")

    sqlite_store = SQLiteStore()
    documents = sqlite_store.get_all_documents()

    if not documents:
        st.info("No documents uploaded yet.")
        return

    for doc in documents:
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

        with col1:
            ext = Path(doc['filename']).suffix.lower()
            icon = get_file_icon(ext)
            st.write(f"{icon} {doc['filename']}")

        with col2:
            upload_date = doc['upload_timestamp']
            st.write(f"📅 {format_date(upload_date)}")

        with col3:
            file_size = get_file_size(doc['file_path'])
            st.write(f"💾 {format_size(file_size)}")

        with col4:
            if st.button("🗑️", key=f"delete_doc_{doc['id']}"):
                with st.spinner("Deleting..."):
                    delete_document(doc['id'])
                st.success("File deleted")
                st.rerun()


def render_file_management():
    render_file_upload_section()
    st.divider()
    render_file_list()


def handle_user_message(message, files):
    chat_id = st.session_state.current_chat_id
    sqlite_store = st.session_state.sqlite_store

    if files:
        if len(files) > 3:
            st.error("Maximum 3 files allowed")
            return

        with st.spinner("Processing files..."):
            processed_files = process_files_to_knowledge_base(files)

        if processed_files:
            file_names = [f.name for f in files]
            sqlite_store.add_chat_message(
                chat_id,
                "user",
                message if message else "Uploaded files",
                files=file_names
            )

            sqlite_store.add_chat_message(
                chat_id,
                "assistant",
                f"✅ Processed {len(processed_files)} file(s) and added to knowledge base."
            )

            st.session_state[f"attachments_{chat_id}"] = []
            st.rerun()
        else:
            st.error("Failed to process files")
            return

    if message and not files:

        if files:
            intent = "file_enrichment"
        else:
            intent = st.session_state.llm_client.classify_intent(message)
            
            intent_map = {
                "information_request": "regular_query",
                "information_provision": "manual_enrichment",
                "conversational": "conversational"
            }
            intent = intent_map.get(intent, "regular_query")

        if intent == "manual_enrichment":
            sqlite_store.add_chat_message(chat_id, "user", message)

            with st.spinner("Processing information..."):
                processor = ManualInputProcessor()
                result = processor.process_manual_input(
                    message,
                    query_text=None
                )

            if result["success"]:
                confirmation = (
                    f"Thank you! I've added this information to my knowledge base:\n\n"
                    f"_{message}_\n\n"
                    f"This information is now available for future questions."
                )
                sqlite_store.add_chat_message(
                    chat_id,
                    "assistant",
                    confirmation
                )
            else:
                sqlite_store.add_chat_message(
                    chat_id,
                    "assistant",
                    f"Sorry, I couldn't process that information. {result.get('message', '')}"
                )

            st.rerun()

        elif intent == "conversational":
            sqlite_store.add_chat_message(chat_id, "user", message)

            llm_client = st.session_state.llm_client
            response = llm_client.generate_conversational_response(message)

            sqlite_store.add_chat_message(
                chat_id,
                "assistant",
                response
            )

            st.rerun()

        elif intent == "regular_query":
            chat_history = get_chat_history_for_llm(chat_id)

            sqlite_store.add_chat_message(chat_id, "user", message)

            with st.spinner("Thinking..."):
                pipeline = st.session_state.rag_pipeline
                response = pipeline.answer_query(message, chat_history=chat_history)

            sqlite_store.add_chat_message(
                chat_id,
                "assistant",
                response["answer"],
                metadata=response
            )

            st.rerun()


def render_chat_input_with_attachments():
    chat_id = st.session_state.current_chat_id

    if f"attachments_{chat_id}" not in st.session_state:
        st.session_state[f"attachments_{chat_id}"] = []

    if st.session_state[f"attachments_{chat_id}"]:
        display_attached_files_preview(chat_id)

    col1, col2 = st.columns([0.1, 0.9])

    with col1:
        if st.button("📎", key=f"attach_btn_{chat_id}", help="Attach files (max 5)"):
            st.session_state[f"show_uploader_{chat_id}"] = not st.session_state.get(f"show_uploader_{chat_id}", False)


    if st.session_state.get(f"show_uploader_{chat_id}", False):
        files = st.file_uploader(
            "Attach files (max 5)",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            key=f"attach_uploader_{chat_id}"
        )
        if files:
            st.session_state[f"attachments_{chat_id}"] = files[:5]
            st.session_state[f"show_uploader_{chat_id}"] = False
            st.rerun()

    with col2:
        user_input = st.chat_input(
            "Type your message...",
            key=f"chat_input_{chat_id}"
        )

    if user_input or st.session_state[f"attachments_{chat_id}"]:
        if user_input:
            files = st.session_state[f"attachments_{chat_id}"]
            handle_user_message(user_input, files)
        elif st.session_state[f"attachments_{chat_id}"]:
            handle_user_message("", st.session_state[f"attachments_{chat_id}"])


def main():
    initialize_session_state()
    render_sidebar()

    st.title("Knowledge Base Chat")

    tab1, tab2 = st.tabs(["Chat", "Files"])

    with tab1:
        st.markdown("Chat with your knowledge base. Attach files or ask questions.")

        current_chat_id = st.session_state.current_chat_id
        display_chat_messages(current_chat_id)

        render_chat_input_with_attachments()

    with tab2:
        render_file_management()


if __name__ == "__main__":
    main()
