# BG RAG: Local RAG System with BgGPT

> **Master's Thesis**
> **Author: Svetlomir Stankov**
> **Sofia University "St. Kliment Ohridski", Faculty of Mathematics and Informatics (FMI)**

This is a **Retrieval-Augmented Generation (RAG)** system (chatbot/dialogue system) designed to work locally on the user's computer. It utilizes **BgGPT** (Gemma-2-9B-IT-v1.0 by INSAIT), which is specifically optimized to improve LLM performance for the Bulgarian language.

### 📺 **System Demo Video (in Bulgarian)**

## **Key Features**
- **Document Management:** Users can upload their own text documents (PDF, DOC, DOCX, TXT, and JSON) to perform context-aware queries. The system supports both Bulgarian and English.
- **Built-in Database:** Includes a pre-indexed "Main Database" containing articles from **Bulgarian Wikipedia** and crawled news from **Focus News**.
- **Hybrid Search:** Uses a combination of **vector and keyword search**, followed by a reranking stage for maximum accuracy.
- **Contextual Query Modification:** Enhances multi-turn dialogue by modifying follow-up queries based on the chat history, ensuring the retrieval remains relevant to the ongoing conversation.
- **Advanced Retrieval Techniques:** Users can choose from several advanced methods to enhance search accuracy: **RAG Fusion**, **HyDE**, and **Generated Subqueries (Query Decomposition)**.
- **Sentence-based Chunking:** Documents are processed using a sentence-based chunking strategy (e.g., **20 sentences per chunk with a 2-sentence overlap**) to maintain context, while smaller documents are preserved in their entirety.
- **LLM-Powered Refinement:** Features a **"Check relevant resources"** mode, where the LLM acts as a final validator to filter and refine the retrieved document chunks before they are included in the context.
- **Multilingual Support:** Includes **Query Translation** for seamless cross-lingual search across both Bulgarian and English document collections.
- **Customizable Context:** Users can adjust the **number of retrieved chunks (Top-N)** used as context for the model to optimize accuracy and performance.
- **Transparency:** The system displays retrieved resources under the generated answer. A dedicated **Log tab** allows users to inspect the specific retrieved chunks and background processes.
- **Interactive User Interface:** A comprehensive menu system designed for ease of use, featuring several specialized modules:
    - **Options Tab:** Advanced configuration for retrieval techniques (RAG Fusion, HyDE, etc.), model parameters, and language selection.
    - **Import Tab:** Full document management for uploading personal files (PDF, DOCX, etc.) or clearing the existing local database.
    - **Save Tab:** Functionality to export the entire conversation history into a structured text file for future reference.
    - **Log Tab:** Real-time visibility into background operations, retrieved document chunks, and internal system processes.
    - **Help Tab:** Detailed user documentation and step-by-step instructions on how to operate the system effectively.
      
## **Model Deployment & Adaptation**
Based on the available VRAM, a suitable **GGUF quantized version** of the LLM is automatically downloaded upon the first run. The system also downloads an embedding model (**BGE-M3**) and a reranking model (**jina-reranker-v2-base-multilingual**).

## **System Requirements**
- **OS:** Windows 10
- **GPU:** NVIDIA GPU (Recommended: **12 GB VRAM**; Minimum: **6-8 GB VRAM**)
- **RAM:** At least **16 GB**
- **Internet:** Required only for the initial download of the models

## **Optimization Note**
If the system runs slowly or RAM is limited, users can disable the "Main Database" by deleting the `db_chunks_and_embeddings.pkl` and `faiss_vector_db.index` files in the `_internal` subfolder. The system will continue to work perfectly with user-imported documents.

## **Installation**
1. Download all required installation files from the **latest release**.
2. Run the installer: `BG_RAG_1.0.0_windows_setup.exe`.
3. Ensure an internet connection is active for the first run to download the necessary models.
