# BG-RAG: Local RAG System with BgGPT

> **Master's Thesis Project**  
> **Faculty of Mathematics and Informatics (FMI), Sofia University "St. Kliment Ohridski"**

This is a **Retrieval-Augmented Generation (RAG)** system (chatbot/dialogue system) designed to work locally on the user's computer. It utilizes **BgGPT** (Gemma-2-9B-IT-v1.0 by INSAIT), which is specifically optimized to improve LLM performance for the Bulgarian language.

## **How it Works**
Based on the available VRAM, a suitable **GGUF quantized version** of the LLM is automatically downloaded upon the first run. The system also downloads an embedding model (**BGE-M3**) and a reranking model (**jina-reranker-v2-base-multilingual**).

## **Key Features**
- **Document Management:** Users can upload their own text documents (PDF, DOC, DOCX, TXT, and JSON) to perform context-aware queries. The system supports both Bulgarian and English.
- **Built-in Database:** Includes a pre-indexed "Main Database" containing articles from **Bulgarian Wikipedia** and crawled news from **Focus News**.
- **Hybrid Search:** Uses a combination of **vector and keyword search**, followed by a reranking stage for maximum accuracy.
- **Advanced Retrieval:** Includes features like **RAG Fusion, HyDE, and Generated Subqueries (Query Decomposition)**. Other features include query translation for cross-lingual search and a "Check relevant resources" mode powered by the LLM.
- **Transparency:** The system displays retrieved resources under the generated answer. A dedicated **Log tab** allows users to inspect the specific retrieved chunks and background processes.
- **User Interface:** The menu is available in Bulgarian and English, including a **Help tab** with instructions and an **Import tab** for document management.

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
