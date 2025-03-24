import logging
import os
import sys
import warnings
import comtypes.client
import fitz
import pandas as pd
from datasets import load_dataset
import spacy.lang.bg.stop_words
from llama_cpp.llama_cpp import load_shared_library

import pathlib
import shutil
from pathlib import Path
import json
import torch
import pickle

import numpy as np
import psutil
import re
import spacy
from spacy.lang.bg import Bulgarian
from docx2pdf import convert
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

import webbrowser
from huggingface_hub import hf_hub_download

from llama_cpp import Llama
import faiss
from simplemma import text_lemmatizer, in_target_language
import bm25s
from faiss import write_index, read_index
from nuclia_eval import REMi

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '0'
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore", message="flash_attn is not installed. Using PyTorch native attention implementation.")


class Rag:
    def __init__(self):
        # change self.evaluation_mode to True if you want your conversations to be saved in
        # 'evaluation_df.pkl' file so that you can evaluate later the RAG performance (False if you don't want this)
        self.evaluation_mode = False

        options_config_json = open('options_config.json', 'r')
        options_config = json.load(options_config_json)
        options_config_json.close()
        self.menu_language = options_config["menu_language_var"]

        self.nlp_bg = Bulgarian()
        self.nlp_bg.add_pipe("sentencizer")

        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_en.add_pipe("sentencizer")

        self.nlp = self.nlp_bg

        self.embedding_model = None
        self.reranking_model = None

        self.stopwords = spacy.lang.bg.stop_words.STOP_WORDS
        self.min_token_length = 20
        self.extra_long_sentence_splitter_size = 133
        self.chunk_sentences_size = 20
        self.chunk_sentences_overlap = 2

        self.use_default_db = True
        self.use_uploaded_db = False

        self.bg_wiki_pickle_filename = 'bg_wiki_df.pkl'
        self.bg_wiki_chunks_df_filtered_pickle_filename = "bg_wiki_chunks_filtered_df.pkl"
        self.bg_wiki_chunks_and_embeddings_dict_pickle_filename = "bg_wiki_chunks_and_embeddings_dict.pkl"
        self.bg_wiki_chunks_and_embeddings_csv_filename = 'bg_wiki_chunks_and_embeddings.csv'
        self.uploaded_files_chunk_dict_filename = "uploaded_files_chunk_dict.pkl"
        self.uploaded_files_faiss_vector_db_filename = "uploaded_files_faiss_vector_db.index"
        self.focus_news_chunk_dict_filename = "focus_news_chunk_dict.pkl"
        self.chunk_dict_filename = "db_chunks_and_embeddings.pkl"
        self.faiss_store_filename = "faiss_vector_db.index"
        self.bg_wiki_faiss_store_filename = "bg_wiki_faiss_vector_db.index"
        self.focus_news_faiss_store_filename = "focus_news_faiss_vector_db.index"
        self.db_bm25s_store_filename = "db_bm25s_chunks_index"
        self.uploaded_files_bm25s_store_filename = "uploaded_files_bm25s_chunks_index"

        self.bg_wiki_df = None
        self.bg_wiki_chunks = []
        self.bg_wiki_chunks_df = None
        self.bg_wiki_chunks_df_filtered = None
        self.bg_wiki_chunks_and_embeddings_df = None
        self.bg_wiki_chunks_and_embeddings_dict = None
        self.bg_wiki_chunks_and_embeddings = None
        self.bg_wiki_embeddings = None
        self.bg_wiki_faiss_index = None
        self.focus_news_chunk_dict = None
        self.focus_news_faiss_index = None
        self.focus_news_embeddings = None

        self.user_uploaded_chunks_and_embeddings = None
        self.user_uploaded_embeddings = None
        self.user_uploaded_faiss_index = None

        self.db_chunks_and_embeddings = None
        self.index_faiss_db = None
        self.db_embeddings = None

        self.chunks_and_embeddings = None
        self.faiss_index = None
        self.embeddings = None

        self.evaluator = None
        self.evaluation_df = None
        self.evaluation_df_store_pickle_filename = "evaluation_df.pkl"

        self.history_str = ""
        self.chat_history = []
        self.recent_chat_history = []
        self.summarised_chat_history = ""
        self.relevant_resources_history = []
        self.unsaved_chat_history = []
        self.unsaved_relevant_resources_history = []

        self.model_gguf_name = "INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0-GGUF"
        self.model_q4ks_file = "BgGPT-Gemma-2-9B-IT-v1.0.Q4_K_S.gguf"
        self.model_q4km_file = "BgGPT-Gemma-2-9B-IT-v1.0.Q4_K_M.gguf"
        self.model_q5ks_file = "BgGPT-Gemma-2-9B-IT-v1.0.Q5_K_S.gguf"
        self.model_q5km_file = "BgGPT-Gemma-2-9B-IT-v1.0.Q5_K_M.gguf"
        self.model_q6k_file = "BgGPT-Gemma-2-9B-IT-v1.0.Q6_K.gguf"
        self.model_q8_file = "BgGPT-Gemma-2-9B-IT-v1.0.Q8_0.gguf"
        self.model_f16_file = "BgGPT-Gemma-2-9B-IT-v1.0.F16.gguf"

        self.instruct_7b_model_gguf_name = "INSAIT-Institute/BgGPT-7B-Instruct-v0.2-GGUF"
        self.instruct_7b_model_q6k_file = "BgGPT-7B-Instruct-v0.2.Q6_K.gguf"
        self.instruct_7b_model_q8_file = "BgGPT-7B-Instruct-v0.2.Q8_0.gguf"
        self.instruct_7b_model_q4km_file = "BgGPT-7B-Instruct-v0.2.Q4_K_M.gguf"
        self.instruct_7b_model_q4ks_file = "BgGPT-7B-Instruct-v0.2.Q4_K_S.gguf"
        self.instruct_7b_model_q5km_file = "BgGPT-7B-Instruct-v0.2.Q5_K_M.gguf"
        self.instruct_7b_model_q5ks_file = "BgGPT-7B-Instruct-v0.2.Q5_K_S.gguf"
        self.instruct_7b_model_f16_file = "BgGPT-7B-Instruct-v0.2.F16.gguf"

        self.model_file = None
        self.llm = None

        if torch.cuda.is_available():
            self.device = 'cuda'
            self.gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
            self.gpu_memory_gb = round(self.gpu_memory_bytes / (2 ** 30))
            print(f"GPU VRAM available: {self.gpu_memory_gb} GB")
            if self.menu_language == "bulgarian":
                logging.info(f"GPU VRAM наличен: {self.gpu_memory_gb} GB")
            else:
                logging.info(f"GPU VRAM available: {self.gpu_memory_gb} GB")
            if self.gpu_memory_gb <= 7:
                self.model_file = self.model_q4ks_file
            elif self.gpu_memory_gb <= 8:
                self.model_file = self.model_q4km_file
            elif self.gpu_memory_gb <= 9:
                self.model_file = self.model_q5ks_file
            elif self.gpu_memory_gb <= 10:
                self.model_file = self.model_q5km_file
            elif self.gpu_memory_gb <= 13:
                self.model_file = self.model_q6k_file
            elif self.gpu_memory_gb <= 20:
                self.model_file = self.model_q8_file
            else:
                self.model_file = self.model_f16_file

            if self.menu_language == "bulgarian":
                logging.info("GPU (Cuda) беше открит. Моделът следователно ще бъде зареден на GPU за по-бърза скорост.")
            else:
                logging.info("GPU (Cuda) was found. The model will therefore be loaded on the GPU for faster speed.")
        else:
            self.device = 'cpu'
            if self.menu_language == "bulgarian":
                logging.info("Внимание! GPU (Cuda) не беше открит! Моделът ще бъде зареден на CPU. Поради тази причина "
                             "драстично ще се забави неговата скорост. Инсталирайте CUDA Toolkit за по-бърза скорост.")
            else:
                logging.info("Warning! GPU (Cuda) not found! Model will be loaded on the CPU. It will therefore be "
                             "drastically slower. Install CUDA Toolkit for speeding up the process.")
            self.ram_memory_bytes = psutil.virtual_memory().total
            self.ram_memory_gb = round(self.ram_memory_bytes / (2 ** 30))
            print(f"RAM memory found: {self.ram_memory_gb} GB")
            if self.menu_language == "bulgarian":
                logging.info(f"RAM памет открита: {self.ram_memory_gb} GB")
            else:
                logging.info(f"RAM memory found: {self.ram_memory_gb} GB")
            if self.ram_memory_gb <= 9:
                self.model_file = self.model_q4ks_file
            elif self.ram_memory_gb <= 10:
                self.model_file = self.model_q4km_file
            elif self.ram_memory_gb <= 11:
                self.model_file = self.model_q5ks_file
            elif self.ram_memory_gb <= 12:
                self.model_file = self.model_q5km_file
            elif self.ram_memory_gb <= 14:
                self.model_file = self.model_q6k_file
            elif self.ram_memory_gb <= 22:
                self.model_file = self.model_q8_file
            else:
                self.model_file = self.model_f16_file
        print(f"LLM file chosen based on computer parameters: {self.model_file}")
        if self.menu_language == "bulgarian":
            logging.info(f"LLM файл, избран на база параметрите на компютъра: {self.model_file}")
        else:
            logging.info(f"LLM file chosen based on computer parameters: {self.model_file}")
        print(f"Device: {self.device} \n")

    @staticmethod
    def clean_text(element):
        """Removes new line ('\n') and other tags from the text column of an element in the dataset"""
        element['text'] = ' '.join(element['text'].split())
        return element

    @staticmethod
    def preprocess_raw_text(text):
        preprocessed_text = ' '.join(text.split())
        return preprocessed_text

    @staticmethod
    def lemmatize_text(text):
        bg_coefficient = in_target_language(text, lang='bg')
        en_coefficient = in_target_language(text, lang='en')
        if bg_coefficient >= en_coefficient:
            language = "bg"
        else:
            language = "en"
        return ' '.join(text_lemmatizer(text, lang=language))

    @staticmethod
    def get_text_tokens_length(text):
        return len(text) / 2.5

    @staticmethod
    def split_list(input_list: list, chunk_size: int) -> list[list[str]]:
        return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

    @staticmethod
    def split_list_with_overlap(input_list: list, chunk_size: int, overlap_size: int) -> list[list[str]]:
        result_list = [input_list[:chunk_size]]
        if len(input_list) <= chunk_size:
            return result_list
        assert overlap_size < chunk_size
        for i in range(chunk_size, len(input_list), chunk_size - overlap_size):
            chunk = input_list[i - overlap_size: i + chunk_size - overlap_size]
            result_list.append(chunk)
        return result_list

    @staticmethod
    def process_ms_word_file(filename):
        filename_without_extension, file_extension = os.path.splitext(filename)
        output_filename = filename_without_extension + '.pdf'
        if file_extension == '.docx':
            convert(filename, output_filename)
            return output_filename
        if file_extension == '.doc':
            wd_format_pdf = 17
            in_file = os.path.abspath(filename)
            out_file = os.path.abspath(output_filename)
            word = comtypes.client.CreateObject('Word.Application')
            doc = word.Documents.Open(in_file)
            doc.SaveAs(out_file, FileFormat=wd_format_pdf)
            doc.Close()
            word.Quit()
            return output_filename

    @staticmethod
    def create_file_embeddings(chunks_dict, index_faiss_user_files=None):
        file_embeddings_list = [el['embedding'] for el in chunks_dict]
        file_embeddings = torch.tensor(np.array(file_embeddings_list), dtype=torch.float32).cpu()
        if index_faiss_user_files is None:
            d = file_embeddings.shape[1]
            m = 64
            ef_construction = 64
            ef_search = 128
            index_faiss_user_files = faiss.IndexHNSWFlat(d, m)
            index_faiss_user_files.hnsw.efConstruction = ef_construction
            index_faiss_user_files.hnsw.efSearch = ef_search
            index_faiss_user_files.add(file_embeddings.numpy())
        else:
            index_faiss_user_files.add(file_embeddings.numpy())
        return index_faiss_user_files, file_embeddings

    @staticmethod
    def read_dataset_files(chunk_dict_filename="db_chunks_and_embeddings.pkl",
                           faiss_store_filename="faiss_vector_db.index"):
        if not os.path.exists(chunk_dict_filename):
            return None, None, None
        with open(chunk_dict_filename, 'rb') as f:
            chunks_and_embeddings = pickle.load(f)
        if not faiss_store_filename:
            faiss_index = None
        else:
            faiss_index = read_index(faiss_store_filename)
        embeddings_list = [element['embedding'] for element in chunks_and_embeddings]
        embeddings = torch.tensor(np.array(embeddings_list), dtype=torch.float32).cpu()
        return chunks_and_embeddings, faiss_index, embeddings

    @staticmethod
    def retrieve_relevant_urls(context_items: list[dict], unique=True) -> [str]:
        relevant_urls = []
        for element in context_items:
            relevant_urls.append(element['url'])
        if unique:
            unique_relevant_urls = list(dict.fromkeys(relevant_urls))
            return unique_relevant_urls
        else:
            return relevant_urls

    @staticmethod
    def reciprocal_rank_fusion_raw(indices_list, k=60, topn_articles=3):
        fused_scores = {}
        for lst in indices_list:
            for rank in range(len(lst)):
                if lst[rank] not in fused_scores:
                    fused_scores[lst[rank]] = 0
                fused_scores[lst[rank]] += 1 / (rank + k)

        sorted_fused_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        reranked_indices = [idx for idx, score in sorted_fused_scores]
        new_scores = [score for idx, score in sorted_fused_scores]

        if topn_articles <= 0:
            topn_articles = 1
        elif topn_articles > len(reranked_indices):
            topn_articles = len(reranked_indices)
        return reranked_indices[:topn_articles], new_scores[:topn_articles]

    @staticmethod
    def is_llama_gpu_available() -> bool:
        # add your own path to the installation directory of the BG_RAG app (up to BG_RAG directory) if you want to use
        # this function which checks whether llama is available for gpu (don't change the part of the path from venv
        # directory onwards)
        lib = load_shared_library(
            'llama', pathlib.Path(rf'{Path.home()}\PycharmProjects\BG_RAG\venv\Lib\site-packages\llama_cpp\lib'))

        return bool(lib.llama_supports_gpu_offload())

    def load_embedding_model(self):
        if self.menu_language == "bulgarian":
            logging.info(f"Зареждане на embedding модел (BAAI/bge-m3) на {self.device}...")
        else:
            logging.info(f"Loading embedding model (BAAI/bge-m3) on {self.device}...")

        try:
            self.embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=self.device)

        except Exception as e:
            logging.exception(f"ERROR: {e}", stack_info=True)
            print(f"ERROR: {e}")
            if self.menu_language == "bulgarian":
                logging.info("ПРОБЛЕМ със зареждането на embedding модел (BAAI/bge-m3)! Системата НЯМА да работи "
                             "правилно! Ако за първи път използвате системата, проверете интернет връзката си. Тя е "
                             "необходима, за да се изтегли embedding модела. Оправете интернет връзката си и "
                             "рестартирайте приложението. Ако това не помогне, инсталирайте наново приложението.")
            else:
                logging.info("PROBLEM with the loading of the embedding model (BAAI/bge-m3)! The system will NOT work "
                             "properly! If you use the system for the first time, check your internet connection. "
                             "It is needed for the download of the LLM. Sort out your internet connection and restart "
                             "the app. If this does not help, then reinstall the app.")
            print("PROBLEM with the loading of the embedding model (BAAI/bge-m3)! The system will NOT work properly! "
                  "If you use the system for the first time, check your internet connection. It is needed for the "
                  "download of the LLM. Sort out your internet connection and restart the app. If this does not "
                  "help, then reinstall the app.")
        else:
            if self.menu_language == "bulgarian":
                logging.info("Embedding модел (BAAI/bge-m3) беше зареден успешно.")
            else:
                logging.info("Embedding model (BAAI/bge-m3) was successfully loaded.")

    def load_reranking_model(self):
        if self.menu_language == "bulgarian":
            logging.info(f"Зареждане на реранкиращ (reranking) модел (jinaai/jina-reranker-v2-base-multilingual) на "
                         f"{self.device}...")
        else:
            logging.info(f"Loading reranking model (jinaai/jina-reranker-v2-base-multilingual) on {self.device}...")
        try:
            self.reranking_model = CrossEncoder(
                "jinaai/jina-reranker-v2-base-multilingual",
                device=self.device,
                automodel_args={"torch_dtype": "auto"},
                trust_remote_code=True,
            )
        except Exception as e:
            logging.exception(f"ERROR: {e}", stack_info=True)
            print(f"ERROR: {e}")
            if self.menu_language == "bulgarian":
                logging.info("ПРОБЛЕМ със зареждането на реранкиращ (reranking) модел "
                             "(jina-reranker-v2-base-multilingual)! Системата НЯМА да работи правилно! "
                             "Ако за първи път използвате системата, проверете интернет връзката си. Тя е "
                             "необходима, за да се изтегли реранкиращият модел. Оправете интернет връзката си и "
                             "рестартирайте приложението. Ако това не помогне, инсталирайте наново приложението.")
            else:
                logging.info("PROBLEM with the loading of the reranking model (jina-reranker-v2-base-multilingual)! "
                             "The system will NOT work properly! If you use the system for the first time, check your "
                             "internet connection. It is needed for the download of the LLM. Sort out your internet "
                             "connection and restart the app. If this does not help, then reinstall the app.")
            print("PROBLEM with the loading of the reranking model (jina-reranker-v2-base-multilingual)! "
                  "The system will NOT work properly! "
                  "If you use the system for the first time, check your internet connection. It is needed for the "
                  "download of the LLM. Sort out your internet connection and restart the app. If this does not "
                  "help, then reinstall the app.")
        else:
            if self.menu_language == "bulgarian":
                logging.info("Реранкиращият модел (jinaai/jina-reranker-v2-base-multilingual) беше зареден успешно.")
            else:
                logging.info("Reranking model (jinaai/jina-reranker-v2-base-multilingual) was successfully loaded.")

    def read_evaluation_df(self):
        if os.path.exists(self.evaluation_df_store_pickle_filename):
            self.evaluation_df = pd.read_pickle(self.evaluation_df_store_pickle_filename)
        else:
            self.evaluation_df = pd.DataFrame(columns=['query', 'answer', 'relevant_contexts', 'answer_relevance_score',
                                                       'answer_relevance_reason', 'context_relevance_score',
                                                       'groundedness_score', 'rag_fusion', 'hyde',
                                                       'generated_subqueries', 'resource_llm_check', 'topn_contexts',
                                                       'language'])

    def evaluate_system(self):
        if not os.path.exists(self.evaluation_df_store_pickle_filename):
            return
        if self.evaluator is None:
            self.evaluator = REMi()
        if self.evaluation_df is None:
            self.read_evaluation_df()

        for index, row in self.evaluation_df.iterrows():
            if row['answer_relevance_score'] == -1:
                print(f"Row {index} evaluation began.")
                result = self.evaluator.evaluate_rag(query=self.evaluation_df.iloc[index]['query'],
                                                     answer=self.evaluation_df.iloc[index]['answer'],
                                                     contexts=self.evaluation_df.iloc[index]['relevant_contexts'])
                print(f"{index} evaluation done.")
                answer_relevance, context_relevances, groundednesses = result
                self.evaluation_df.at[index, 'answer_relevance_score'] = answer_relevance.score
                self.evaluation_df.at[index, 'answer_relevance_reason'] = answer_relevance.reason
                self.evaluation_df.at[index, 'context_relevance_score'] = [cr.score for cr in context_relevances]
                self.evaluation_df.at[index, 'groundedness_score'] = [g.score for g in groundednesses]

        self.evaluation_df.to_pickle(self.evaluation_df_store_pickle_filename)
        print(f"{self.evaluation_df_store_pickle_filename} file updated.")

    def print_evaluation_results(self, print_techniques_used=False, print_relevant_contexts=False):
        if not os.path.exists(self.evaluation_df_store_pickle_filename):
            print("No evaluation dataset found!")
            return
        if self.evaluation_df is None:
            self.read_evaluation_df()
        for i, row in self.evaluation_df.iterrows():
            print(f"Query:                    {row['query']}")
            print(f"answer_relevance_score:   {row['answer_relevance_score']}")
            print(f"answer_relevance_reason:  {row['answer_relevance_reason']}")
            print(f"context_relevance_score:  {row['context_relevance_score']}")
            print(f"groundedness_score:       {row['groundedness_score']}\n")
            if print_techniques_used:
                print(f"rag_fusion:           {row['rag_fusion']}")
                print(f"hyde:                 {row['hyde']}")
                print(f"generated_subqueries: {row['generated_subqueries']}")
                print(f"resource_llm_check:   {row['resource_llm_check']}")
                print(f"topn_contexts:        {row['topn_contexts']}")
                print(f"language:             {row['language']}\n\n")
            if print_relevant_contexts:
                print(f"relevant_contexts:    {row['relevant_contexts']}\n\n")

    @staticmethod
    def calculate_df_averages(df, technique=""):
        answer_relevance_scores_avg = round(df['answer_relevance_score'].mean(), 2)

        context_relevance_scores_avg_list = df['context_relevance_score'].tolist()
        cr_mean_list = [np.mean(cr) for cr in context_relevance_scores_avg_list]
        context_relevance_scores_avg = np.mean(cr_mean_list)
        context_relevance_scores_avg = round(context_relevance_scores_avg, 2)

        context_max_relevance_list = [max(cr) for cr in context_relevance_scores_avg_list]
        context_max_relevance_avg = round(np.mean(context_max_relevance_list), 2)

        groundedness_scores_avg_list = df['groundedness_score'].tolist()
        gr_mean_list = [np.mean(gr) for gr in groundedness_scores_avg_list]
        groundedness_scores_avg = np.mean(gr_mean_list)
        groundedness_scores_avg = round(groundedness_scores_avg, 2)

        groundedness_max_scores_list = [max(gr) for gr in groundedness_scores_avg_list]
        groundedness_max_scores_avg = round(np.mean(groundedness_max_scores_list), 2)

        print(f"{technique}_answer_relevance_scores_avg:  {answer_relevance_scores_avg}")
        print(f"{technique}_context_relevance_scores_avg: {context_relevance_scores_avg}")
        print(f"{technique}_groundedness_scores_avg:      {groundedness_scores_avg}")
        print(f"{technique}_context_max_relevance_avg:    {context_max_relevance_avg}")
        print(f"{technique}_groundedness_max_scores_avg:  {groundedness_max_scores_avg}\n")

    def calculate_evaluation_averages(self, language=""):
        # Note: language argument should be of values "bg" or "en"
        if not os.path.exists(self.evaluation_df_store_pickle_filename):
            print("No evaluation dataset found!")
            return
        if self.evaluation_df is None:
            self.read_evaluation_df()

        self.calculate_df_averages(self.evaluation_df, technique="general")

        rag_fusion_rows = self.evaluation_df.loc[self.evaluation_df['rag_fusion'] == True]
        hyde_rows = self.evaluation_df.loc[self.evaluation_df['hyde'] == True]

        both_rows = self.evaluation_df.loc[(self.evaluation_df['rag_fusion'] == True) &
                                           (self.evaluation_df['hyde'] == True)]
        none_rows = self.evaluation_df.loc[(self.evaluation_df['rag_fusion'] == False) &
                                           (self.evaluation_df['hyde'] == False)]
        only_rag_fusion_rows = self.evaluation_df.loc[(self.evaluation_df['rag_fusion'] == True) &
                                                      (self.evaluation_df['hyde'] == False)]
        only_hyde_rows = self.evaluation_df.loc[(self.evaluation_df['rag_fusion'] == False) &
                                                (self.evaluation_df['hyde'] == True)]

        if language:
            rag_fusion_rows = rag_fusion_rows.loc[rag_fusion_rows['language'] == language]
            hyde_rows = hyde_rows.loc[hyde_rows['language'] == language]
            both_rows = both_rows.loc[both_rows['language'] == language]
            none_rows = none_rows.loc[none_rows['language'] == language]
            only_rag_fusion_rows = only_rag_fusion_rows.loc[only_rag_fusion_rows['language'] == language]
            only_hyde_rows = only_hyde_rows.loc[only_hyde_rows['language'] == language]

        self.calculate_df_averages(rag_fusion_rows, technique="rag_fusion")
        self.calculate_df_averages(hyde_rows, technique="hyde")
        self.calculate_df_averages(both_rows, technique="both")
        self.calculate_df_averages(none_rows, technique="none")
        self.calculate_df_averages(only_rag_fusion_rows, technique="only_rag_fusion")
        self.calculate_df_averages(only_hyde_rows, technique="only_hyde")

        print(f"rag_fusion_rows: {len(rag_fusion_rows)}")
        print(f"hyde_rows: {len(hyde_rows)}")
        print(f"both_rows: {len(both_rows)}")
        print(f"none_rows: {len(none_rows)}")
        print(f"only_rag_fusion_rows: {len(only_rag_fusion_rows)}")
        print(f"only_hyde_rows: {len(only_hyde_rows)}\n")

    def preprocess_text(self, text: str) -> str:
        text = " ".join(re.sub(r'[^\w\s-]', '', text).split()).lower()
        text = ' '.join([word for word in text.split() if word not in self.stopwords])
        return text

    def split_extra_long_sentences(self, sentences_list):
        new_sent_list = []
        for sentence in sentences_list:
            if len(sentence) > 500:
                for short_sent in self.split_list(sentence, self.extra_long_sentence_splitter_size):
                    new_sent_list.append(short_sent)
            else:
                new_sent_list.append(sentence)
        return new_sent_list

    def write_pdf_embeddings_to_faiss(self, chunks_dict):
        pdf_embeddings_list = [el['embedding'] for el in chunks_dict]
        pdf_embeddings = torch.tensor(np.array(pdf_embeddings_list), dtype=torch.float32).cpu()
        d = pdf_embeddings.shape[1]
        m = 64
        ef_construction = 64
        ef_search = 128
        pdf_index = faiss.IndexHNSWFlat(d, m)
        pdf_index.hnsw.efConstruction = ef_construction
        pdf_index.hnsw.efSearch = ef_search
        pdf_index.add(pdf_embeddings.numpy())
        write_index(pdf_index, self.faiss_store_filename)
        return pdf_embeddings

    def process_pdf(self, filename):
        if not os.path.exists(filename):
            print("File does not exist!")
        pdf_doc = fitz.open(filename)  # open a document
        chunks_dict = []
        text = ""
        for page_number, page in enumerate(pdf_doc, 1):  # iterate the document pages
            text += page.get_text() + '\n'  # get plain text encoded as UTF-8
        text = self.preprocess_raw_text(text)

        bg_coefficient = in_target_language(text, lang='bg')
        en_coefficient = in_target_language(text, lang='en')
        if bg_coefficient >= en_coefficient:
            self.nlp = self.nlp_bg
        else:
            self.nlp = self.nlp_en

        sentences_list = list(self.nlp(text).sents)
        sentences_list = [str(sentence) for sentence in sentences_list]
        sentences_list = self.split_extra_long_sentences(sentences_list)
        sentence_count = len(sentences_list)
        chunks_list = self.split_list_with_overlap(input_list=sentences_list, chunk_size=self.chunk_sentences_size,
                                                   overlap_size=self.chunk_sentences_overlap)
        chunks_count = len(chunks_list)

        i = 0
        for chunk_sentences in chunks_list:
            i += 1
            joined_chunk_sentences = "".join(chunk_sentences).replace("  ", " ").strip()
            re.sub(r'([.?!:;/\""])([А-ЯA-Z])', r'\1 \2', text)
            joined_chunk_sentences = re.sub(r'([.?!])([А-ЯA-Z])', r'\1 \2', joined_chunk_sentences)
            processed_chunk = self.lemmatize_text(self.preprocess_text(joined_chunk_sentences))

            chunks_dict.append({"id": i,
                                "url": os.path.basename(filename),
                                "title": os.path.basename(filename),
                                "chunk_text": joined_chunk_sentences,
                                "chunk_char_count": len(joined_chunk_sentences),
                                "chunk_word_count": len([word for word in joined_chunk_sentences.split(" ")]),
                                "chunk_token_count": self.get_text_tokens_length(joined_chunk_sentences),
                                "processed_chunk": processed_chunk})
        chunks_dict = [element for element in chunks_dict if element['chunk_token_count'] > self.min_token_length]
        for element in chunks_dict:
            element['embedding'] = self.embedding_model.encode(element['chunk_text'], max_length=8192)['dense_vecs']
        return chunks_dict

    def process_text_file(self, filename):
        if not os.path.exists(filename):
            print("File does not exist!")
        text_doc = open(filename, encoding='utf-8', mode="r")
        text = text_doc.read()
        chunks_dict = []

        text = self.preprocess_raw_text(text)

        bg_coefficient = in_target_language(text, lang='bg')
        en_coefficient = in_target_language(text, lang='en')
        if bg_coefficient >= en_coefficient:
            self.nlp = self.nlp_bg
        else:
            self.nlp = self.nlp_en

        sentences_list = list(self.nlp(text).sents)
        sentences_list = [str(sentence) for sentence in sentences_list]
        sentences_list = self.split_extra_long_sentences(sentences_list)
        sentence_count = len(sentences_list)
        chunks_list = self.split_list_with_overlap(input_list=sentences_list, chunk_size=self.chunk_sentences_size,
                                                   overlap_size=self.chunk_sentences_overlap)
        chunks_count = len(chunks_list)
        i = 0
        for chunk_sentences in chunks_list:
            i += 1
            joined_chunk_sentences = "".join(chunk_sentences).replace("  ", " ").strip()
            re.sub(r'([.?!:;/\""])([А-ЯA-Z])', r'\1 \2', text)
            joined_chunk_sentences = re.sub(r'([.?!])([А-ЯA-Z])', r'\1 \2', joined_chunk_sentences)
            processed_chunk = self.lemmatize_text(self.preprocess_text(joined_chunk_sentences))

            chunks_dict.append({"id": i,
                                "url": os.path.basename(filename),
                                "title": os.path.basename(filename),
                                "chunk_text": joined_chunk_sentences,
                                "chunk_char_count": len(joined_chunk_sentences),
                                "chunk_word_count": len([word for word in joined_chunk_sentences.split(" ")]),
                                "chunk_token_count": self.get_text_tokens_length(joined_chunk_sentences),
                                "processed_chunk": processed_chunk})
        chunks_dict = [element for element in chunks_dict if element['chunk_token_count'] > self.min_token_length]
        for element in chunks_dict:
            element['embedding'] = self.embedding_model.encode(element['chunk_text'], max_length=8192)['dense_vecs']
        return chunks_dict

    def process_json_file(self, filename):
        if not os.path.exists(filename):
            return []
        with open(filename, mode='r', encoding="utf-8") as file:
            data = file.read()
        dictionary_list = json.loads(data)
        chunks_dict = []
        for dictionary in dictionary_list:
            text = dictionary["text"]
            text = self.preprocess_raw_text(text)

            bg_coefficient = in_target_language(text, lang='bg')
            en_coefficient = in_target_language(text, lang='en')
            if bg_coefficient >= en_coefficient:
                self.nlp = self.nlp_bg
            else:
                self.nlp = self.nlp_en

            sentences_list = list(self.nlp(text).sents)
            sentences_list = [str(sentence) for sentence in sentences_list]
            sentences_list = self.split_extra_long_sentences(sentences_list)
            sentence_count = len(sentences_list)
            chunks_list = self.split_list_with_overlap(input_list=sentences_list, chunk_size=self.chunk_sentences_size,
                                                       overlap_size=self.chunk_sentences_overlap)
            chunks_count = len(chunks_list)
            dictionary["char_count"] = len(text)
            dictionary["words_count"] = len(text.split(" "))
            dictionary["token_count"] = len(text.split(". "))
            dictionary["sentences_list"] = sentence_count
            dictionary["chunks_list"] = chunks_list
            dictionary["chunks_count"] = chunks_count

            for chunk_sentences in chunks_list:
                joined_chunk_sentences = "".join(chunk_sentences).replace("  ", " ").strip()
                re.sub(r'([.?!:;/\""])([А-ЯA-Z])', r'\1 \2', text)
                joined_chunk_sentences = re.sub(r'([.?!])([А-ЯA-Z])', r'\1 \2', joined_chunk_sentences)
                processed_chunk = self.lemmatize_text(self.preprocess_text(joined_chunk_sentences))

                chunks_dict.append({"id": dictionary["id"],
                                    "url": dictionary["url"],
                                    "title": dictionary["title"],
                                    "chunk_text": joined_chunk_sentences,
                                    "chunk_char_count": len(joined_chunk_sentences),
                                    "chunk_word_count": len([word for word in joined_chunk_sentences.split(" ")]),
                                    "chunk_token_count": self.get_text_tokens_length(joined_chunk_sentences),
                                    "processed_chunk": processed_chunk})
        chunks_dict = [element for element in chunks_dict if element['chunk_token_count'] > self.min_token_length]
        for element in chunks_dict:
            element['embedding'] = self.embedding_model.encode(element['chunk_text'], max_length=8192)['dense_vecs']
        return chunks_dict

    def process_uploaded_file(self, filename):
        filename_without_extension, file_extension = os.path.splitext(filename)
        if file_extension == '.txt':
            chunks_dict = self.process_text_file(filename)
            return chunks_dict
        if file_extension == '.json':
            chunks_dict = self.process_json_file(filename)
            return chunks_dict
        if file_extension == '.pdf':
            chunks_dict = self.process_pdf(filename)
            return chunks_dict
        if file_extension == '.docx' or file_extension == '.doc':
            pdf_filename = self.process_ms_word_file(filename)
            chunks_dict = self.process_pdf(pdf_filename)
            return chunks_dict
        return []

    def upload_files(self, user_filenames, united_chunks_dictionary=None, index_faiss_user_files=None,
                     united_embeddings=None, chunk_dict_filename="uploaded_files_chunk_dict.pkl",
                     faiss_store_filename="uploaded_files_faiss_vector_db.index"):
        if self.menu_language == "bulgarian":
            logging.info("Качване на файловете...")
        else:
            logging.info("Uploading files...")
        if not user_filenames:
            return [], None, [], []
        if united_embeddings is None:
            united_embeddings = torch.empty((0, 1024), dtype=torch.float32)
        if united_chunks_dictionary is None:
            united_chunks_dictionary = []
        i = 0
        uploaded_filenames = []
        for filename in user_filenames:
            try:
                chunks_dictionary = self.process_uploaded_file(filename)
                if not chunks_dictionary:
                    if self.menu_language == "bulgarian":
                        logging.info(
                            f"Има проблем с файла '{filename}', който качвате! Вероятно НЕ може да бъде прочетен.")
                        logging.info(f"Следователно, файлът '{filename}' НЯМА да бъде качен!")
                    else:
                        logging.info(f"There is an issue with file '{filename}' that you upload! Probably it CANNOT be "
                                     f"read.")
                        logging.info(f"Therefore, file '{filename}' will NOT be imported!")
                    print(f"There is an issue with file '{filename}' that you upload! Probably it cannot be read.")
                    print(f"Therefore, file '{filename}' will NOT be imported!")
                    continue
                index_faiss_user_files, file_embeddings = \
                    self.create_file_embeddings(chunks_dict=chunks_dictionary,
                                                index_faiss_user_files=index_faiss_user_files)
                united_chunks_dictionary.extend(chunks_dictionary)
                united_embeddings = torch.cat((united_embeddings, file_embeddings))
                uploaded_filenames.append(os.path.basename(filename))
            except Exception as e:
                logging.exception(f"ERROR: {e}", stack_info=True)
                print(f"ERROR: {e}")
                if self.menu_language == "bulgarian":
                    logging.info(
                        f"Има проблем с файла '{filename}', който качвате! Вероятно НЕ може да бъде прочетен.")
                    logging.info(f"Следователно, файлът '{filename}' НЯМА да бъде качен!")
                else:
                    logging.info(f"There is an issue with file '{filename}' that you upload! Probably it CANNOT be "
                                 f"read.")
                    logging.info(f"Therefore, file '{filename}' will NOT be imported!")
                print(f"There is an issue with file '{filename}' that you upload! Probably it cannot be read.")
                print(f"Therefore, file '{filename}' will NOT be imported!")
                continue

            i += 1
            print("\rProgress: {} / {}".format(i, len(user_filenames)), end="")
            if self.menu_language == "bulgarian":
                logging.info(f"Прогрес: {i} / {len(user_filenames)}")
            else:
                logging.info(f"Progress: {i} / {len(user_filenames)}")

        print()
        if i != 0:
            if os.path.exists(self.uploaded_files_bm25s_store_filename):
                shutil.rmtree(self.uploaded_files_bm25s_store_filename)
            write_index(index_faiss_user_files, faiss_store_filename)
            with open(chunk_dict_filename, 'wb') as chunk_dict_file:
                pickle.dump(united_chunks_dictionary, chunk_dict_file, protocol=pickle.HIGHEST_PROTOCOL)

        if self.menu_language == "bulgarian":
            diff = len(user_filenames) - len(uploaded_filenames)
            if diff == 0:
                logging.info("Файловете бяха качени успешно.")
            if diff > 0:
                if diff == len(user_filenames):
                    logging.info("Файловете НЕ бяха качени! Имаше проблеми с прочитането на всички файлове.")
                else:
                    logging.info(f"С някои файлове ({diff} от тях) имаше ПРОБЛЕМ и не бяха качени. Вижте по-горе в "
                                 f"лога за повече информация. Останалите файлове бяха качени успешно.")
        else:
            diff = len(user_filenames) - len(uploaded_filenames)
            if diff == 0:
                logging.info("Files were uploaded successfully.")
            if diff > 0:
                if diff == len(user_filenames):
                    logging.info("The files were NOT imported! There was a problem with the reading of all of them.")
                else:
                    logging.info(f"There was a PROBLEM with some of the files ({diff} of them) and they were not "
                                 f"imported. Look previously in the log to see more information. The other files were "
                                 f"imported successfully.")
        return united_chunks_dictionary, index_faiss_user_files, united_embeddings, uploaded_filenames

    def read_uploaded_files_data(self):
        if self.menu_language == "bulgarian":
            logging.info("Четене на базата данни от качените файлове от " + self.uploaded_files_chunk_dict_filename +
                         " файл...")
        else:
            logging.info("Reading uploaded files data from " + self.uploaded_files_chunk_dict_filename + " file...")
        if not os.path.exists(self.uploaded_files_chunk_dict_filename):
            return None, None, None
        try:
            with open(self.uploaded_files_chunk_dict_filename, 'rb') as f:
                self.user_uploaded_chunks_and_embeddings = pickle.load(f)
            uploaded_embeddings_list = [el['embedding'] for el in self.user_uploaded_chunks_and_embeddings]
            self.user_uploaded_embeddings = torch.tensor(np.array(uploaded_embeddings_list), dtype=torch.float32).cpu()
            self.user_uploaded_faiss_index = read_index(self.uploaded_files_faiss_vector_db_filename)

        except Exception as e:
            logging.exception(f"ERROR: {e}", stack_info=True)
            print(f"ERROR: {e}")
            if self.menu_language == "bulgarian":
                logging.info("ПРОБЛЕМ с прочитането на базата данни от качените файлове. Системата може да НЕ "
                             "функционира правилно с тази опция.")
            else:
                logging.info("PROBLEM with the reading of the uploaded files data. The system may NOT function "
                             "properly with that option.")
            print("PROBLEM with the reading of the uploaded files data. The system may NOT function "
                  "properly with that option.")
        else:
            if self.menu_language == "bulgarian":
                logging.info("Базата данни от качените файлове беше прочетена успешно.")
            else:
                logging.info("Uploaded files data was read and loaded successfully.")
            print("Uploaded files data was read and loaded successfully.")

    def read_focus_news_json_files(self, first_file_number=0, last_file_number=14, new_files=2):
        if self.menu_language == "bulgarian":
            logging.info("Четене на focus news json файлове...")
        else:
            logging.info("Reading focus news json files...")
        # before adding new focus news json files remove "focus_news_chunk_dict.pkl" and
        # "focus_news_faiss_vector_db.index" files for a completely new creation of the database or simply change which
        # new files should be read in the following lines, thus expanding the focus news database with the new files
        focus_news_json_file_tmpl = 'focus_news/focus_news_articles_{}.json'
        focus_news_json_files_list = [focus_news_json_file_tmpl.format(f"0_{i}") for i in range(new_files, 0, -1)]
        focus_news_json_files_list += [focus_news_json_file_tmpl.format(i) for i in range(first_file_number,
                                                                                          last_file_number + 1)]
        try:
            self.focus_news_chunk_dict, self.focus_news_faiss_index, self.focus_news_embeddings, _ = \
                self.upload_files(focus_news_json_files_list,
                                  united_chunks_dictionary=self.focus_news_chunk_dict,
                                  index_faiss_user_files=self.focus_news_faiss_index,
                                  united_embeddings=self.focus_news_embeddings,
                                  chunk_dict_filename=self.focus_news_chunk_dict_filename,
                                  faiss_store_filename=self.focus_news_faiss_store_filename)

        except Exception as e:
            logging.exception(f"ERROR: {e}", stack_info=True)
            print(f"ERROR: {e}")
            if self.menu_language == "bulgarian":
                logging.info("ГРЕШКА! ПРОБЛЕМ с прочитането на focus news json файловете!")
            else:
                logging.info("ERROR! PROBLEM with the reading of the focus news json files!")
            print("ERROR! PROBLEM with the reading of the focus news json files!")
        else:
            if self.menu_language == "bulgarian":
                logging.info("Focus news json файловете бяха прочетени и заредени успешно.")
            else:
                logging.info("Focus news json files were read and loaded successfully.")

    def delete_uploaded_files(self):
        if self.menu_language == "bulgarian":
            logging.info("Изтриване на качените файлове...")
        else:
            logging.info("Deleting uploaded files...")

        os.remove(self.uploaded_files_chunk_dict_filename)
        os.remove(self.uploaded_files_faiss_vector_db_filename)
        if os.path.exists(self.uploaded_files_bm25s_store_filename):
            shutil.rmtree(self.uploaded_files_bm25s_store_filename)

        if self.menu_language == "bulgarian":
            logging.info("Изтриването на качените файлове беше завършено успешно.")
        else:
            logging.info("Deletion of uploaded files completed.")

        self.user_uploaded_chunks_and_embeddings = None
        self.user_uploaded_faiss_index = None
        self.user_uploaded_embeddings = None

    def create_count_columns(self, element):
        element['char_count'] = len(element['text'])
        element['words_count'] = len(element['text'].split(" "))
        element['token_count'] = self.get_text_tokens_length(element['text'])

        bg_coefficient = in_target_language(element['text'], lang='bg')
        en_coefficient = in_target_language(element['text'], lang='en')
        if bg_coefficient >= en_coefficient:
            self.nlp = self.nlp_bg
        else:
            self.nlp = self.nlp_en

        element['sentences_list'] = list(self.nlp(element['text']).sents)
        element['sentences_list'] = [str(sentence) for sentence in element['sentences_list']]
        element['sentences_list'] = self.split_extra_long_sentences(element['sentences_list'])
        element['sentence_count'] = len(element['sentences_list'])
        element['chunks_list'] = self.split_list_with_overlap(input_list=element['sentences_list'],
                                                              chunk_size=self.chunk_sentences_size,
                                                              overlap_size=self.chunk_sentences_overlap)
        element['chunks_count'] = len(element['chunks_list'])
        return element

    def load_bg_wiki_dataset_to_pickle(self, date="20250120"):
        if self.menu_language == "bulgarian":
            logging.info("Зареждане на bg wiki база данни от източника...")
        else:
            logging.info('Loading bg wiki dataset from source...')
        bg_wiki = load_dataset("wikipedia", language="bg", date=date, trust_remote_code=True)
        bg_wiki['train'] = bg_wiki['train'].map(self.clean_text)
        char_count_col = [1] * len(bg_wiki['train'])
        bg_wiki['train'] = bg_wiki['train'].add_column('char_count', char_count_col)
        words_count_col = [1] * len(bg_wiki['train'])
        bg_wiki['train'] = bg_wiki['train'].add_column('words_count', words_count_col)
        sentence_count_col = [1] * len(bg_wiki['train'])
        bg_wiki['train'] = bg_wiki['train'].add_column('sentence_count', sentence_count_col)
        token_count_col = [1] * len(bg_wiki['train'])
        bg_wiki['train'] = bg_wiki['train'].add_column('token_count', token_count_col)
        sentences_list_col = [1] * len(bg_wiki['train'])
        bg_wiki['train'] = bg_wiki['train'].add_column('sentences_list', sentences_list_col)
        chunks_list_col = [1] * len(bg_wiki['train'])
        bg_wiki['train'] = bg_wiki['train'].add_column('chunks_list', chunks_list_col)
        chunks_count_col = [1] * len(bg_wiki['train'])
        bg_wiki['train'] = bg_wiki['train'].add_column('chunks_count', chunks_count_col)

        bg_wiki['train'] = bg_wiki['train'].map(self.create_count_columns)

        bg_wiki_df = bg_wiki['train'].to_pandas()
        bg_wiki_df.to_pickle(self.bg_wiki_pickle_filename)
        if self.menu_language == "bulgarian":
            logging.info('Bg wiki база данни беше заредена успешно. Съдържанието беше записано в ' +
                         self.bg_wiki_pickle_filename)
        else:
            logging.info('Bg wiki dataset was loaded successfully. The content was written to ' +
                         self.bg_wiki_pickle_filename)

    def read_bg_wiki_dataset_from_pickle(self):
        if self.menu_language == "bulgarian":
            logging.info('Четене на bg wiki база данни от ' + self.bg_wiki_pickle_filename + ' ...')
        else:
            logging.info('Reading bg wiki dataset from ' + self.bg_wiki_pickle_filename + ' ...')
        self.bg_wiki_df = pd.read_pickle(self.bg_wiki_pickle_filename)
        if self.menu_language == "bulgarian":
            logging.info('Bg wiki база данни беше записана успешно.')
        else:
            logging.info('Bg wiki dataset was read successfully.')

    def create_bg_wiki_chunks(self):
        if self.menu_language == "bulgarian":
            logging.info('Създаване на откъси (chunks) за bg wiki база данни плюс филтриране (премахване на твърде '
                         'кратки откъси)...')
        else:
            logging.info('Creating chunks for bg wiki dataset plus filtering (removal of too short chunks)...')

        for index, element in self.bg_wiki_df.iterrows():
            for chunk_sentences in element['chunks_list']:
                chunks_dict = {'id': element['id'], 'url': element['url'], 'title': element['title']}
                joined_chunk_sentences = "".join(chunk_sentences).replace("  ", " ").strip()
                joined_chunk_sentences = re.sub(r'([.?!])([А-ЯA-Z])', r'\1 \2', joined_chunk_sentences)
                chunks_dict['chunk_text'] = joined_chunk_sentences
                chunks_dict['chunk_char_count'] = len(joined_chunk_sentences)
                chunks_dict['chunk_word_count'] = len([word for word in joined_chunk_sentences.split(" ")])
                chunks_dict['chunk_token_count'] = self.get_text_tokens_length(joined_chunk_sentences)
                self.bg_wiki_chunks.append(chunks_dict)

        self.bg_wiki_chunks_df = pd.DataFrame(self.bg_wiki_chunks)
        self.bg_wiki_chunks_df_filtered = self.bg_wiki_chunks_df[self.bg_wiki_chunks_df["chunk_token_count"] >
                                                                 self.min_token_length]
        self.bg_wiki_chunks_df_filtered.to_pickle(self.bg_wiki_chunks_df_filtered_pickle_filename)
        if self.menu_language == "bulgarian":
            logging.info('Bg wiki откъсите (chunks) бяха създадени и филтрирани успешно.')
        else:
            logging.info('The bg wiki chunks were created and filtered successfully.')

    def read_bg_wiki_chunks_filtered(self):
        if self.menu_language == "bulgarian":
            logging.info('Четене на филтрирани bg wiki откъси (chunks) от ' +
                         self.bg_wiki_chunks_df_filtered_pickle_filename + ' ...')
        else:
            logging.info('Reading the filtered bg wiki chunks from ' + self.bg_wiki_chunks_df_filtered_pickle_filename +
                         ' ...')
        self.bg_wiki_chunks_df_filtered = pd.read_pickle(self.bg_wiki_chunks_df_filtered_pickle_filename)
        if self.menu_language == "bulgarian":
            logging.info('Филтрираните bg wiki откъси бяха прочетени успешно.')
        else:
            logging.info('The filtered bg wiki chunks were read successfully.')

    def embed_bg_wiki_chunks_filtered(self):
        if self.menu_language == "bulgarian":
            logging.info('Създаване на bg wiki ембединги (embeddings)...')
        else:
            logging.info("Creating bg wiki embeddings...")
        print("Creating bg wiki embeddings...")

        np.set_printoptions(threshold=sys.maxsize)
        self.bg_wiki_chunks_df_filtered['embedding'] = ""
        self.bg_wiki_chunks_df_filtered['processed_chunk'] = ""
        n = len(self.bg_wiki_chunks_df_filtered)
        m = n // 1000 + 1 if n % 1000 != 0 else n // 1000
        p = m // 10 + 1 if m % 10 != 0 else m // 10
        for i, element in self.bg_wiki_chunks_df_filtered.iterrows():
            self.bg_wiki_chunks_df_filtered.at[i, 'embedding'] = \
                self.embedding_model.encode(element['chunk_text'], max_length=8192)['dense_vecs']
            self.bg_wiki_chunks_df_filtered.at[i, 'processed_chunk'] = \
                self.lemmatize_text(self.preprocess_text(element['chunk_text']))
            i += 1
            if i == n and n % 1000 != 0:
                print("\rProgress: {} / {}".format(m, m), end="")
                if self.menu_language == "bulgarian":
                    logging.info(f"Прогрес: {p} / {p}")
                else:
                    logging.info(f"Progress: {p} / {p}")
            if i % 1000 == 0:
                print("\rProgress: {} / {}".format(i // 1000, m), end="")
            if i % 10000 == 0:
                if self.menu_language == "bulgarian":
                    logging.info(f"Прогрес: {i // 10000} / {p}")
                else:
                    logging.info(f"Progress: {i // 10000} / {p}")

        if self.menu_language == "bulgarian":
            logging.info(f"Записване на bg wiki ембединги в {self.bg_wiki_chunks_and_embeddings_csv_filename} ...")
        else:
            logging.info(f"Saving bg wiki embeddings to {self.bg_wiki_chunks_and_embeddings_csv_filename} ...")
        print("\nSaving bg wiki embeddings...")

        self.bg_wiki_chunks_df_filtered.to_csv(self.bg_wiki_chunks_and_embeddings_csv_filename, index=False)

        if self.menu_language == "bulgarian":
            logging.info('Bg wiki ембедингите бяха записани успешно.')
        else:
            logging.info("Bg wiki embeddings were saved successfully.")

    def read_bg_wiki_chunks_and_embeddings(self):
        print("Reading bg wiki chunks and embeddings...")
        if self.menu_language == "bulgarian":
            logging.info(f"Четене на bg wiki откъси и ембединги (chunks and embeddings) от "
                         f"{self.bg_wiki_chunks_and_embeddings_csv_filename}...")
        else:
            logging.info(f"Reading bg wiki chunks and embeddings from "
                         f"{self.bg_wiki_chunks_and_embeddings_csv_filename}...")
        self.bg_wiki_chunks_and_embeddings_df = pd.read_csv(self.bg_wiki_chunks_and_embeddings_csv_filename,
                                                            dtype={'id': int, 'url': object, 'title': object,
                                                                   'chunk_text': object,
                                                                   'chunk_char_count': int, 'chunk_word_count': int,
                                                                   'chunk_token_count': float, 'embedding': object,
                                                                   'processed_chunk': object})
        print("bg_wiki_chunks_and_embeddings.csv file loaded successfully!")
        if self.menu_language == "bulgarian":
            logging.info('Bg wiki откъси и ембединги (chunks and embeddings) бяха заредени успешно.')
        else:
            logging.info("Bg wiki chunks and embeddings loaded successfully.")
        print("Creating a dictionary from bg wiki chunks and embeddings...")
        if self.menu_language == "bulgarian":
            logging.info('Създаване на dictionary от bg wiki chunks and embeddings...')
        else:
            logging.info("Creating a dictionary from bg wiki chunks and embeddings...")
        self.bg_wiki_chunks_and_embeddings_df["embedding"] = self.bg_wiki_chunks_and_embeddings_df["embedding"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" "))

        self.bg_wiki_chunks_and_embeddings_dict = self.bg_wiki_chunks_and_embeddings_df.to_dict(orient="records")
        print("Bg wiki chunks and embeddings dictionary was created successfully.")
        if self.menu_language == "bulgarian":
            logging.info('Bg wiki chunks and embeddings dictionary беше създаден успешно.')
        else:
            logging.info("Bg wiki chunks and embeddings dictionary was created successfully.")

    def write_bg_wiki_chunks_dict_to_file(self):
        if self.menu_language == "bulgarian":
            logging.info("Записване на bg wiki chunks and embeddings dictionary в " +
                         self.bg_wiki_chunks_and_embeddings_dict_pickle_filename + " file...")
        else:
            logging.info("Writing bg wiki chunks and embeddings dictionary to " +
                         self.bg_wiki_chunks_and_embeddings_dict_pickle_filename + " file...")

        with open(self.bg_wiki_chunks_and_embeddings_dict_pickle_filename, 'wb') as chunk_dict_file:
            pickle.dump(self.bg_wiki_chunks_and_embeddings_dict, chunk_dict_file, protocol=pickle.HIGHEST_PROTOCOL)

        if self.menu_language == "bulgarian":
            logging.info('Bg wiki chunks and embeddings dictionary беше записан успешно във файла.')
        else:
            logging.info("Bg wiki chunks and embeddings dictionary was written successfully to the file.")

    def write_bg_wiki_faiss_index(self):
        if self.menu_language == "bulgarian":
            logging.info("Създаване на bg wiki faiss index и негово записване в " + self.bg_wiki_faiss_store_filename +
                         " файл...")
        else:
            logging.info("Creating bg wiki faiss index and writing it to " + self.bg_wiki_faiss_store_filename +
                         " file...")

        if not self.bg_wiki_embeddings:
            _, _, self.bg_wiki_embeddings = self.read_dataset_files(
                chunk_dict_filename=self.bg_wiki_chunks_and_embeddings_dict_pickle_filename, faiss_store_filename="")
        d = self.bg_wiki_embeddings.shape[1]
        m = 64
        ef_construction = 64
        ef_search = 128
        index = faiss.IndexHNSWFlat(d, m)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(self.bg_wiki_embeddings.numpy())
        write_index(index, self.bg_wiki_faiss_store_filename)

        if self.menu_language == "bulgarian":
            logging.info('Bg wiki faiss index беше записан успешно.')
        else:
            logging.info("Bg wiki faiss index was written successfully.")

    def read_bg_wiki_faiss_index(self):
        if self.menu_language == "bulgarian":
            logging.info("Четене на bg wiki faiss index от " + self.bg_wiki_faiss_store_filename + " файл...")
        else:
            logging.info("Reading bg wiki faiss index from " + self.bg_wiki_faiss_store_filename + " file...")

        self.bg_wiki_faiss_index = read_index(self.bg_wiki_faiss_store_filename)

        if self.menu_language == "bulgarian":
            logging.info('Bg wiki faiss index беше зареден успешно.')
        else:
            logging.info("Bg wiki faiss index was loaded successfully.")

    def read_user_uploaded_faiss_index(self):
        if self.menu_language == "bulgarian":
            logging.info("Четене на user uploaded faiss index (faiss index на качените от потребителя файлове) "
                         "от " + self.uploaded_files_faiss_vector_db_filename + " файл...")
        else:
            logging.info("Reading user uploaded faiss index from " + self.uploaded_files_faiss_vector_db_filename +
                         " file...")

        self.user_uploaded_faiss_index = read_index(self.uploaded_files_faiss_vector_db_filename)

        if self.menu_language == "bulgarian":
            logging.info('User uploaded faiss index беше зареден успешно.')
        else:
            logging.info("User uploaded faiss index was loaded successfully.")

    def read_bg_wiki_database(self):
        print("Reading bg_wiki database...")
        if self.menu_language == "bulgarian":
            logging.info('Четене на bg wiki база данни...')
        else:
            logging.info("Reading bg wiki database...")

        self.bg_wiki_chunks_and_embeddings, self.bg_wiki_faiss_index, self.bg_wiki_embeddings = \
            self.read_dataset_files(chunk_dict_filename=self.bg_wiki_chunks_and_embeddings_dict_pickle_filename,
                                    faiss_store_filename=self.bg_wiki_faiss_store_filename)

        if self.menu_language == "bulgarian":
            logging.info('Bg wiki база данни беше заредена успешно.')
        else:
            logging.info("Bg wiki database loaded successfully.")

    def read_focus_news_database(self):
        print("Reading focus_news database...")
        if self.menu_language == "bulgarian":
            logging.info('Четене на focus news база данни...')
        else:
            logging.info("Reading focus news database...")

        self.focus_news_chunk_dict, self.focus_news_faiss_index, self.focus_news_embeddings = \
            self.read_dataset_files(chunk_dict_filename=self.focus_news_chunk_dict_filename,
                                    faiss_store_filename=self.focus_news_faiss_store_filename)

        if self.menu_language == "bulgarian":
            logging.info('Focus news база данни беше заредена успешно.')
        else:
            logging.info("Focus news database loaded successfully.")

    def unify_databases(self):
        if self.menu_language == "bulgarian":
            logging.info('Обединяване на bg wiki и focus news бази данни...')
        else:
            logging.info("Unifying bg wiki and focus news database...")

        self.db_chunks_and_embeddings = self.bg_wiki_chunks_and_embeddings
        self.db_chunks_and_embeddings.extend(self.focus_news_chunk_dict)
        self.db_embeddings = torch.cat((self.bg_wiki_embeddings, self.focus_news_embeddings))
        d = self.db_embeddings.shape[1]
        m = 64
        ef_construction = 64
        ef_search = 128
        self.index_faiss_db = faiss.IndexHNSWFlat(d, m)
        self.index_faiss_db.hnsw.efConstruction = ef_construction
        self.index_faiss_db.hnsw.efSearch = ef_search
        self.index_faiss_db.add(self.db_embeddings.numpy())
        write_index(self.index_faiss_db, self.faiss_store_filename)
        if os.path.exists(self.db_bm25s_store_filename):
            shutil.rmtree(self.db_bm25s_store_filename)
        with open(self.chunk_dict_filename, 'wb') as chunk_dict_file:
            pickle.dump(self.db_chunks_and_embeddings, chunk_dict_file, protocol=pickle.HIGHEST_PROTOCOL)

        if self.menu_language == "bulgarian":
            logging.info("Базите данни бяха обединени успешно. Обединените откъси и ембединги (chunks and embeddings) "
                         "бяха записани в " + self.chunk_dict_filename + " файл.")
        else:
            logging.info("Databases unified successfully. The unified chunks and embeddings were written to " +
                         self.chunk_dict_filename + " file.")

    def read_database(self):
        print("Reading database...")
        if self.menu_language == "bulgarian":
            logging.info('Четене на базата данни...')
        else:
            logging.info("Reading database...")

        try:
            self.db_chunks_and_embeddings, self.index_faiss_db, self.db_embeddings = self.read_dataset_files()

        except Exception as e:
            logging.exception(f"ERROR: {e}", stack_info=True)
            print(f"ERROR: {e}")
            if self.menu_language == "bulgarian":
                logging.info("Проблем с четенето на базата данни! Системата може да не функционира правилно!")
            else:
                logging.info("PROBLEM with the reading of the database! The system may not function properly!")
            print("PROBLEM with the reading of the database! The system may not function properly!")
        else:
            if self.menu_language == "bulgarian":
                logging.info('Базата данни беше заредена успешно.')
            else:
                logging.info("Database loaded successfully.")
            print("Database loaded successfully.")

    def download_bg_wiki_database_and_update(self, date="20250120"):
        print("Downloading bg wiki database and updating the default database...")
        if self.menu_language == "bulgarian":
            logging.info("Изтегляне на bg wiki база данни и обновяване на основната база данни...")
        else:
            logging.info("Downloading bg wiki database and updating the default database...")
        try:
            self.load_bg_wiki_dataset_to_pickle(date=date)
            self.read_bg_wiki_dataset_from_pickle()
            self.create_bg_wiki_chunks()
            self.read_bg_wiki_chunks_filtered()
            self.embed_bg_wiki_chunks_filtered()
            self.read_bg_wiki_chunks_and_embeddings()
            self.write_bg_wiki_chunks_dict_to_file()
            self.write_bg_wiki_faiss_index()
            self.read_bg_wiki_database()
            if not (self.focus_news_chunk_dict and self.focus_news_faiss_index and self.focus_news_embeddings):
                self.read_focus_news_database()
            self.unify_databases()

        except Exception as e:
            print(f"EXCEPTION: {e}")
            if self.menu_language == "bulgarian":
                logging.info("ГРЕШКА! ПРОБЛЕМ с изтеглянето на bg wiki база данни и обновяването на основната база "
                             "данни! Проверете си интернет връзката, защото тя е нужна за изтеглянето на bg wiki "
                             "файловете. Проверете също датата за bg wiki файловете, избрана от Вас за обновяването - "
                             "може да бъде невалидна или може да няма налични bg wiki файлове за тази дата.")
            else:
                logging.info("ERROR! PROBLEM with the download of bg wiki database and the update of the default "
                             "database! Check your internet connection because it is needed for the download of the bg "
                             "wiki files. Check also the date of the bg wiki files that you have chosen for the update "
                             "- it may be invalid or bg wiki files may not be available for this date.")
            print("ERROR! PROBLEM with the download of bg wiki database and the update of the default "
                  "database! Check your internet connection because it is needed for the download of the bg "
                  "wiki files. Check also the date of the bg wiki files that you have chosen for the update "
                  "- it may be invalid or bg wiki files may not be available for this date.")
        else:
            print("Bg wiki database downloaded and default database updated successfully.")
            if self.menu_language == "bulgarian":
                logging.info("Bg wiki база данни беше изтеглена и основната база данни беше обновена успешно.")
            else:
                logging.info("Bg wiki database downloaded and default database updated successfully.")

    def update_both_databases(self, bg_wiki_dump_date="20250120"):
        self.read_focus_news_json_files()
        self.download_bg_wiki_database_and_update(date=bg_wiki_dump_date)

    def retrieve_relevant_articles(self, query, topn_articles=3):
        query_embedding = torch.tensor(self.embedding_model.encode(query, max_length=8192)['dense_vecs'],
                                       dtype=torch.float32).cpu().numpy()
        scores, indices = self.faiss_index.search(np.array([query_embedding]), k=topn_articles)
        return scores, indices

    def show_most_relevant_articles_to_query(self, query, topn_articles=3, open_url=False):
        print(f"Query: {query}\n")
        scores, indices = self.retrieve_relevant_articles(query=query, topn_articles=topn_articles)
        print("Most relevant articles:")
        is_top_result = True
        for score, idx in zip(scores[0], indices[0]):
            print(f"Score: {score:.4f}")
            print(f"Title: {self.chunks_and_embeddings[idx]['title']}")
            print(f"Url: {self.chunks_and_embeddings[idx]['url']}")
            if is_top_result and open_url:
                webbrowser.open(self.chunks_and_embeddings[idx]['url'])
            is_top_result = False
            print("\n")

    def calc_bm25_scores(self, query, topn_articles=3, use_uploaded_files_db=False):
        if self.use_default_db and not use_uploaded_files_db:
            filename = self.db_bm25s_store_filename
        else:
            filename = self.uploaded_files_bm25s_store_filename
        if not os.path.exists(filename):
            processed_chunks = [row['processed_chunk'] for row in self.chunks_and_embeddings]
            retriever = bm25s.BM25()
            retriever.index(bm25s.tokenize(processed_chunks))
            retriever.save(filename)
        retriever = bm25s.BM25.load(filename, mmap=True)
        cleaned_query = self.preprocess_text(query)
        lemmatized_query = self.lemmatize_text(cleaned_query)
        if len(lemmatized_query) == 0:
            lemmatized_query = query
        corpus_indices = [idx for idx in range(len(self.chunks_and_embeddings))]
        if len(self.chunks_and_embeddings) < topn_articles:
            topn_articles = len(self.chunks_and_embeddings)
        indices, scores = retriever.retrieve(bm25s.tokenize(lemmatized_query), corpus=corpus_indices, k=topn_articles)
        return indices, scores

    def load_llm(self):
        print("Loading LLM...")
        if self.menu_language == "bulgarian":
            logging.info(f'Зареждане на LLM ({self.model_file})...')
        else:
            logging.info(f"Loading LLM ({self.model_file})...")

        try:
            model_path = hf_hub_download(self.model_gguf_name, filename=self.model_file)
            n_gpu_layers = -1 if self.device == 'cuda' else 0
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,  # Number of model layers to offload to GPU, -1 for all layers to GPU
                n_ctx=8192,
                penalize_nl=False,
            )
            self.llm.verbose = False

        except Exception as e:
            print(f"ERROR: {e}")
            logging.exception(f"ERROR: {e}", stack_info=True)
            if self.menu_language == "bulgarian":
                logging.info("ПРОБЛЕМ със зареждането на LLM! Системата НЯМА да работи правилно! Ако за първи път "
                             "използвате системата, проверете интернет връзката си. Тя е необходима, за да се "
                             "изтегли LLM. Оправете интернет връзката си и рестартирайте приложението. Ако това не "
                             "помогне, инсталирайте наново приложението.")
            else:
                logging.info("PROBLEM with the loading of the LLM! The system will NOT work properly! If you use the "
                             "system for the first time, check your internet connection. It is needed for the download "
                             "of the LLM. Sort out your internet connection and restart the app. If this does not "
                             "help, then reinstall the app.")
            print("PROBLEM with the loading of the LLM! The system will NOT work properly! If you use the "
                  "system for the first time, check your internet connection. It is needed for the download "
                  "of the LLM. Sort out your internet connection and restart the app. If this does not "
                  "help, then reinstall the app.")
        else:
            if self.menu_language == "bulgarian":
                logging.info('LLM беше зареден успешно.')
            else:
                logging.info("LLM loaded successfully.")
            print("LLM loaded successfully.")

    def prompt_formatter(self, query: str, context_items: list[dict]) -> [str, str]:
        context = "- " + "\n\n- ".join([element["chunk_text"] for element in context_items])

        bg_coefficient = in_target_language(query, lang='bg')
        en_coefficient = in_target_language(query, lang='en')
        if bg_coefficient >= en_coefficient:
            query_language = "български"
        else:
            query_language = "английски"

        if self.history_str == "":
            self.history_str = "Това е началото на разговора."

        prompt = f""" 
                 # Системни инструкции
                 Ти си полезен асистент. Генерирай отговора на заявката на същия език, на който е заявката.
                 Използвай дадения контекст, за да отговориш на заявката.
                 Може да използваш и дадения разговор между теб (асистента) и потребителя до момента.
                 Дай си време да помислиш и намери подходящи пасажи от дадения контекст, отнасящи се за заявката.
                 Използвай намерените пасажи, за да отговориш на заявката.
                 Отговаряй на базата на достоверна информация, която намираш в контекста.
                 Ако не намираш информация за заявката в контекста, кажи, че не намираш информация за заявката и
                 не отговаряй.
                 Не си измисляй несъществуващи факти. Не отговаряй на всяка цена, а само ако намериш информация
                 за проблема в дадения контекст. Трябва да откриеш информация в контекста и само тогава да 
                 отговориш. Целта е отговорът ти да е на базата дадения контекст. Ако в контекста няма достатъчно
                 информация за въпроса, сподели, че липсва достатъчно информация като отговор и не съчинявай
                 неподкрепени с факти твърдения. Много важно е да си фактически коректен и отговорът ти да бъде на 
                 базата на информация в дадения контекст.
                 
                 # Контекст (използвай контекста, за да отговориш на заявката)
                 {context}
                 
                 # Досегашен разговор между теб (асистента) и потребителя до момента
                 {self.history_str}
                 
                 # Заявка на потребителя (отговори на нея на {query_language} език)
                 {query}
                 """
        return prompt

    def reciprocal_rank_fusion(self, query, translate_query=False, k=60, topn_articles=3, use_uploaded_files_db=False):
        if self.use_default_db and self.use_uploaded_db:
            rrf_topn_threshold = 25
        else:
            rrf_topn_threshold = 50
        if len(self.chunks_and_embeddings) >= rrf_topn_threshold:
            topn_relevant_articles = rrf_topn_threshold
        else:
            topn_relevant_articles = len(self.chunks_and_embeddings)

        vs_res_scores, vs_res_indices = self.retrieve_relevant_articles(query=query,
                                                                        topn_articles=topn_relevant_articles)
        vs_scores = [vs_res_scores[0][i] for i in range(len(vs_res_scores[0]))]

        vs_indices = [vs_res_indices[0][i] for i in range(len(vs_res_indices[0]))]

        bm25_res_indices, bm25_res_scores = self.calc_bm25_scores(query, topn_articles=topn_relevant_articles,
                                                                  use_uploaded_files_db=use_uploaded_files_db)
        bm25_scores = [bm25_res_scores[0, i] for i in range(bm25_res_scores.shape[1])]

        bm25_indices = [bm25_res_indices[0, i] for i in range(bm25_res_indices.shape[1])]

        relevant_indices_lists = [vs_indices, bm25_indices]

        if translate_query:
            tr_query = self.translate_query(query)

            print(f"Translated: {tr_query}")
            if self.menu_language == "bulgarian":
                logging.info(f"Преведено: {tr_query}")
            else:
                logging.info(f"Translated: {tr_query}")

            tr_vs_res_scores, tr_vs_res_indices = self.retrieve_relevant_articles(query=tr_query,
                                                                                  topn_articles=topn_relevant_articles)
            tr_vs_scores = [tr_vs_res_scores[0][i] for i in range(len(tr_vs_res_scores[0]))]

            tr_vs_indices = [tr_vs_res_indices[0][i] for i in range(len(tr_vs_res_indices[0]))]

            tr_bm25_res_indices, tr_bm25_res_scores = self.calc_bm25_scores(tr_query,
                                                                            topn_articles=topn_relevant_articles,
                                                                            use_uploaded_files_db=use_uploaded_files_db)
            tr_bm25_scores = [tr_bm25_res_scores[0, i] for i in range(tr_bm25_res_scores.shape[1])]

            tr_bm25_indices = [tr_bm25_res_indices[0, i] for i in range(tr_bm25_res_indices.shape[1])]

            bm25_scores.extend(tr_bm25_scores)
            bm25_indices.extend(tr_bm25_indices)
            vs_scores.extend(tr_vs_scores)
            vs_indices.extend(tr_vs_indices)

            final_vs_indices = [index for _, index in sorted(zip(vs_scores, vs_indices))]
            final_bm25_indices = [index for _, index in sorted(zip(bm25_scores, bm25_indices), reverse=True)]

            relevant_indices_lists = [final_vs_indices, final_bm25_indices]

        reranked_indices, reranked_scores = self.reciprocal_rank_fusion_raw(indices_list=relevant_indices_lists, k=k,
                                                                            topn_articles=topn_articles)

        return reranked_indices, reranked_scores

    def show_reciprocal_rank_fusion(self, query, k=60, topn_articles=3):
        r_indices, r_scores = self.reciprocal_rank_fusion(query, k=k, topn_articles=topn_articles)
        print()
        r = 0
        for ind in r_indices:
            r += 1
            sc = float("{:.4f}".format(r_scores[r - 1]))
            print(
                f"Rank: {r} (score: {sc}) {self.chunks_and_embeddings[ind]['url']}\nChunk: "
                f"{self.chunks_and_embeddings[ind]['chunk_text']}")

    def is_context_passage_relevant_to_query(self, query, context_passage, print_result=False):
        prompt = f""" 
                     # Системни инструкции  
                     Отговори дали в някаква част от контекста се съдържа полезна информация, отнасяща се по някакъв 
                     начин към зададената заявка или не. 
                     Отговори с Да, ако някъде в контекста има полезна информация, свързана със заявката. 
                     Отговори с Не, ако никоя част от контекста не е свързана по никакъв начин със заявката.
                     Отговори само с Да или Не.

                     # Заявка
                     {query}
                     
                     # Контекст
                     {context_passage}
                 """

        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]
        llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                       top_k=5, temperature=0.3)
        llm_response_text = llm_response["choices"][0]["message"]["content"]

        if llm_response_text[:2] == "Не" or llm_response_text[:2] == "не" or llm_response_text[:2] == "No" or \
                llm_response_text[:2] == "no":
            if print_result:
                print("NOT relevant:")
                print(context_passage)
                if self.menu_language == "bulgarian":
                    logging.info("НЕрелевантен:")
                else:
                    logging.info("NOT relevant:")
                logging.info(context_passage)
            return False
        else:
            if print_result:
                print("RELEVANT:")
                print(context_passage)
                if self.menu_language == "bulgarian":
                    logging.info("РЕЛЕВАНТЕН:")
                else:
                    logging.info("RELEVANT:")
                logging.info(context_passage)
            return True

    def summarize_too_long_query(self, query):
        prompt = f""" 
                     # Системни инструкции  
                     Обобщи твърде дългата заявка на потребителя, като запазиш само съществената част на 
                     заявката. Важно е да запазиш какво потребителя иска да бъде направено, какво търси той
                     с тази заявка.

                     # Задача
                     Обобщи в рамките на 1000 tokens максимум тази заявка на потребителя:
                     {query}
                 """
        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]
        llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                       top_k=5, temperature=0.3)
        llm_response_text = llm_response["choices"][0]["message"]["content"]
        return llm_response_text

    def modify_too_long_prompt(self, prompt_tokens_length, query, relevant_context_elements):
        prompt = self.prompt_formatter(query=query, context_items=relevant_context_elements)
        while prompt_tokens_length > 8192 and len(relevant_context_elements) > 1:
            relevant_context_elements = relevant_context_elements[:-1]
            prompt = self.prompt_formatter(query=query, context_items=relevant_context_elements)
            prompt_tokens_length = len(self.llm.tokenize(prompt.encode('utf-8')))

        if self.menu_language == "bulgarian":
            logging.info("Промптът (prompt) надхвърля лимита за дължина на контекста за LLM. "
                         "Затова, някои от по-малко релевантните откъси (използвани за даване на контекст) "
                         "бяха премахнати от промпта към LLM.")
        else:
            logging.info("The prompt became too long and could not fit the context length limit of the LLM. "
                         "Therefore, some of the less relevant chunks that provide context were removed from "
                         "the prompt.")

        if prompt_tokens_length > 8192:
            query_tokens_length = len(self.llm.tokenize(query.encode('utf-8')))
            if query_tokens_length > 8000:
                return "", []
            else:
                query = self.summarize_too_long_query(query)
                query_tokens_length = len(self.llm.tokenize(query.encode('utf-8')))
                if query_tokens_length > 2000:
                    return "", []
                if self.menu_language == "bulgarian":
                    logging.info(f"Заявката е твърде голяма и трябваше да бъде обобщена, за да се побере в "
                                 f"контекстния лимит на LLM. Ето обобщената заявка: \n {query}")
                else:
                    logging.info(f"The query is too long and needed to be summarised in order to fit the context "
                                 f"length limit of the LLM. Here is the summarised query: \n {query}")
                prompt = self.prompt_formatter(query=query, context_items=relevant_context_elements)
        return prompt, relevant_context_elements

    def generate_similar_queries(self, query, n_similar_queries=4):
        bg_coefficient = in_target_language(query, lang='bg')
        en_coefficient = in_target_language(query, lang='en')
        if bg_coefficient >= en_coefficient:
            task = f"Генерирай {n_similar_queries} заявки на български, подобни на тази заявка: {query}"
        else:
            task = f"Generate {n_similar_queries} queries in English, similar to this query: {query}"
        prompt = f""" 
                     # Системни инструкции
                     Ти си полезен асистент, който генерира няколко заявки, подобни на
                     първоначалната заявка. Темата на генерираните от теб заявки трябва да е като темата на
                     първоначалната заявка. Разликата с първоначалната заявка трябва да бъде малка.
                     Запази контекста от първоначалната заявка при генерирането на подобните заявки.
                     Запази също и предмета на питане от първоначалната заявка. Само модифицирай леко 
                     допълнителните елементи от въпроса/заявката. Нека това, за което се пита, да присъства
                     и в генерираните заявки.
                     В генерираните заявки запази имена, числа, дати, години и специфични уточнения, които са в 
                     първоначалната заявка.
                     Нека генерираните заявки да са номинирани с число и точка (1. 2. 3.). 
                     Нека генерираните заявки да са на същия език, на който е първоначалната заявка.
                     В новите заявки не бива да променяш темата, за която се пита, а само да добавиш допълнителни
                     нюанси към заявката. Генерираните заявки трябва да са свързани напълно с темата на първоначалната
                     заявка. Важно е да запазиш същите имена, години, теми, които са в първоначалната заявка.
                     Най-важното е да не променяш темата на заявката и да запазиш конкретното нещо, за което се пита 
                     или се търси, и в новите, генерирани от теб заявки.
                             
                     # Задача
                     {task} 
                     """

        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]

        llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                       top_k=40, temperature=0.5)
        llm_response_text = llm_response["choices"][0]["message"]["content"]
        return llm_response_text

    def rag_fusion(self, query, n_similar_queries=4, print_queries=False, k=60, translate_query=False, topn_articles=3,
                   use_uploaded_files_db=False):
        llm_answer = self.generate_similar_queries(query, n_similar_queries=n_similar_queries)

        if print_queries:
            print("RAG Fusion option activated.")
            if self.menu_language == "bulgarian":
                logging.info("RAG Fusion опция е активирана.")
            else:
                logging.info("RAG Fusion option activated.")

            print("The generated queries similar to the original query (used for improving the extraction "
                  "of relevant resourcers): ")
            if self.menu_language == "bulgarian":
                logging.info("Генерираните подобни заявки на първоначалната заявка "
                             "(използвани са за подобряване на извличането на релевантни ресурси):")
            else:
                logging.info("The generated queries similar to the original query (used for improving the extraction "
                             "of relevant resourcers): ")

            print(llm_answer)
            logging.info(llm_answer)

        queries_list = [line[3:] for line in llm_answer.splitlines()]
        rank_fusion_indices = []

        for q in queries_list:
            q_ranked_indices, _ = self.reciprocal_rank_fusion(q, k=k, translate_query=translate_query,
                                                              topn_articles=topn_articles,
                                                              use_uploaded_files_db=use_uploaded_files_db)

            rank_fusion_indices.append(q_ranked_indices)
        return rank_fusion_indices

    def ask_query(self, query, use_rag_fusion=False, use_hyde=False, use_generated_subqueries=False,
                  use_relevant_resource_llm_check=False, translate_query=False,
                  temperature=0.7, max_answer_tokens=42, return_llm_response=True, return_context_elements=True,
                  return_urls=True, topn_articles=3, print_steps=False, use_uploaded_files_db=False):

        if translate_query:
            if self.menu_language == "bulgarian":
                logging.info("Опцията за превод на заявки е активирана. Потребителската заявка и всички генерирани"
                             "заявки в процеса на извличане от ресурсите (активирани от 'Опции' в менюто) ще бъдат "
                             "преведени, за да се позволи търсене от ресурси на български и на английски. Ако Вашите "
                             "качени файлове са само на един език и той съвпада с езика, на който е написана Вашата "
                             "заявка, то изключете тази опция в менюто.")
            else:
                logging.info("Translate query option activated. The user query and all generated queries in the "
                             "retrieval process (activated from 'Options' in the menu) will be translated, so that "
                             "the searching from resources on Bulgarian and English language is enabled. If your "
                             "uploaded files are in one language only and it is the same language on which your query "
                             "is written, then turn off this option in the menu.")

        if self.use_default_db and self.use_uploaded_db:
            rrf_topn_threshold = 10
        else:
            rrf_topn_threshold = 20

        query_ranked_indices, query_fused_scores = self.reciprocal_rank_fusion(query, translate_query=translate_query,
                                                                               topn_articles=rrf_topn_threshold,
                                                                               use_uploaded_files_db=
                                                                               use_uploaded_files_db)
        rank_fusion_indices = [query_ranked_indices]

        if use_rag_fusion:
            rank_fusion_indices += self.rag_fusion(query, n_similar_queries=4, print_queries=True, k=60,
                                                   translate_query=translate_query, topn_articles=rrf_topn_threshold,
                                                   use_uploaded_files_db=use_uploaded_files_db)

        if use_hyde:
            llm_answer = self.ask_llm(query, print_query=False, print_response=False)
            if print_steps:
                print(
                    "Hyde approach option was triggered. Here is the llm response on the query that is used to "
                    "find relevant resources for the RAG system:")
                if self.menu_language == "bulgarian":
                    logging.info("Hyde подходът е активиран. Ето отговорът на LLM на заявката, който е използван за "
                                 "подобряване намирането на релевантни източници за заявката:")
                else:
                    logging.info("Hyde approach option was activated. Here is the LLM response on the query that is "
                                 "used to improve the finding of relevant resources for the query:")
                print(llm_answer)
                logging.info(llm_answer)
            hyde_ranked_indices, _ = self.reciprocal_rank_fusion(llm_answer, translate_query=translate_query,
                                                                 topn_articles=rrf_topn_threshold,
                                                                 use_uploaded_files_db=use_uploaded_files_db)
            rank_fusion_indices.append(hyde_ranked_indices)

        if use_generated_subqueries:
            subqueries = self.generate_subqueries(query)
            if print_steps:
                print("Generate subqueries option triggered. Here are the generated subqueries:")
                if self.menu_language == "bulgarian":
                    logging.info("Генерирането на подзаявки е активирано. Ето генерираните подзаявки:")
                else:
                    logging.info("Generate subqueries option activated. Here are the generated subqueries:")
                print(subqueries)
                logging.info(subqueries)

            generated_subqueries_list = [line[3:] for line in subqueries.splitlines()]
            for sub_query in generated_subqueries_list:
                q_ranked_indices, _ = self.reciprocal_rank_fusion(sub_query, translate_query=translate_query,
                                                                  topn_articles=rrf_topn_threshold,
                                                                  use_uploaded_files_db=use_uploaded_files_db)
                rank_fusion_indices.append(q_ranked_indices)

        if len(rank_fusion_indices) > 1:
            final_ranked_indices, final_fused_scores = self.reciprocal_rank_fusion_raw(rank_fusion_indices,
                                                                                       topn_articles=rrf_topn_threshold)
        else:
            final_ranked_indices = query_ranked_indices
            final_fused_scores = query_fused_scores

        relevant_chunks_and_embeddings = [self.chunks_and_embeddings[i] for i in final_ranked_indices]
        relevant_chunks = [element['chunk_text'] for element in relevant_chunks_and_embeddings]

        reranked_chunks = self.reranking_model.rank(query, relevant_chunks, return_documents=True,
                                                    convert_to_tensor=True)
        reranked_elements = []
        reranking_scores = []
        for ranking in reranked_chunks:
            reranked_elements.append(relevant_chunks_and_embeddings[ranking['corpus_id']])
            reranking_scores.append(ranking['score'])

        relevant_context_elements = []

        for i, element in enumerate(reranked_elements):
            element['score'] = reranking_scores[i]
            if use_relevant_resource_llm_check:
                if self.is_context_passage_relevant_to_query(query=query, context_passage=element['chunk_text']):
                    relevant_context_elements.append(element)
            else:
                relevant_context_elements.append(element)
            if len(relevant_context_elements) == topn_articles:
                break

        if not relevant_context_elements:
            relevant_context_elements.append(relevant_chunks_and_embeddings[0])

        relevant_urls = self.retrieve_relevant_urls(relevant_context_elements)

        if return_llm_response:
            prompt = self.prompt_formatter(query=query, context_items=relevant_context_elements)

            prompt_tokens_length = len(self.llm.tokenize(prompt.encode('utf-8')))
            if prompt_tokens_length > 8192:
                prompt, relevant_context_elements = \
                    self.modify_too_long_prompt(prompt_tokens_length=prompt_tokens_length, query=query,
                                                relevant_context_elements=relevant_context_elements)
                relevant_urls = self.retrieve_relevant_urls(relevant_context_elements)

                if not prompt:
                    if self.menu_language == "bulgarian":
                        logging.info(
                            "Твърде голяма заявка. Не може да бъде преработена. Опитайте се да напишете доста "
                            "по-кратка заявка.")
                        return "Твърде голяма заявка. Не може да бъде преработена. Опитайте се да напишете " \
                               "доста по-кратка заявка.", [], []
                    else:
                        logging.info(
                            "Too long query. It cannot be processed. Try writing a much smaller query.")
                        return "Too long query. It cannot be processed. Try writing a much smaller query.", [], []

            dialogue_template = [
                {"role": "user",
                 "content": prompt}
            ]

            llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                           top_k=max_answer_tokens, temperature=temperature)

            llm_response_text = llm_response["choices"][0]["message"]["content"]
        else:
            llm_response_text = ""

        if return_urls:
            if return_context_elements:
                return llm_response_text, relevant_context_elements, relevant_urls
            else:
                return llm_response_text, relevant_urls
        else:
            if return_context_elements:
                return llm_response_text, relevant_context_elements
            else:
                return llm_response_text

    def ask_rag(self, query, use_rag_fusion=False, use_hyde=False, use_generated_subqueries=False,
                use_relevant_resource_llm_check=False, translate_query=False,
                print_query=False, print_response=False, print_urls=False, print_context_passages=False,
                print_steps=True, temperature=0.7, max_answer_tokens=42, topn_articles=3):

        if len(self.chunks_and_embeddings) < topn_articles:
            topn_articles = len(self.chunks_and_embeddings)

        return_llm_response = False if self.use_default_db and self.use_uploaded_db else True

        llm_response_text, relevant_context_elements, relevant_urls = \
            self.ask_query(query=query, use_rag_fusion=use_rag_fusion, use_hyde=use_hyde,
                           use_generated_subqueries=use_generated_subqueries,
                           use_relevant_resource_llm_check=use_relevant_resource_llm_check,
                           translate_query=translate_query, temperature=temperature,
                           max_answer_tokens=max_answer_tokens, return_llm_response=return_llm_response,
                           return_context_elements=True, return_urls=True, topn_articles=topn_articles,
                           print_steps=print_steps, use_uploaded_files_db=False)

        if self.use_default_db and self.use_uploaded_db:
            self.chunks_and_embeddings = self.user_uploaded_chunks_and_embeddings
            self.faiss_index = self.user_uploaded_faiss_index
            self.embeddings = self.user_uploaded_embeddings
            uploaded_llm_response_text, uploaded_relevant_context_elements, uploaded_relevant_urls = \
                self.ask_query(query=query, use_rag_fusion=use_rag_fusion, use_hyde=use_hyde,
                               use_generated_subqueries=use_generated_subqueries,
                               use_relevant_resource_llm_check=use_relevant_resource_llm_check,
                               translate_query=translate_query, temperature=temperature,
                               max_answer_tokens=max_answer_tokens, return_llm_response=return_llm_response,
                               return_context_elements=True, return_urls=True, topn_articles=topn_articles,
                               print_steps=print_steps, use_uploaded_files_db=True)

            relevant_context_elements.extend(uploaded_relevant_context_elements)
            relevant_context_elements = sorted(relevant_context_elements, key=lambda el: el['score'], reverse=True)
            relevant_context_elements = relevant_context_elements[:topn_articles]
            relevant_urls = self.retrieve_relevant_urls(relevant_context_elements)

            prompt = self.prompt_formatter(query=query, context_items=relevant_context_elements)

            prompt_tokens_length = len(self.llm.tokenize(prompt.encode('utf-8')))
            if prompt_tokens_length > 8192:
                prompt, relevant_context_elements = \
                    self.modify_too_long_prompt(prompt_tokens_length=prompt_tokens_length, query=query,
                                                relevant_context_elements=relevant_context_elements)
                relevant_urls = self.retrieve_relevant_urls(relevant_context_elements)

                if not prompt:
                    if self.menu_language == "bulgarian":
                        logging.info(
                            "Твърде голяма заявка. Не може да бъде преработена. Опитайте се да напишете доста "
                            "по-кратка заявка.")
                        return "Твърде голяма заявка. Не може да бъде преработена. Опитайте се да напишете " \
                               "доста по-кратка заявка."
                    else:
                        logging.info(
                            "Too long query. It cannot be processed. Try writing a much smaller query.")
                        return "Too long query. It cannot be processed. Try writing a much smaller query."

            dialogue_template = [
                {"role": "user",
                 "content": prompt}
            ]

            llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                           top_k=max_answer_tokens, temperature=temperature)

            llm_response_text = llm_response["choices"][0]["message"]["content"]

            self.chunks_and_embeddings = self.db_chunks_and_embeddings
            self.faiss_index = self.index_faiss_db
            self.embeddings = self.db_embeddings

        if self.evaluation_mode:
            bg_coefficient = in_target_language(query, lang='bg')
            en_coefficient = in_target_language(query, lang='en')
            if bg_coefficient >= en_coefficient:
                query_language = "bg"
            else:
                query_language = "en"

            relevant_contexts = []
            for element in relevant_context_elements:
                relevant_contexts.append(element['chunk_text'])

            new_evaluation_df_row = [query, llm_response_text, relevant_contexts, -1, "empty", [-1, -1, -1],
                                     [-1, -1, -1], use_rag_fusion, use_hyde, use_generated_subqueries,
                                     use_relevant_resource_llm_check, topn_articles, query_language]
            self.evaluation_df.loc[len(self.evaluation_df)] = new_evaluation_df_row

        if print_query:
            print(f"Query: {query}")
            if self.menu_language == "bulgarian":
                logging.info("Заявка: " + query)
            else:
                logging.info("Query: " + query)
        if print_response:
            print("RAG response:\n")
            if self.menu_language == "bulgarian":
                logging.info("RAG отговор: ")
            else:
                logging.info("RAG response: ")
            print(llm_response_text)
            logging.info(llm_response_text)

        if print_urls:
            print("\nRelevant resources:")
            if self.menu_language == "bulgarian":
                logging.info("Релевантни източници: " + query)
            else:
                logging.info("Relevant resources: ")
            for url in relevant_urls:
                print(url)
                logging.info(url)
        if print_context_passages:
            print("\nRelevant chunks for context:\n")
            if self.menu_language == "bulgarian":
                logging.info("Релевантни откъси за контекст: \n")
            else:
                logging.info("Relevant chunks for context: \n")
            i = 1
            for element in relevant_context_elements:
                print(f"Chunk {i}:")
                print(element['chunk_text'])
                print(f"Resource: {element['url']}\n")
                if self.menu_language == "bulgarian":
                    logging.info(f"Откъс {i}:")
                    logging.info(element['chunk_text'])
                    logging.info(f"Източник: {element['url']}\n")
                else:
                    logging.info(f"Chunk {i}:")
                    logging.info(element['chunk_text'])
                    logging.info(f"Resource: {element['url']}\n")
                i += 1
        return llm_response_text, relevant_urls

    def ask_llm(self, query, use_chat_history=False, print_query=False, print_response=False,
                temperature=0.7, max_answer_tokens=42):
        bg_coefficient = in_target_language(query, lang='bg')
        en_coefficient = in_target_language(query, lang='en')
        if bg_coefficient >= en_coefficient:
            query_language = "български"
        else:
            query_language = "английски"
        if use_chat_history and self.history_str:
            prompt = f""" 
                         # Системни инструкции
                         Ти си полезен асистент, който отговаря на заявка на потребителя на базата на
                         досегашния разговор. Намери информация в досегашния разговор и отговори на заявката.
                         Нека отговорът се състои от поне няколко изречения.
                         
                         # Досегашен разговор между потребителя и асистента
                         {self.history_str} 
                         
                         # Заявка на потребителя (отговори на нея на {query_language} език)
                         {query}
                    """
        elif use_chat_history and not self.history_str:
            prompt = f""" 
                         # Системни инструкции
                         Ти си полезен асистент, който отговаря на заявки на потребителя.
                         Това е началото на разговора. Представи се и попитай потребителя с какво можеш да му
                         помогнеш.
                         
                         # Заявка на потребителя (отговори на нея на {query_language} език)
                         {query}
                     """
        elif not use_chat_history and self.history_str:
            prompt = f""" 
                         # Системни инструкции
                         Ти си полезен асистент, който отговаря на заявка на потребителя.
                         Нека отговорът се състои от поне няколко изречения.
                         
                         # Досегашен разговор между теб (асистента) и потребителя
                         {self.history_str} 
                         
                         # Заявка на потребителя (отговори на нея на {query_language} език)
                         {query}
                      """
        else:
            prompt = f""" 
                         # Системни инструкции
                         Ти си полезен асистент, който отговаря на заявки на потребителя.
                         Нека отговорът се състои от поне няколко изречения.
                         
                         # Заявка на потребителя (отговори на нея на {query_language} език)
                         {query} 
                     """

        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]
        llm_response = self.llm.create_chat_completion(messages=dialogue_template, top_k=max_answer_tokens,
                                                       temperature=temperature)
        llm_response_text = llm_response["choices"][0]["message"]["content"]
        if print_query:
            print(f"Query: {query}")
            if self.menu_language == "bulgarian":
                logging.info("Заявка: " + query)
            else:
                logging.info("Query: " + query)
        if print_response:
            print(f"Response: {llm_response_text}")
            if self.menu_language == "bulgarian":
                logging.info("Response: " + llm_response_text)
            else:
                logging.info("Отговор: " + llm_response_text)
        return llm_response_text

    def generate_subqueries(self, query):
        bg_coefficient = in_target_language(query, lang='bg')
        en_coefficient = in_target_language(query, lang='en')
        if bg_coefficient >= en_coefficient:
            query_language = "български"
        else:
            query_language = "английски"
        prompt = f"""
                     # Системни инструкции 
                     Ти си полезен асистент, който разделя първоначалната заявка на подзаявки (разделяш 
                     изречението на съставните му изречения).
                     Важно е обаче всяка от подзаявките да съдържа предмета на питане (тоест това, за което се пита
                     да бъде ясно формулирано в подзаявката). Целта е всяка от подзаявките да си бъде самодостатъчна, 
                     тоест да е като самостоятелен въпрос, в който се знае за какво се пита. 
                     Добави във всяка подзаявка достатъчно подробна информация за предмета на търсене от потребителя.
                     Замени непреките допълнения (му, го, ѝ, т.н) с предмета на търсене в заявката.
                     Не бива да има непреки допълнения в генерираните от теб подзаявки. Подзаявка, генерирана от теб 
                     трябва да е напълно самодостатъчна и изчерпателна, не трябва да има нищо недоизказано в нея.
                     Ако изречение в заявката е просто (не може да се раздели), запази го без изобщо да го 
                     модифицираш. Ако изречението в заявката може да се раздели и е сложно съставно, раздели го 
                     на няколко прости изречения, като добавиш предмета на питане (тоест това, за което се пита).
                     Повтори този процес за всички изречения от заявката.
                     Нека генерираните подзаявки да са номинирани с числа. 
                     Нека генерираните подзаявки да са на същия език, на който е първоначалната заявка. 
                     
                     # Пример
                     Заявката "Кога се е формирала Стара Велика България и къде е била разположена? Кой е бил нейният 
                     основател и с какво е известен той? Коя е била столицата ѝ?" се
                     превръща в следните примерни подзаявки:
                     1. Кога се е формирала Стара Велика България?
                     2. Къде е била разположена Стара Велика България?
                     3. Кой е бил основателят на Стара Велика България?
                     4. С какво е известен основателят на Стара Велика България?
                     5. Коя е била столицата на Стара Велика България?
                     Обърни внимание, че предмета на питане Стара Велика България присъства във всички генерирани 
                     подзаявки, подзаявките не са просто отделени като изречения, а са допълнени с подходящ контекст, 
                     който ги прави да са напълно самодостатъчни, да може да се ползват като отделни заявки.
                     
                     # Пример
                     Заявката "Кога е роден Васил Левски? Кое е родното му място? Каква е връзката му с България? Какъв 
                     е неговият принос за бъдещото ѝ развитие като държава?" се превръща в следните подзаявки:
                     1. Кога е роден Васил Левски?
                     2. Кое е родното място на Васил Левски?
                     3. Каква е връзката на Васил Левски с България?
                     4. Какъв е приносът на Васил Левски за бъдещото развитие на България като държава?
                     
                     # Задача
                     Генерирай на {query_language} език подзаявките на тази заявка: {query}
                """

        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]

        llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                       top_k=40, temperature=0.3)
        llm_response_text = llm_response["choices"][0]["message"]["content"]
        return llm_response_text

    def is_composed_query(self, query):
        prompt = f""" 
                     # Системни инструкции  
                     Отговори дали дадената заявка се състои от
                     няколко отделни заявки (отделни въпроса) или не.
                     Отговори на български с Да или Не.
                     
                     # Заявка
                     {query}
                 """

        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]
        llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                       top_k=1, temperature=0.3)
        llm_response_text = llm_response["choices"][0]["message"]["content"]
        if llm_response_text[:2] == "Не" or llm_response_text[:2] == "не" or llm_response_text[:2] == "No" or \
                llm_response_text[:2] == "no":
            return False
        else:
            return True

    def does_need_rag(self, query):
        prompt = f"""
                     # Системни инструкции 
                     Ти си асистент, който разглежда даден текст на потребителя, и
                     решава дали за да се отговори на потребителя се изисква допълнителна информация
                     (отговори с 'Да' в такъв случай) или не (налични са небходимите знания за пълноценен 
                     отговор).
                     Отговаряй единствено с 'Да' или 'Не' (на български).
                     
                     # Досегашен разговор между теб (асистента) и потребителя
                     {self.history_str}
                     
                     # Текст на потребителя
                     {query} 
                  """

        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]

        llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                       top_k=1, temperature=0.3)
        llm_response_text = llm_response["choices"][0]["message"]["content"]
        return llm_response_text

    def get_chat_history_tokens_length(self, chat_history, last_n_slice=6):
        if not chat_history:
            return 0
        tokens_length = 0
        sliced_chat_history = chat_history[-last_n_slice:]
        for i in range(len(sliced_chat_history)):
            tokens_length += len(self.llm.tokenize(sliced_chat_history[i].encode('utf-8')))
        return tokens_length

    def shorten_history_str(self, chat_str, summary="", tokens_limit=1200, print_result=False):
        if summary:
            prompt = f""" 
                         # Системни инструкции
                         Ти си асистент, който прави кратко обобщение на разговор между потребителя и асистента. 
                         Направи сбито и информативно обобщение на този разговор в рамките на 
                         {tokens_limit} tokens. Запази съществена информация, която може да е необходима, за да се 
                         изпълни бъдеща заявка на потребителя. Обобщи всяка итерация между потребителя и асистента в
                         1-2 изречения, като запазиш само същественото.
                         Ако ти е трудно да обобщиш всичко в рамките на {tokens_limit} tokens, 
                         дай приоритет на последните части от разговора. 
                         Ще получиш досегашното обобщение на разговора и новата част от разговора, която трябва да 
                         обобщиш. Съкрати досегашното обобщение още повече, и го допълни с обобщението на новата част от
                         разговора, така че да се вмести отговора ти в {tokens_limit} tokens. Ако не можеш да вместиш
                         отговора си в {tokens_limit} tokens, то не взимай предвид първата част на досегашното 
                         обощение, а само последните няколко изречения от него. 
                         
                         # Досегашно обобщение на разговора:
                         {summary}
                         
                         # Новa част от разговора между асистента и потребителя (обобщи го според системните инструкции)
                         {chat_str}
                     """
        else:
            prompt = f""" 
                         # Системни инструкции
                         Ти си асистент, който прави кратко обобщение на разговор между потребителя и асистента. 
                         Направи сбито и информативно обобщение на този разговор в рамките на 
                         {tokens_limit} tokens. Запази съществена информация, която може да е необходима, за да се 
                         изпълни бъдеща заявка на потребителя. 
                         Ако ти е трудно да обобщиш всичко в рамките на {tokens_limit} tokens, 
                         дай приоритет на последните части от разговора.
                         
                         # Разговор между асистента и потребителя (обобщи го според системните инструкции)
                         {chat_str}
                     """

        # user_prompt = f""" Направи обобщение на досегашния разговор между потребителя и асистента:
        #                    {self.history_str} """
        # user_prompt = user_prompt.format(history_str=self.history_str)
        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]

        llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                       top_k=40, temperature=0.5)
        llm_response_text = llm_response["choices"][0]["message"]["content"]
        if print_result:
            print("Shortened chat conversation:")
            if self.menu_language == "bulgarian":
                logging.info("Съкратен чат до момента:")
            else:
                logging.info("Shortened chat conversation:")
            print(llm_response_text)
            logging.info(llm_response_text)
        return llm_response_text

    def shorten_chat_history(self, last_n_slice=16, tokens_limit=1700, last_n_slice_tokens_limit=1200):
        if not self.recent_chat_history:
            return ""

        recent_chat_history_str = "\n".join(self.recent_chat_history)
        history_str = self.summarised_chat_history + '\n' + recent_chat_history_str

        history_str_tokens_length = len(self.llm.tokenize(history_str.encode('utf-8')))

        if history_str_tokens_length < tokens_limit:
            return history_str

        chat_history_len = len(self.recent_chat_history)
        if last_n_slice > chat_history_len:
            last_n_slice = chat_history_len
        last_n_sliced_chat_history_tokens_length = self.get_chat_history_tokens_length(self.recent_chat_history,
                                                                                       last_n_slice=last_n_slice)
        current_slice = last_n_slice
        while last_n_sliced_chat_history_tokens_length > last_n_slice_tokens_limit and current_slice > 1:
            current_slice -= 1
            last_n_sliced_chat_history_tokens_length = self.get_chat_history_tokens_length(self.recent_chat_history,
                                                                                           last_n_slice=current_slice)
        last_n_str = "\n".join(self.recent_chat_history[-current_slice:])

        first_n_sliced_chat_history_tokens_length = \
            self.get_chat_history_tokens_length(self.recent_chat_history[:chat_history_len - current_slice],
                                                last_n_slice=0)
        first_n_str = ""
        if first_n_sliced_chat_history_tokens_length > 0:
            first_n_str = "\n".join(self.recent_chat_history[:chat_history_len - current_slice])

        shortened_first_n_str = self.summarised_chat_history + " " + first_n_str
        first_n_sliced_chat_history_tokens_length = len(self.llm.tokenize(shortened_first_n_str.encode('utf-8')))

        shortened_last_n_str = last_n_str

        if first_n_sliced_chat_history_tokens_length > 8000:
            shortened_first_n_str = ""
            first_n_sliced_chat_history_tokens_length = 0
        if last_n_sliced_chat_history_tokens_length > 8000:
            shortened_last_n_str = ""
            last_n_sliced_chat_history_tokens_length = 0

        i = 0
        while last_n_sliced_chat_history_tokens_length > last_n_slice_tokens_limit:
            if i > 2:
                break
            shortened_last_n_str = self.shorten_history_str(chat_str=shortened_last_n_str,
                                                            tokens_limit=last_n_slice_tokens_limit)
            last_n_sliced_chat_history_tokens_length = len(self.llm.tokenize(shortened_last_n_str.encode('utf-8')))
            i += 1
        i = 0

        while first_n_sliced_chat_history_tokens_length > tokens_limit - last_n_sliced_chat_history_tokens_length:
            if i > 2:
                break
            if i < 1:
                shortened_first_n_str = self.shorten_history_str(chat_str=first_n_str,
                                                                 summary=self.summarised_chat_history,
                                                                 tokens_limit=tokens_limit - last_n_slice_tokens_limit)
            else:
                shortened_first_n_str = self.shorten_history_str(chat_str=first_n_str,
                                                                 tokens_limit=tokens_limit - last_n_slice_tokens_limit)
            first_n_sliced_chat_history_tokens_length = len(self.llm.tokenize(shortened_first_n_str.encode('utf-8')))
            i += 1

        if i <= 2:
            self.summarised_chat_history = shortened_first_n_str
        else:
            self.summarised_chat_history = ""
        self.recent_chat_history = self.recent_chat_history[-current_slice:]

        if shortened_first_n_str == "":
            return shortened_last_n_str
        return shortened_first_n_str + "\n\n" + shortened_last_n_str

    def generate_query(self, query, print_result=False):
        if self.history_str == "":
            return query
        bg_coefficient = in_target_language(query, lang='bg')
        en_coefficient = in_target_language(query, lang='en')
        if bg_coefficient >= en_coefficient:
            query_language = "български"
        else:
            query_language = "английски"
        prompt = f""" 
                     # Системни инструкции
                     Ти си асистент, който при нужда минимално модифицира дадената заявка на потребителя на база
                     разговора до момента. Не отговаряй на заявката, а единствено я модифицирай при нужда.
                     В отговора си задължително не включвай следния текст: "При нужда, модифицирай тази заявка:". 
                     Целта е върнатата от теб заявка да може да се подаде на RAG (Retrieval
                     Generated Augmentation) система, т.е. в заявката трябва да се съдържа достатъчно подробен контекст.
                     Обогати заявката с достатъчен контекст от разговора до момента, ако има нужда.
                     Ако прецениш, че заявката е самодостатъчна (разбира се за какво конкретно се пита),
                     не променяй изобщо заявката, а я върни абсолютно същата.
                     
                     # Пример 1
                     Пример за заявка, имаща нужда от модифициране: "При нужда, модифицирай тази заявка: Кога е 
                     спечелил първия си трофей?" - Не се разбира за кого се пита, има нужда от преглед на 
                     разговора до момента, за да се разбере.
                     Тогава трябва да се допълни въпроса с името на човека, за когото се пита на база разговора
                     до момента. (Примерно отговаряш така: "Кога Рафаел Надал е спечелил първия си трофей?")
                     
                     # Пример 2
                     Пример за заявка, имаща нужда от модифициране: "При нужда, модифицирай тази заявка: Кога е 
                     написана книгата?" - Не се разбира за коя книга се пита, има нужда от преглед на 
                     разговора до момента, за да се разбере.
                     Тогава трябва да се допълни въпроса със заглавието на книгата, за която се пита на база разговора
                     до момента. (Примерно отговаряш така: "Кога е написана книгата "Под игото"?")
                     
                     # Пример 3
                     Пример за заявка, имаща нужда от модифициране: "При нужда, модифицирай тази заявка: 'Кой се 
                     занимава с въвеждането на тези мерки и с мониторинга за тяхното спазване в България?' - 
                     Не се разбира кои са тези мерки има нужда от преглед на 
                     разговора до момента, за да се разбере.
                     Тогава трябва да се допълни въпроса със заглавието на книгата, за която се пита на база разговора
                     до момента. (Примерно, ако в разговора се говори за мерки за намаляване на замърсяването на 
                     въздуха в градовете, отговаряш така: 
                     "Кой се занимава с въвеждането на тези мерки за намаляване на замърсяването на въздуха в 
                     градовете и с мониторинга за тяхното спазване в България?")
                     
                     # Пример 4
                     Пример за заявка, която е самодостатъчна: "При нужда, модифицирай тази заявка: Кога България
                     е основана като държава?" -
                     В този случай заявката е напълно изчерпателна, разбира се изцяло за какво се пита, връща се
                     същата заявка (Отговаряш така: "Кога България е основана като държава?").
                     В отговора си върни единствено своя резултат (променената заявка или същата заявка) без да 
                     добавяш разсъждения в отговора си. 
                     Нека върнатата от теб заявка да бъде на същия език, както дадената заявка. 
                     
                     # Разговор до момента
                     {self.history_str} 
                     
                     # Задача
                     При нужда, модифицирай на {query_language} език тази заявка: {query}
                  """

        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]

        llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                       top_k=1, temperature=0.1)
        llm_response_text = llm_response["choices"][0]["message"]["content"]
        if print_result:
            print("Generated query based on the current conversation:")
            if self.menu_language == "bulgarian":
                logging.info("Генерирана заявка на база разговора до момента:")
            else:
                logging.info("Generated query based on the current conversation:")
            print(llm_response_text)
            logging.info(llm_response_text)
        return llm_response_text

    def is_query(self, user_input, print_response=False):
        if self.history_str == "":
            prompt = f"""
                         # Системни инструкции 
                         Разглеждаш текст, въведен от потребител.
                         Отговори с 'Да', ако текстът представлява въпрос или заявка (query).
                         Отговори с 'Не', ако е коментар или личен към теб въпрос за твоето състояние.
                         
                         # Текст, въведен от потребителя
                         {user_input}
                      """
        else:
            prompt = f"""
                         # Системни инструкции  
                         Разглеждаш текст, въведен от потребител.
                         Отговори с 'Да', ако текстът представлява въпрос или заявка (query).
                         Отговори с 'Не', ако е коментар или личен към теб въпрос за твоето състояние.

                         # Разговор до момента между потребителя и асистента
                         {self.history_str} 
                         
                         # Текст, въведен от потребителя:
                         {user_input}
                      """

        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]

        llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                       top_k=1, temperature=0.3)
        llm_response_text = llm_response["choices"][0]["message"]["content"]
        if print_response:
            print("Is the text entered by the user a query?")
            if self.menu_language == "bulgarian":
                logging.info("Заявка ли е текстът, въведен от потребителя?")
            else:
                logging.info("Is the text entered by the user a query?")
            print(llm_response_text)
            logging.info(llm_response_text)
        if llm_response_text[:2] == "Не" or llm_response_text[:2] == "не" or llm_response_text[:2] == "No" or \
                llm_response_text[:2] == "no":
            return False
        else:
            return True

    def is_relevant_to_chat(self, query, print_response=False):
        if self.history_str == "":
            return False
        prompt = f"""
                     # Системни инструкции  
                     Ти си асистент, който преценява дали заявката има нужда от промяна, за да бъде самодостатъчна. 
                     Ако заявката съдържа цялата информация за какво/кого се пита без нуждата от разглеждане на
                     разговора до момента, то тя няма нужда от промяна (Отговор "Не").
                     Ако нещо в заявката трябва да се допълни, като се види разговора до момента, то 
                     заявката има нужда от промяна, за да бъде самодостатъчна (Отговор "Да")
                     
                     # Пример 1
                     Нека предположим, че в разговора се говори за водата и следващата заявка е "При каква температура 
                     ври?". В такъв случай заявката има нужда от промяна, за да бъде самодостатъчна. Ако се разгледа
                     заявката без разговора до момента, то не се разбира за какво се пита, в случая "При каква 
                     температура ври" не се разбира кое ври. Отговори в такива случаи с "Да".
                     Ако заявката беше "При каква температура ври водата?" тогава няма нужда от промяна, за да бъде 
                     самодостатъчна заявката, тя си е самодостатъчна, може да се зададе като самостоятелна заявка и без
                     да се знае разговора до момента. В такива случаи отговори с "Не".
                     
                     # Пример 2
                     Нека разговорът до момента да е за историята на египтяните и следващата заявка е "Кой е 
                     най-успешният им владетел?". 
                     В случая заявката има нужда от промяна, за да е самодостатъчна, не се разбира за 
                     владетеля на кои става въпрос (им е неопределено). Трябва да се види разговора до момента, където 
                     се вижда, че се говори за египтяните и заявката се изяснява. Затова тази заявка не си е 
                     самодостатъчна и има нужда от промяна. Отговори с "Да" в такива случаи.
                     Ако заявката беше "Кой е най-успешният владетел на египтяните?" или "Кой е най-успешният владетел 
                     в историята на Египет?", тогава заявката щеше да бъде самодостатъчна и нямаше да има нужда от 
                     промяна, тоест отговорът ти щеше да бъде "Не".
                     
                     # Разговор до момента
                     {self.history_str} 
                     
                     # Задача
                     Има ли нужда от промяна тази заявка, за да си е самодостатъчна: {query}
                  """

        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]

        llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                       top_k=1, temperature=0.3)
        llm_response_text = llm_response["choices"][0]["message"]["content"]
        if print_response:
            print("Does the query need modification in order to be self-sufficient?")
            if self.menu_language == "bulgarian":
                logging.info("Има ли нужда от промяна тази заявка, за да си е самодостатъчна:")
            else:
                logging.info("Does the query need modification in order to be self-sufficient?")
            print(llm_response_text)
            logging.info(llm_response_text)
        if llm_response_text[:2] == "Не" or llm_response_text[:2] == "не" or llm_response_text[:2] == "No" or \
                llm_response_text[:2] == "no":
            return False
        else:
            return True

    def can_use_chat_history(self, query, print_answer=False):
        if self.history_str == "":
            return False
        prompt = f"""
                     # Системни инструкции 
                     Ти си асистент, който разглежда дадена заявка на потребителя и досегашният разговор между
                     потребителя и асистента. Отговори с Да или Не дали може да се отговори на заявката на
                     потребителя само на база досегашния разговор. 
                     
                     # Досегашен разговор между потребителя и асистента
                     {self.history_str}
                     
                     # Задача
                     Отговори с Да или Не: Може ли само на базата на досегашния разговор
                     да се отговори на следния въпрос на потребителя: {query}
                  """

        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]

        llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                       top_k=1, temperature=0.1)
        llm_response_text = llm_response["choices"][0]["message"]["content"]
        if print_answer:
            print("Can the chat conversation so far be used to answer the user query?")
            if self.menu_language == "bulgarian":
                logging.info("Може ли да се използва само досегашния разговор, за да се отговори на потребителския "
                             "въпрос?")
            else:
                logging.info("Can the chat conversation so far be used to answer the user query?")
            print(llm_response_text)
            logging.info(llm_response_text)
        if llm_response_text[:2] == "Не" or llm_response_text[:2] == "не" or llm_response_text[:2] == "Не" or \
                llm_response_text[:2] == "no" or self.history_str == "":
            return False
        else:
            return True

    def translate_query(self, query):
        bg_coefficient = in_target_language(query, lang='bg')
        en_coefficient = in_target_language(query, lang='en')
        if bg_coefficient >= en_coefficient:
            query_language = "български"
            translate_language = "английски"
        else:
            query_language = "английски"
            translate_language = "български"

        prompt = f"""
                             # Системни инструкции 
                             Трябва да преведеш заявката на потребителя от {query_language} на {translate_language} 
                             език.
                             Преводът ти трябва да бъде максимално точен и близък до първоначалната заявка.
                             Като отговор върни единствено преведаната заявка.
                             # Задача
                             Преведи следната заявка на {translate_language} език: 
                             {query}
                          """
        dialogue_template = [
            {"role": "user",
             "content": prompt}
        ]
        llm_response = self.llm.create_chat_completion(messages=dialogue_template,
                                                       top_k=1, temperature=0.1)
        translated_query = llm_response["choices"][0]["message"]["content"]
        return translated_query

    def ask_system(self, user_input, just_llm=False, always_rag=False, use_rag_fusion=False, use_hyde=False,
                   use_generated_subqueries=False, use_relevant_resource_llm_check=False, use_default_db=True,
                   use_uploaded_db=False, modify_query=True, translate_query=False,
                   print_query=False, print_response=False, print_urls=False,
                   print_context_passages=False, print_steps=True, temperature=0.7, max_answer_tokens=42,
                   topn_articles=3):

        self.use_default_db = use_default_db
        self.use_uploaded_db = use_uploaded_db
        if use_default_db:
            self.chunks_and_embeddings = self.db_chunks_and_embeddings
            self.faiss_index = self.index_faiss_db
            self.embeddings = self.db_embeddings
        else:
            self.chunks_and_embeddings = self.user_uploaded_chunks_and_embeddings
            self.faiss_index = self.user_uploaded_faiss_index
            self.embeddings = self.user_uploaded_embeddings

        if self.evaluation_mode and (self.evaluation_df is None):
            self.read_evaluation_df()

        if user_input.strip() == "":
            return "", []

        user_input_tokens_length = len(self.llm.tokenize(user_input.encode('utf-8')))

        if user_input_tokens_length > 8000:
            if self.menu_language == "bulgarian":
                logging.info(
                    "Твърде голяма заявка. Не може да бъде преработена. Опитайте се да напишете доста "
                    "по-кратка заявка.")
                return "Твърде голяма заявка. Не може да бъде преработена. Опитайте се да напишете " \
                       "доста по-кратка заявка.", []
            else:
                logging.info(
                    "Too long query. It cannot be processed. Try writing a much smaller query.")
                return "Too long query. It cannot be processed. Try writing a much smaller query.", []

        if not self.recent_chat_history == []:
            self.history_str = self.shorten_chat_history()

        if modify_query:
            query = self.generate_query(query=user_input, print_result=True)
        else:
            query = user_input

        if always_rag or self.is_query(query):
            needs_rag_text = "Yes"
        else:
            needs_rag_text = "No"

        if just_llm or needs_rag_text == "No":
            needs_rag = False
        else:
            needs_rag = True

        print(f"Needs RAG: {needs_rag}")
        if self.menu_language == "bulgarian":
            logging.info(f"Има ли нужда от RAG?: {needs_rag}")
        else:
            logging.info(f"Is RAG needed?: {needs_rag}")

        if use_generated_subqueries:
            use_generated_subqueries = self.is_composed_query(query)
            print(f"use_generated_subqueries: {use_generated_subqueries}")
            if self.menu_language == "bulgarian":
                logging.info(f"Да се използват ли генерирани подзаявки: "
                             f"{use_generated_subqueries}")
            else:
                logging.info(f"Should generated subqueries be used: "
                             f"{use_generated_subqueries}")

        relevant_urls = []

        if needs_rag:
            response, relevant_urls = self.ask_rag(query, use_rag_fusion=use_rag_fusion, use_hyde=use_hyde,
                                                   use_generated_subqueries=use_generated_subqueries,
                                                   use_relevant_resource_llm_check=use_relevant_resource_llm_check,
                                                   translate_query=translate_query,
                                                   print_query=print_query, print_response=print_response,
                                                   print_urls=print_urls, print_context_passages=print_context_passages,
                                                   print_steps=print_steps, temperature=temperature,
                                                   max_answer_tokens=max_answer_tokens, topn_articles=topn_articles)
        else:
            response = self.ask_llm(query, use_chat_history=False)

        if self.menu_language == "bulgarian":
            user_input = "Потребител: " + user_input
            system_response = "Асистент: " + response
        else:
            user_input = "User: " + user_input
            system_response = "Assistant: " + response

        self.chat_history.append(user_input)
        self.chat_history.append(system_response)
        self.recent_chat_history.append(user_input)
        self.recent_chat_history.append(system_response)

        self.relevant_resources_history.append(relevant_urls)

        self.unsaved_chat_history.append(user_input)
        self.unsaved_chat_history.append(system_response)
        self.unsaved_relevant_resources_history.append(relevant_urls)
        return response, relevant_urls
