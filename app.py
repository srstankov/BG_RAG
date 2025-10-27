import os
import logging
import customtkinter as ctk
from menu import Menu
from chat import ChatInput, ChatOutput
from threading import Thread
from rag import Rag, resource_path


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry('1000x600')
        self.minsize(1000, 600)
        ctk.set_appearance_mode('dark')
        self.title('BG RAG')
        self.iconbitmap(resource_path('icon.ico'))
        self.rowconfigure(0, weight=4, uniform='b')
        self.rowconfigure(1, weight=1, uniform='b')
        self.columnconfigure(0, weight=1, uniform='a')
        self.columnconfigure(1, weight=3, uniform='a')

        self.menu = Menu(self)
        self.chat_input = ChatInput(self)
        self.chat_output = ChatOutput(self)
        self.bind_all("<Key>", self.on_key_release, "+")
        self.user_input = ''
        self.rag_response = ctk.StringVar(value='')
        self.relevant_resources = []
        self.protocol('WM_DELETE_WINDOW', self.close_window)

        if self.menu.options_frame.menu_language_var.get() == "bulgarian":
            logging.info("Стартиране на приложението.")
            logging.info("BG RAG приложението зарежда LLM и базите данни.")
            logging.info("Повечето функции на приложението ще бъдат НЕДОСТЪПНИ, докато не завърши зареждането.")
            logging.info("Ако това е първото стартиране на приложението след инсталацията, то ще трябва добра интернет "
                         "връзка и значително време, за да се изтеглят моделите, използвани в системата.")
            logging.info("Моля, изчакайте.")
        else:
            logging.info("Starting the app.")
            logging.info("BG RAG app is loading the LLM and the datasets.")
            logging.info("Most functions of the app will be DISABLED until the loading is completed.")
            logging.info("If this is the first launching of the app after the installation, then a good internet "
                         "connection and significant time will be needed in order for the models used in the system to "
                         "be downloaded.")
            logging.info("Please wait.")

        self.update_idletasks()

        self.rag = Rag()

        self.load_rag()

        self.menu.import_frame.import_button_clicked.trace('w', self.import_files_callback)
        self.menu.import_frame.delete_button_confirmed.trace('w', self.delete_files)
        self.chat_input.input_text.trace('w', self.input_text_callback)
        self.chat_input.clear_chat_button_clicked.trace('w', self.clear_chat)

        self.rag_response.trace('w', lambda x, y, z: self.chat_output.rag_response_text_callback(
                                                     self.rag_response.get(),
                                                     self.relevant_resources))

        self.menu.save_frame.save_button_clicked.trace('w', self.save_chat)
        self.menu.options_frame.menu_language_var.trace('w', self.change_language)
        self.window_closed = False
        self.mainloop()

    def load_rag(self):
        load_thread = Thread(target=self.load_rag_databases)
        load_thread.start()

    def load_rag_databases(self):
        # self.rag.update_both_databases()  # when the default database has been updated to a newer version uncomment

        self.rag.load_llm()
        self.rag.load_embedding_model()
        self.rag.load_reranking_model()

        self.rag.read_database()
        if os.path.exists(resource_path('imported_files.txt', app_generated_file=True)):
            self.rag.read_uploaded_files_data()
            self.menu.import_frame.delete_button.configure(state=ctk.NORMAL)
        self.menu.import_frame.choose_files_button.configure(state=ctk.NORMAL)
        self.chat_input.send_button.configure(state=ctk.NORMAL)

        if self.menu.options_frame.menu_language_var.get() == "bulgarian":
            logging.info("Забележка: Този лог ще се запази във файла 'rag_log.log' в "
                         f"директорията '{resource_path('', app_generated_file=True)}'.")
            logging.info("Зареждането завърши.")
            logging.info("Приложението може вече да бъде използвано нормално.")
            if self.rag.evaluation_mode:
                logging.info("Активиран е режим за оценяване на системата.")
        else:
            logging.info("Note: This log will be stored in the file 'rag_log.log' in the "
                         f"directory '{resource_path('', app_generated_file=True)}'.")
            logging.info("Loading completed.")
            logging.info("The app can now be used normally.")
            if self.rag.evaluation_mode:
                logging.info("Evaluation mode for the system is activated.")

    def close_window(self):
        self.window_closed = True
        if self.rag.evaluation_df is not None and self.rag.evaluation_mode:
            if self.menu.options_frame.menu_language_var.get() == "bulgarian":
                logging.info("Запазване на базата данни с оценяванията...")
            else:
                logging.info("Saving evaluation dataset...")

            self.rag.evaluation_df.to_pickle(self.rag.evaluation_df_store_pickle_filename)

            if self.menu.options_frame.menu_language_var.get() == "bulgarian":
                logging.info(f"Базата данни с оценяванията беше запазена успешно в "
                             f"{self.rag.evaluation_df_store_pickle_filename}")
            else:
                logging.info(f"Evaluation dataset saved successfully to {self.rag.evaluation_df_store_pickle_filename}")

        if self.menu.options_frame.menu_language_var.get() == "bulgarian":
            logging.info("Затваряне на приложението.\n\n\n")
        else:
            logging.info("Closing the app.\n\n\n")
        self.after(1000, self.destroy)

    def clear_chat(self, *args):
        for label in self.chat_output.winfo_children():
            label.destroy()
        self.chat_output = ChatOutput(self)
        self.rag.chat_history = []
        self.rag.recent_chat_history = []
        self.rag.summarised_chat_history = ""
        self.rag.relevant_resources_history = []
        self.rag.unsaved_chat_history = []
        self.rag.unsaved_relevant_resources_history = []
        self.rag.history_str = ""
        if self.menu.options_frame.menu_language_var.get() == "bulgarian":
            logging.info("Досегашният разговор беше изчистен от системата.")

        else:
            logging.info("The current conversation was removed from the system.")

    def get_rag_response(self):
        self.relevant_resources = []

        rag_fusion = getattr(getattr(self.menu, 'options_frame'), 'rag_fusion_var').get()
        hyde = getattr(getattr(self.menu, 'options_frame'), 'hyde_var').get()
        subqueries = getattr(getattr(self.menu, 'options_frame'), 'subqueries_var').get()
        default_db = getattr(getattr(self.menu, 'options_frame'), 'default_db_var').get()
        uploaded_db = getattr(getattr(self.menu, 'options_frame'), 'uploaded_db_var').get()
        display_resources = getattr(getattr(self.menu, 'options_frame'), 'display_resources_var').get()
        just_llm = getattr(getattr(self.menu, 'options_frame'), 'just_llm_var').get()
        always_rag = getattr(getattr(self.menu, 'options_frame'), 'always_rag_var').get()
        modify = getattr(getattr(self.menu, 'options_frame'), 'modify_query_var').get()
        translate = getattr(getattr(self.menu, 'options_frame'), 'translate_query_var').get()
        relevant_resource_llm_check = getattr(getattr(self.menu, 'options_frame'),
                                              'relevant_resource_llm_check_var').get()
        log_relevant_chunks_check = getattr(getattr(self.menu, 'log_frame'),
                                            'log_relevant_chunks_var').get()

        if rag_fusion == 'rag_fusion':
            use_rag_fusion = True
        else:
            use_rag_fusion = False

        if hyde == 'hyde':
            use_hyde = True
        else:
            use_hyde = False

        if subqueries == 'subq':
            use_generated_subqueries = True
        else:
            use_generated_subqueries = False

        if just_llm == 'just_llm':
            use_just_llm = True
        else:
            use_just_llm = False

        if always_rag == 'always_rag':
            use_always_rag = True
        else:
            use_always_rag = False

        if modify == 'modify':
            modify_query = True
        else:
            modify_query = False

        if translate == 'translate':
            translate_query = True
        else:
            translate_query = False

        if relevant_resource_llm_check == 'rel_llm_check':
            use_relevant_resource_llm_check = True
        else:
            use_relevant_resource_llm_check = False

        if default_db == 'default_db':
            use_default_db = True
        else:
            use_default_db = False

        if uploaded_db == 'uploaded_db':
            use_uploaded_db = True
        else:
            use_uploaded_db = False

        if log_relevant_chunks_check == "log_chunks":
            log_relevant_chunks = True
        else:
            log_relevant_chunks = False

        topn_articles = int(self.menu.options_frame.topn_articles_slider.get())

        try:
            resp, rel_urls = self.rag.ask_system(user_input=self.user_input, just_llm=use_just_llm,
                                                 always_rag=use_always_rag, use_rag_fusion=use_rag_fusion,
                                                 use_hyde=use_hyde, use_generated_subqueries=use_generated_subqueries,
                                                 use_relevant_resource_llm_check=use_relevant_resource_llm_check,
                                                 modify_query=modify_query, translate_query=translate_query,
                                                 use_default_db=use_default_db, use_uploaded_db=use_uploaded_db,
                                                 print_context_passages=log_relevant_chunks,
                                                 topn_articles=topn_articles)

        except Exception as e:
            print(f"ERROR: {e}")
            logging.exception(f"ERROR: {e}", stack_info=True)
            if self.menu.options_frame.menu_language_var.get() == "bulgarian":
                logging.info("ГРЕШКА! Генерирането на отговор на заявката (функция ask_system от класа Rag) "
                             "връща ИЗКЛЮЧЕНИЕ (EXCEPTION)!")
                resp = "ГРЕШКА! Генерирането на отговор на заявката (функция ask_system от класа Rag) " \
                       "връща ИЗКЛЮЧЕНИЕ (EXCEPTION)!"
            else:
                logging.info("ERROR! The answer generation to the query (function ask_system in Rag class) "
                             "returns an EXCEPTION!")
                resp = "ERROR! The answer generation to the query (function ask_system in Rag class) returns an " \
                       "EXCEPTION!"
            print("ERROR! The answer generation to the query (function ask_system in Rag class) returns an EXCEPTION!")
            rel_urls = []

        if display_resources == 'resources':
            self.relevant_resources = rel_urls
        else:
            self.relevant_resources = []
            if rel_urls:
                if self.menu.options_frame.menu_language_var.get() == "bulgarian":
                    logging.info("Заявка: " + self.user_input)
                    logging.info("Релевантни източници:")
                else:
                    logging.info("Query: " + self.user_input)
                    logging.info("Relevant resources:")

                for resource in rel_urls:
                    logging.info(resource)

        self.rag_response.set(resp)
        self.chat_input.send_button.configure(state=ctk.NORMAL)
        self.chat_input.clear_chat_button.configure(state=ctk.NORMAL)

    def input_text_callback(self, *args):
        self.user_input = self.chat_input.input_text.get()
        self.chat_output.add_text_label(text=self.user_input, is_user=True)
        input_thread = Thread(target=self.get_rag_response)
        input_thread.start()

    def change_language(self, *args):
        self.menu.change_language(new_language=self.menu.options_frame.menu_language_var.get())
        self.chat_input.change_language(new_language=self.menu.options_frame.menu_language_var.get())
        self.chat_output.menu_language = self.menu.options_frame.menu_language_var.get()
        self.menu.import_frame.change_language(new_language=self.menu.options_frame.menu_language_var.get())
        self.menu.save_frame.change_language(new_language=self.menu.options_frame.menu_language_var.get())
        self.menu.log_frame.change_language(new_language=self.menu.options_frame.menu_language_var.get())
        self.menu.help_frame.change_language(new_language=self.menu.options_frame.menu_language_var.get())
        self.rag.menu_language = self.menu.options_frame.menu_language_var.get()

    def import_files_callback(self, *args):
        self.menu.import_frame.import_button.configure(state=ctk.DISABLED)
        self.menu.import_frame.delete_button.configure(state=ctk.DISABLED)
        self.menu.import_frame.choose_files_button.configure(state=ctk.DISABLED)
        self.menu.import_frame.clear_selected_files_button.configure(state=ctk.DISABLED)
        self.chat_input.send_button.configure(state=ctk.DISABLED)

        if self.menu.options_frame.menu_language_var.get() == "bulgarian":
            self.menu.import_frame.info_text.set('Качване на файловете...\nМоже да отнеме няколко минути.')
        else:
            self.menu.import_frame.info_text.set('Importing files...\nIt may take a few minutes.')

        self.menu.import_frame.import_files_animation()
        import_thread = Thread(target=self.import_files)
        import_thread.start()

    def import_files(self):
        _, _, _, imported_files = \
            self.rag.upload_files(user_filenames=[file.name for file in self.menu.import_frame.selected_files],
                                  united_chunks_dictionary=self.rag.user_uploaded_chunks_and_embeddings,
                                  index_faiss_user_files=self.rag.user_uploaded_faiss_index,
                                  united_embeddings=self.rag.user_uploaded_embeddings)
        self.menu.import_frame.selected_filenames = imported_files
        self.rag.read_uploaded_files_data()

        if self.menu.options_frame.just_llm_var.get() == 'no_just_llm':
            self.menu.options_frame.uploaded_db_check.configure(state=ctk.NORMAL)
        self.chat_input.send_button.configure(state=ctk.NORMAL)
        if os.path.exists(resource_path('imported_files.txt', app_generated_file=True)):
            imported_files_txt = open(resource_path('imported_files.txt', app_generated_file=True), 'a',
                                      encoding="utf-8")
        else:
            imported_files_txt = open(resource_path('imported_files.txt', app_generated_file=True), 'w',
                                      encoding="utf-8")
        for file in self.menu.import_frame.selected_filenames:
            imported_files_txt.write(file)
            imported_files_txt.write('\n')
        imported_files_txt.close()
        print(f"Imported files: {self.menu.import_frame.selected_filenames}")
        self.menu.import_frame.time_to_stop_import_animation = True

    def delete_files(self, *args):
        self.menu.options_frame.uploaded_db_var.set('no_uploaded_db')
        if self.menu.options_frame.just_llm_var.get() == 'no_just_llm':
            self.menu.options_frame.default_db_var.set('default_db')
        self.menu.options_frame.uploaded_db_check.configure(state=ctk.DISABLED)
        os.remove(resource_path('imported_files.txt', app_generated_file=True))
        self.rag.delete_uploaded_files()
        print(f"Deleted files: {self.menu.import_frame.imported_filenames}")

    def save_chat(self, *args):
        try:
            save_file = open(os.path.join(self.menu.save_frame.chosen_directory.get(), self.menu.save_frame.filename),
                             mode='a', encoding="utf-8")
            i = 0
            for reply in self.rag.unsaved_chat_history:
                if i % 2 == 1 and self.menu.save_frame.add_resources_var.get() == "add_resources" and \
                        self.rag.unsaved_relevant_resources_history and \
                        self.rag.unsaved_relevant_resources_history[i // 2]:

                    if self.menu.options_frame.menu_language_var.get() == "bulgarian":
                        save_file.write(f"{reply}\n\nИзточници: \n")
                    else:
                        save_file.write(f"{reply}\n\nResources: \n")
                    for resource in self.rag.unsaved_relevant_resources_history[i // 2]:
                        save_file.write(f"{resource}\n")
                    save_file.write("\n\n\n")
                else:
                    save_file.write(f"{reply}\n\n")
                i += 1
            self.rag.unsaved_chat_history = []
            self.rag.unsaved_relevant_resources_history = []
            save_file.close()
        except Exception as e:
            self.menu.save_frame.successful_save = False
            print(f"ERROR: {e}")
            print("ERROR! Chat conversation could not be saved!")
            logging.exception(f"ERROR: {e}", stack_info=True)
            if self.menu.options_frame.menu_language_var.get() == "bulgarian":
                logging.info("ГРЕШКА! Чат разговорът НЕ беше запазен!")
            else:
                logging.info("ERROR! Chat conversation was NOT saved!")
        else:
            self.menu.save_frame.successful_save = True

    @staticmethod
    # https://stackoverflow.com/a/47496024
    def on_key_release(event):
        ctrl = (event.state & 0x4) != 0

        if event.keycode == 90 and ctrl and event.keysym.lower() != "z":
            event.widget.event_generate("<<Undo>>")

        if event.keycode == 89 and ctrl and event.keysym.lower() != "y":
            event.widget.event_generate("<<Redo>>")

        if event.keycode == 88 and ctrl and event.keysym.lower() != "x":
            event.widget.event_generate("<<Cut>>")

        if event.keycode == 86 and ctrl and event.keysym.lower() != "v":
            event.widget.event_generate("<<Paste>>")

        if event.keycode == 67 and ctrl and event.keysym.lower() != "c":
            event.widget.event_generate("<<Copy>>")

        if event.keycode == 65 and ctrl and event.keysym.lower() != "a":
            event.widget.event_generate("<<SelectAll>>")
