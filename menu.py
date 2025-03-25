import json
import customtkinter as ctk
from PIL import Image
from tkinter import filedialog as fd
import os
from tkinter import messagebox
import logging
from pathlib import Path
from rag import resource_path


class TextHandler(logging.Handler):
    # This class allows you to log to a Tkinter Text or ScrolledText widget
    # Adapted from Moshe Kaplan: https://gist.github.com/moshekaplan/c425f861de7bbf28ef06

    def __init__(self, text):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Store a reference to the Text it will log to
        self.text = text

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text.configure(state=ctk.NORMAL)
            if msg != "flash_attn is not installed. Using PyTorch native attention implementation.":
                self.text.insert(ctk.END, msg + '\n\n')
            self.text.configure(state=ctk.DISABLED)
            # Autoscroll to the bottom
            self.text.yview(ctk.END)

        # This is necessary because we can't modify the Text from other threads
        self.text.after(0, append)


class Menu(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color='transparent', corner_radius=0)
        self.grid(row=0, column=0, rowspan=2, sticky='nsew')

        self.menu = ctk.CTkTabview(master=self)
        options_config_json = open(resource_path('options_config.json'), 'r')
        options_config = json.load(options_config_json)
        options_config_json.close()
        if options_config['menu_language_var'] == "english":
            self.menu.add('Options')
            self.menu.add('Import')
            self.menu.add('Save')
            self.menu.add('Log')
            self.menu.add('Help')
            self.menu.set('Log')
        elif options_config['menu_language_var'] == "bulgarian":
            self.menu.add('Опции')
            self.menu.add('Качи')
            self.menu.add('Запази')
            self.menu.add('Лог')
            self.menu.add('Помощ')
            self.menu.set('Лог')

        if options_config['menu_language_var'] == "english":
            self.options_frame = OptionsFrame(self.menu.tab('Options'))
            self.import_frame = ImportFrame(self.menu.tab('Import'))
            self.save_frame = SaveFrame(self.menu.tab('Save'))
            self.log_frame = LogFrame(self.menu.tab('Log'))
            self.help_frame = HelpFrame(self.menu.tab('Help'))
        elif options_config['menu_language_var'] == "bulgarian":
            self.options_frame = OptionsFrame(self.menu.tab('Опции'))
            self.import_frame = ImportFrame(self.menu.tab('Качи'))
            self.save_frame = SaveFrame(self.menu.tab('Запази'))
            self.log_frame = LogFrame(self.menu.tab('Лог'))
            self.help_frame = HelpFrame(self.menu.tab('Помощ'))

        menu_icon_white_ctk = ctk.CTkImage(
            light_image=Image.open(resource_path('images') + os.sep + 'menu_icon_black.png'),
            dark_image=Image.open(resource_path('images') + os.sep + 'menu_icon_white.png'),
            size=(40, 40)
        )
        self.menu_button = ctk.CTkButton(self, text='', image=menu_icon_white_ctk, command=self.show,
                                         fg_color='transparent', hover_color='#383333', height=50, width=50)

        self.close_menu_button = ctk.CTkButton(self, text='X', command=self.close, width=10, height=10,
                                               fg_color='red', hover_color='#bf1717')
        self.show()

    def show(self):
        self.menu_button.place_forget()
        self.menu.place(relx=0, rely=0.025, anchor='nw', relwidth=1, relheight=0.975)
        self.close_menu_button.place(relx=0.92, rely=0.00, anchor='nw')

    def close(self):
        self.close_menu_button.place_forget()
        self.menu.place_forget()
        self.menu_button.place(relx=0.05, rely=0.025, anchor='nw')

    def change_language(self, new_language):
        if new_language == "bulgarian":
            self.menu.rename("Options", "Опции")
            self.menu.rename("Import", "Качи")
            self.menu.rename("Save", "Запази")
            self.menu.rename("Log", "Лог")
            self.menu.rename("Help", "Помощ")
            self.menu.set("Опции")
        elif new_language == "english":
            self.menu.rename("Опции", "Options")
            self.menu.rename("Качи", "Import")
            self.menu.rename("Запази", "Save")
            self.menu.rename("Лог", "Log")
            self.menu.rename("Помощ", "Help")
            self.menu.set("Options")


class OptionsFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(expand=True, fill='both')

        self.rag_fusion_var = ctk.StringVar(value='no_rag_fusion')
        self.rag_fusion_check = ctk.CTkCheckBox(self, text="RAG Fusion", variable=self.rag_fusion_var,
                                                onvalue='rag_fusion',
                                                offvalue='no_rag_fusion')

        self.hyde_var = ctk.StringVar(value='no_hyde')
        self.hyde_check = ctk.CTkCheckBox(self, text="HyDE", variable=self.hyde_var, onvalue='hyde',
                                          offvalue='no_hyde')

        self.subqueries_var = ctk.StringVar(value='no_subq')
        self.subqueries_check = ctk.CTkCheckBox(self, text="Use generated subqueries", variable=self.subqueries_var,
                                                onvalue='subq', offvalue='no_subq')

        self.default_db_var = ctk.StringVar(value='default_db')
        self.default_db_check = ctk.CTkCheckBox(self, text="Default", variable=self.default_db_var,
                                                onvalue='default_db', offvalue='no_default_db',
                                                command=self.database_click)

        self.uploaded_db_var = ctk.StringVar(value='no_uploaded_db')
        self.uploaded_db_check = ctk.CTkCheckBox(self, text="Uploaded files", variable=self.uploaded_db_var,
                                                 onvalue='uploaded_db', offvalue='no_uploaded_db',
                                                 command=self.database_click)

        if not os.path.exists(resource_path('imported_files.txt')):
            self.uploaded_db_check.configure(state=ctk.DISABLED)

        self.display_resources_var = ctk.StringVar(value='resources')
        self.display_resources_check = ctk.CTkCheckBox(self, text="Display resources",
                                                       variable=self.display_resources_var,
                                                       onvalue='resources', offvalue='no_resources')

        self.just_llm_var = ctk.StringVar(value='no_just_llm')
        self.just_llm_check = ctk.CTkCheckBox(self, text="Use just the\nLLM (not RAG)", variable=self.just_llm_var,
                                              onvalue='just_llm', offvalue='no_just_llm',
                                              command=self.just_llm_click)

        self.always_rag_var = ctk.StringVar(value='no_always_rag')
        self.always_rag_check = ctk.CTkCheckBox(self, text="Use RAG\nalways", variable=self.always_rag_var,
                                                onvalue='always_rag', offvalue='no_always_rag',
                                                command=self.always_rag_click)

        self.modify_query_var = ctk.StringVar(value='modify')
        self.modify_query_check = ctk.CTkCheckBox(self, text="Modify query based on chat",
                                                  variable=self.modify_query_var,
                                                  onvalue='modify', offvalue='no_modify')

        self.translate_query_var = ctk.StringVar(value='no_translate')
        self.translate_query_check = ctk.CTkCheckBox(self, text="Translate query for multilingual\nresources",
                                                     variable=self.translate_query_var,
                                                     onvalue='translate', offvalue='no_translate')

        self.relevant_resource_llm_check_var = ctk.StringVar(value='no_rel_llm_check')
        self.relevant_resource_llm_check_checkbox = ctk.CTkCheckBox(self, text="Check relevant resources by LLM",
                                                                    variable=self.relevant_resource_llm_check_var,
                                                                    onvalue='rel_llm_check',
                                                                    offvalue='no_rel_llm_check')

        self.topn_articles_var = ctk.IntVar(value=3)
        self.topn_articles_slider = ctk.CTkSlider(self, from_=1, to=7, number_of_steps=6,
                                                  variable=self.topn_articles_var,
                                                  command=self.move_topn_articles_slider)

        self.menu_language_var = ctk.StringVar(value='english')
        self.bulgarian_language_check = ctk.CTkCheckBox(self, text="Български", variable=self.menu_language_var,
                                                        onvalue='bulgarian', offvalue='english',
                                                        command=lambda: self.change_language('bulgarian'))

        self.english_language_check = ctk.CTkCheckBox(self, text="English", variable=self.menu_language_var,
                                                      onvalue='english', offvalue='bulgarian',
                                                      command=lambda: self.change_language('english'))

        label_font = ctk.CTkFont(family='Helvetica', size=16, weight='bold')
        self.advanced_rag_label = ctk.CTkLabel(self, text='Advanced RAG', font=label_font)
        self.database_label = ctk.CTkLabel(self, text='Database', font=label_font)
        self.other_label = ctk.CTkLabel(self, text='Other', font=label_font)
        self.topn_articles_label = ctk.CTkLabel(self, text=f'Top n relevant chunks for context: '
                                                           f'{int(self.topn_articles_slider.get())}')
        self.menu_language_label = ctk.CTkLabel(self, text='Menu Language', font=label_font)

        self.read_options_config()
        self.bind("<Destroy>", lambda event: self.update_options_config(on_exit=True))

        self.advanced_rag_label.place(relx=0.5, rely=0.03, relwidth=0.92, relheight=0.04, anchor='center')
        self.rag_fusion_check.place(relx=0.04, rely=0.07, relwidth=0.44, relheight=0.06, anchor='nw')
        self.hyde_check.place(relx=0.52, rely=0.07, relwidth=0.44, relheight=0.06, anchor='nw')
        self.subqueries_check.place(relx=0.04, rely=0.14, relwidth=0.92, relheight=0.06, anchor='nw')

        self.database_label.place(relx=0.5, rely=0.23, relwidth=0.92, relheight=0.04, anchor='center')
        self.default_db_check.place(relx=0.04, rely=0.27, relwidth=0.44, relheight=0.06, anchor='nw')
        self.uploaded_db_check.place(relx=0.52, rely=0.27, relwidth=0.48, relheight=0.06, anchor='nw')

        self.other_label.place(relx=0.5, rely=0.36, relwidth=0.92, relheight=0.04, anchor='center')
        self.display_resources_check.place(relx=0.04, rely=0.4, relwidth=0.92, relheight=0.05, anchor='nw')
        self.just_llm_check.place(relx=0.04, rely=0.47, relwidth=0.5, relheight=0.06, anchor='nw')
        self.always_rag_check.place(relx=0.6, rely=0.47, relwidth=0.4, relheight=0.06, anchor='nw')
        self.modify_query_check.place(relx=0.04, rely=0.54, relwidth=0.92, relheight=0.06, anchor='nw')
        self.translate_query_check.place(relx=0.04, rely=0.61, relwidth=0.92, relheight=0.06, anchor='nw')
        self.relevant_resource_llm_check_checkbox.place(relx=0.04, rely=0.68, relwidth=0.92, relheight=0.06,
                                                        anchor='nw')
        self.topn_articles_label.place(relx=0.5, rely=0.78, relwidth=0.92, relheight=0.05, anchor='center')
        self.topn_articles_slider.place(relx=0.04, rely=0.81, relwidth=0.92, relheight=0.03, anchor='nw')

        self.menu_language_label.place(relx=0.5, rely=0.89, relwidth=0.92, relheight=0.04, anchor='center')
        self.bulgarian_language_check.place(relx=0.04, rely=0.93, relwidth=0.44, relheight=0.06, anchor='nw')
        self.english_language_check.place(relx=0.52, rely=0.93, relwidth=0.48, relheight=0.06, anchor='nw')

    def read_options_config(self, just_llm_unchecked=False):
        options_config_json = open(resource_path('options_config.json'), 'r')
        options_config = json.load(options_config_json)
        options_config_json.close()
        if just_llm_unchecked:
            self.rag_fusion_var.set(options_config['rag_fusion_var'])
            self.hyde_var.set(options_config['hyde_var'])
            self.subqueries_var.set(options_config['subqueries_var'])
            if os.path.exists(resource_path('imported_files.txt')):
                self.default_db_var.set(options_config['default_db_var'])
                self.uploaded_db_var.set(options_config['uploaded_db_var'])
            else:
                self.default_db_var.set('default_db')
                self.uploaded_db_var.set('no_uploaded_db')
                self.uploaded_db_check.configure(state=ctk.DISABLED)
            self.display_resources_var.set(options_config['display_resources_var'])
            self.always_rag_var.set(options_config['always_rag_var'])
            self.modify_query_var.set(options_config['modify_query_var'])
            self.translate_query_var.set(options_config['translate_query_var'])
            self.relevant_resource_llm_check_var.set(options_config['relevant_resource_llm_check_var'])

        elif options_config['just_llm_var'] == 'no_just_llm':
            if os.path.exists(resource_path('imported_files.txt')):
                self.uploaded_db_check.configure(state=ctk.NORMAL)
            self.rag_fusion_var.set(options_config['rag_fusion_var'])
            self.hyde_var.set(options_config['hyde_var'])
            self.subqueries_var.set(options_config['subqueries_var'])
            if os.path.exists(resource_path('imported_files.txt')):
                self.default_db_var.set(options_config['default_db_var'])
                self.uploaded_db_var.set(options_config['uploaded_db_var'])
            else:
                self.default_db_var.set('default_db')
                self.uploaded_db_var.set('no_uploaded_db')
                self.uploaded_db_check.configure(state=ctk.DISABLED)
            self.display_resources_var.set(options_config['display_resources_var'])
            self.just_llm_var.set(options_config['just_llm_var'])
            self.always_rag_var.set(options_config['always_rag_var'])
            self.modify_query_var.set(options_config['modify_query_var'])
            self.translate_query_var.set(options_config['translate_query_var'])
            self.relevant_resource_llm_check_var.set(options_config['relevant_resource_llm_check_var'])
            self.topn_articles_var.set(options_config['topn_articles_var'])
            self.menu_language_var.set(options_config['menu_language_var'])
            if self.menu_language_var.get() == "bulgarian":
                self.change_language("bulgarian")

        elif options_config['just_llm_var'] == 'just_llm':
            self.rag_fusion_var.set('no_rag_fusion')
            self.hyde_var.set('no_hyde')
            self.subqueries_var.set('no_subq')
            self.default_db_var.set('no_default_db')
            self.uploaded_db_var.set('no_uploaded_db')
            self.display_resources_var.set('no_resources')
            self.just_llm_var.set(options_config['just_llm_var'])
            self.always_rag_var.set('no_always_rag')
            self.modify_query_var.set('no_modify')
            self.translate_query_var.set('no_translate')
            self.relevant_resource_llm_check_var.set('no_rel_llm_check')
            self.topn_articles_var.set(options_config['topn_articles_var'])
            self.menu_language_var.set(options_config['menu_language_var'])

            self.rag_fusion_check.configure(state=ctk.DISABLED)
            self.hyde_check.configure(state=ctk.DISABLED)
            self.subqueries_check.configure(state=ctk.DISABLED)
            self.default_db_check.configure(state=ctk.DISABLED)
            self.uploaded_db_check.configure(state=ctk.DISABLED)
            self.display_resources_check.configure(state=ctk.DISABLED)
            self.modify_query_check.configure(state=ctk.DISABLED)
            self.translate_query_check.configure(state=ctk.DISABLED)
            self.relevant_resource_llm_check_checkbox.configure(state=ctk.DISABLED)
            self.topn_articles_slider.configure(state=ctk.DISABLED)
            if self.menu_language_var.get() == "bulgarian":
                self.change_language("bulgarian")

    def update_options_config(self, on_exit=False):
        options_config_json = open(resource_path('options_config.json'), 'r+')
        options_config = json.load(options_config_json)
        if not on_exit or self.just_llm_var.get() == 'no_just_llm':
            options_config['rag_fusion_var'] = self.rag_fusion_var.get()
            options_config['hyde_var'] = self.hyde_var.get()
            options_config['subqueries_var'] = self.subqueries_var.get()
            options_config['default_db_var'] = self.default_db_var.get()
            options_config['uploaded_db_var'] = self.uploaded_db_var.get()
            options_config['display_resources_var'] = self.display_resources_var.get()
            options_config['always_rag_var'] = self.always_rag_var.get()
            options_config['modify_query_var'] = self.modify_query_var.get()
            options_config['translate_query_var'] = self.translate_query_var.get()
            options_config['relevant_resource_llm_check_var'] = self.relevant_resource_llm_check_var.get()
            options_config['topn_articles_var'] = self.topn_articles_var.get()

        options_config['menu_language_var'] = self.menu_language_var.get()
        options_config['just_llm_var'] = self.just_llm_var.get()

        options_config_json.seek(0)
        options_config_json.truncate(0)
        json.dump(options_config, options_config_json)
        options_config_json.close()

    def database_click(self):
        if self.default_db_var.get() == "no_default_db" and self.uploaded_db_var.get() == "no_uploaded_db":
            self.default_db_var.set("default_db")

    def change_language(self, language):
        if language == "bulgarian":
            self.advanced_rag_label.configure(text="Усъвършенстван RAG")
            self.subqueries_check.configure(text="Генериране на подзаявки")
            self.database_label.configure(text="База данни")
            self.default_db_check.configure(text="Основна")
            self.uploaded_db_check.configure(text="Качени\nфайлове")
            self.other_label.configure(text="Други")
            self.display_resources_check.configure(text="Показване на ресурсите")
            self.just_llm_check.configure(text="Само LLM\n(без RAG)")
            self.always_rag_check.configure(text="Винаги\nRAG")
            self.modify_query_check.configure(text="Промени заявката на база\nчата")
            self.translate_query_check.configure(text="Преведи заявката за\nмногоезични източници")
            self.relevant_resource_llm_check_checkbox.configure(text="Провери релевантните\nизточници с LLM")
            self.menu_language_label.configure(text="Език на менюто")
            self.move_topn_articles_slider(self.topn_articles_var.get())
        elif language == "english":
            self.advanced_rag_label.configure(text="Advanced RAG")
            self.subqueries_check.configure(text="Use generated subqueries")
            self.database_label.configure(text="Database")
            self.default_db_check.configure(text="Default")
            self.uploaded_db_check.configure(text="Uploaded files")
            self.other_label.configure(text="Other")
            self.display_resources_check.configure(text="Display resources")
            self.just_llm_check.configure(text="Use just the\nLLM (not RAG)")
            self.always_rag_check.configure(text="Use RAG\nalways")
            self.modify_query_check.configure(text="Modify query based on chat")
            self.translate_query_check.configure(text="Translate query for multilingual\nresources")
            self.relevant_resource_llm_check_checkbox.configure(text="Check relevant resources by LLM")
            self.menu_language_label.configure(text="Menu Language")
            self.move_topn_articles_slider(self.topn_articles_var.get())

    def just_llm_click(self):
        if self.just_llm_var.get() == 'just_llm':
            self.update_options_config()
            self.rag_fusion_var.set('no_rag_fusion')
            self.hyde_var.set('no_hyde')
            self.subqueries_var.set('no_subq')
            self.default_db_var.set('no_default_db')
            self.uploaded_db_var.set('no_uploaded_db')
            self.display_resources_var.set('no_resources')
            self.always_rag_var.set('no_always_rag')
            self.modify_query_var.set('no_modify')
            self.translate_query_var.set('no_translate')
            self.relevant_resource_llm_check_var.set('no_rel_llm_check')

            self.rag_fusion_check.configure(state=ctk.DISABLED)
            self.hyde_check.configure(state=ctk.DISABLED)
            self.subqueries_check.configure(state=ctk.DISABLED)
            self.default_db_check.configure(state=ctk.DISABLED)
            self.uploaded_db_check.configure(state=ctk.DISABLED)
            self.display_resources_check.configure(state=ctk.DISABLED)
            self.modify_query_check.configure(state=ctk.DISABLED)
            self.translate_query_check.configure(state=ctk.DISABLED)
            self.relevant_resource_llm_check_checkbox.configure(state=ctk.DISABLED)
            self.topn_articles_slider.configure(state=ctk.DISABLED)
        else:
            self.rag_fusion_check.configure(state=ctk.NORMAL)
            self.hyde_check.configure(state=ctk.NORMAL)
            self.subqueries_check.configure(state=ctk.NORMAL)
            self.default_db_check.configure(state=ctk.NORMAL)
            self.uploaded_db_check.configure(state=ctk.NORMAL)
            self.display_resources_check.configure(state=ctk.NORMAL)
            self.modify_query_check.configure(state=ctk.NORMAL)
            self.translate_query_check.configure(state=ctk.NORMAL)
            self.relevant_resource_llm_check_checkbox.configure(state=ctk.NORMAL)
            self.topn_articles_slider.configure(state=ctk.NORMAL)
            self.read_options_config(just_llm_unchecked=True)

    def always_rag_click(self):
        if self.always_rag_var.get() == 'always_rag':
            if self.just_llm_var.get() == 'just_llm':
                self.just_llm_var.set('no_just_llm')
                self.just_llm_click()
                self.always_rag_var.set('always_rag')

    def move_topn_articles_slider(self, value):
        if self.menu_language_var.get() == "english":
            self.topn_articles_label.configure(text=f"Top n relevant chunks for context: {int(value)}")
        elif self.menu_language_var.get() == "bulgarian":
            self.topn_articles_label.configure(text=f"Брой релевантни откъси: {int(value)}")


class ImportFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(expand=True, fill='both')
        self.text_font = ctk.CTkFont(family='Helvetica', size=16)
        self.choose_files_button = ctk.CTkButton(master=self, text='Choose files', font=self.text_font,
                                                 state=ctk.DISABLED, command=self.choose_files)
        self.choose_files_button.place(relx=0.5, rely=0.07, relwidth=0.7, relheight=0.06, anchor='center')
        self.selected_files = []
        self.selected_filenames = []
        self.imported_filenames = []
        self.time_to_stop_import_animation = False
        self.import_button_clicked = ctk.BooleanVar(value=False)
        self.import_button = ctk.CTkButton(master=self, text='Import', font=self.text_font, fg_color='green',
                                           hover_color='darkgreen', state=ctk.DISABLED,
                                           command=lambda: self.import_button_clicked.set(True))
        self.import_button.place(relx=0.5, rely=0.17, relwidth=0.7, relheight=0.06, anchor='center')

        self.delete_button_confirmed = ctk.BooleanVar(value=False)
        self.delete_button = ctk.CTkButton(master=self, text='Delete database', font=self.text_font, fg_color='red',
                                           hover_color='darkred', state=ctk.DISABLED, command=self.delete_files)
        self.delete_button.place(relx=0.5, rely=0.27, relwidth=0.7, relheight=0.06, anchor='center')

        self.info_text = ctk.StringVar(value="Click on 'Choose files' if you want\nto add files to the RAG database.")
        self.info_label = ctk.CTkLabel(master=self, textvariable=self.info_text, wraplength=225, fg_color='transparent',
                                       text_color='yellow',
                                       font=ctk.CTkFont(family='Helvetica', size=14), justify='left')
        self.info_label.place(relx=0.5, rely=0.37, relwidth=0.98, relheight=0.12, anchor='center')

        self.selected_files_label = ctk.CTkLabel(master=self, text='Selected', wraplength=225,
                                                 text_color='#34abeb',
                                                 font=ctk.CTkFont(family='Helvetica', weight='bold', size=16),
                                                 justify='left')
        self.selected_files_label.place(relx=0.5, rely=0.47, relwidth=0.44, relheight=0.05, anchor='center')
        self.clear_selected_files_button = ctk.CTkButton(master=self, text='Clear',
                                                         font=ctk.CTkFont(family='Helvetica', size=14),
                                                         fg_color='orange', hover_color='darkorange',
                                                         state=ctk.DISABLED,
                                                         command=self.clear_selected_files)
        self.clear_selected_files_button.place(relx=0.86, rely=0.47, relwidth=0.27, relheight=0.05, anchor='center')
        self.selected_files_frame = ctk.CTkScrollableFrame(master=self)
        self.selected_files_frame.place(relx=0.5, rely=0.6, relwidth=1, relheight=0.2, anchor='center')
        self.imported_files_label = ctk.CTkLabel(master=self, text='Imported', wraplength=225,
                                                 text_color='#34eb43',
                                                 font=ctk.CTkFont(family='Helvetica', weight='bold', size=16),
                                                 justify='left')
        self.imported_files_label.place(relx=0.5, rely=0.75, relwidth=0.64, relheight=0.05, anchor='center')
        self.imported_files_frame = ctk.CTkScrollableFrame(master=self)
        self.imported_files_frame.place(relx=0.5, rely=0.88, relwidth=1, relheight=0.2, anchor='center')

        self.check_for_previously_imported_files()

        options_config_json = open(resource_path('options_config.json'), 'r')
        options_config = json.load(options_config_json)
        options_config_json.close()
        self.menu_language = options_config["menu_language_var"]
        if self.menu_language == "bulgarian":
            self.change_language("bulgarian")

    def check_for_previously_imported_files(self):
        if os.path.exists(resource_path('imported_files.txt')):
            imported_files_txt = open(resource_path('imported_files.txt'), 'r', encoding="utf-8")
            self.imported_filenames = imported_files_txt.read().splitlines()
            self.selected_filenames = self.imported_filenames
            self.add_files_labels(self.imported_files_frame)
            self.selected_filenames = []
            imported_files_txt.close()

    def choose_files(self):
        filetypes = (
            ('pdf files', '*.pdf'),
            ('doc files', '*.doc'),
            ('docx files', '*.docx'),
            ('json files', '*.json'),
            ('text files', '*.txt')
        )
        self.selected_filenames = []
        self.selected_files += fd.askopenfiles(title='Select files', filetypes=filetypes, initialdir=str(Path.home()))
        for file in self.selected_files:
            self.selected_filenames.append(os.path.basename(file.name))
        if self.selected_files:
            self.clear_selected_files_button.configure(state=ctk.NORMAL)
            self.info_label.configure(font=ctk.CTkFont(family='Helvetica', size=14), text_color='yellow')
            if self.menu_language == "english":
                self.info_text.set("To import the selected files, click 'Import'.")
            elif self.menu_language == "bulgarian":
                self.info_text.set("За да качиш избраните файлове, натисни 'Качи'.")
            for label in self.selected_files_frame.winfo_children():
                label.destroy()
            self.add_files_labels(frame=self.selected_files_frame)
            self.import_button.configure(state=ctk.NORMAL)

    def import_files_animation(self):
        if self.time_to_stop_import_animation:
            self.imported_filenames = self.selected_filenames
            diff = len(self.selected_files) - len(self.imported_filenames)

            self.add_files_labels(frame=self.imported_files_frame)
            for label in self.selected_files_frame.winfo_children():
                label.destroy()
            self.selected_files = []
            self.selected_filenames = []
            self.import_button.configure(state=ctk.DISABLED)
            self.delete_button.configure(state=ctk.NORMAL)
            self.choose_files_button.configure(state=ctk.NORMAL)
            self.clear_selected_files_button.configure(state=ctk.DISABLED)
            self.info_label.configure(font=ctk.CTkFont(family='Helvetica', size=13))

            if diff > 0:
                self.info_label.configure(text_color='#fc2a05')
                if len(self.imported_filenames) == 0:
                    if self.menu_language == "english":
                        self.info_text.set(
                            "The files were NOT imported! There was a problem with the reading of all of them.")
                    elif self.menu_language == "bulgarian":
                        self.info_text.set(
                            "Файловете НЕ бяха качени! Имаше проблеми с прочитането на всички файлове.")
                else:
                    if self.menu_language == "english":
                        self.info_text.set(
                            f"With {diff} files there was a PROBLEM. Check out 'Log' for more information.")
                    elif self.menu_language == "bulgarian":
                        self.info_text.set(
                            f"С {diff} файла имаше ПРОБЛЕМ! Вижте 'Лог' за повече информация.")
                self.time_to_stop_import_animation = False
                return
            self.info_label.configure(text_color='#34eb43')
            if self.menu_language == "english":
                self.info_text.set("Files imported successfully! Check the 'Uploaded files' database in 'Options'")
            elif self.menu_language == "bulgarian":
                self.info_text.set("Файловете са качени успешно! Изберете базата данни 'Качени файлове' в 'Опции'")
            self.time_to_stop_import_animation = False
            return

        if self.menu_language == "bulgarian":
            dots_str = self.info_text.get()[20:23]
        else:
            dots_str = self.info_text.get()[15:18]
        if dots_str == '...':
            dots_str = '   '
        elif dots_str == '   ':
            dots_str = '.  '
        elif dots_str == '.  ':
            dots_str = '.. '
        elif dots_str == '.. ':
            dots_str = '...'

        new_info_text_str = ""
        if self.menu_language == "english":
            new_info_text_str = 'Importing files' + dots_str + '\nIt may take a few minutes.'
        elif self.menu_language == "bulgarian":
            new_info_text_str = 'Качване на файловете' + dots_str + '\nМоже да отнеме няколко минути.'

        self.info_label.configure(font=ctk.CTkFont(family='Helvetica', size=14))
        self.info_text.set(new_info_text_str)
        self.after(1000, self.import_files_animation)

    def delete_files(self):
        if self.menu_language == "bulgarian":
            confirm_deletion = messagebox.askyesno(title='Изтриване на база данни от качени файлове',
                                                   message='Сигурни ли сте, че искате да изтриете базата данни, '
                                                           'създадена от Вашите качени файлове?')
        else:
            confirm_deletion = messagebox.askyesno(title='Deletion of imported files database',
                                                   message='Are you sure you want to delete the database created from '
                                                           'your imported files?')
        if confirm_deletion:
            self.delete_button.configure(state=ctk.DISABLED)
            self.delete_button_confirmed.set(True)
            for label in self.imported_files_frame.winfo_children():
                label.destroy()
            for label in self.selected_files_frame.winfo_children():
                label.destroy()
            self.selected_files = []
            self.selected_filenames = []
            self.imported_filenames = []
            self.clear_selected_files_button.configure(state=ctk.DISABLED)
            self.info_label.configure(font=ctk.CTkFont(family='Helvetica', size=14), text_color='yellow')
            if self.menu_language == "english":
                self.info_text.set("Click on 'Choose files' if you want to add files to the RAG database.")
            elif self.menu_language == "bulgarian":
                self.info_text.set("Натиснете 'Избери файлове', ако искате да добавите файлове към базата на RAG.")

    def add_files_labels(self, frame):
        for filename in self.selected_filenames:
            if frame == self.selected_files_frame:
                text_color = '#34abeb'
            else:
                text_color = '#34eb43'
            label = ctk.CTkLabel(master=frame, text=filename, wraplength=200, text_color=text_color, anchor='w',
                                 width=200,
                                 font=ctk.CTkFont(family='Helvetica', weight='bold', size=12), justify='left')
            label.pack(side=ctk.TOP, pady=5)

    def clear_selected_files(self):
        self.selected_files = []
        self.selected_filenames = []
        for label in self.selected_files_frame.winfo_children():
            label.destroy()
        self.clear_selected_files_button.configure(state=ctk.DISABLED)
        self.import_button.configure(state=ctk.DISABLED)
        if self.menu_language == "english":
            self.info_text.set("Click on 'Choose files' if you want to add files to the RAG database.")
        elif self.menu_language == "bulgarian":
            self.info_text.set("Натиснете 'Избери файлове', ако искате да добавите файлове към базата на RAG.")

    def change_language(self, new_language):
        self.info_label.configure(text_color='yellow')
        if new_language == "english":
            self.menu_language = "english"
            self.choose_files_button.configure(text="Choose files")
            self.import_button.configure(text="Import")
            self.delete_button.configure(text='Delete database')
            self.info_text.set("Click on 'Choose files' if you want to add files to the RAG database.")
            self.selected_files_label.configure(text='Selected')
            self.clear_selected_files_button.configure(text='Clear')
            self.imported_files_label.configure(text='Imported')
        elif new_language == "bulgarian":
            self.menu_language = "bulgarian"
            self.choose_files_button.configure(text="Избери файлове")
            self.import_button.configure(text="Качи")
            self.delete_button.configure(text='Изтрий базата данни')
            self.info_text.set("Натисни на 'Избери файлове',за да качиш файлове в базата данни на RAG.")
            self.selected_files_label.configure(text='Избрани')
            self.clear_selected_files_button.configure(text='Изчисти')
            self.imported_files_label.configure(text='Качени')


class SaveFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(expand=True, fill='both')

        self.help_info_label = ctk.CTkLabel(master=self, text='You can save the current chat conversation as a text '
                                                              '(*.txt) file in a chosen directory.',
                                            wraplength=225, fg_color='transparent', text_color='yellow',
                                            font=ctk.CTkFont(family='Helvetica', size=14), justify='left')
        self.help_info_label.place(relx=0.5, rely=0.1, relwidth=0.98, relheight=0.12, anchor='center')

        self.chosen_directory = ctk.StringVar(value='')
        self.choose_directory_button = ctk.CTkButton(master=self, text='Choose directory',
                                                     font=ctk.CTkFont(family='Helvetica', size=16),
                                                     command=self.choose_directory)
        self.choose_directory_button.place(relx=0.5, rely=0.25, relwidth=0.7, relheight=0.06, anchor='center')

        self.chosen_directory_label = ctk.CTkLabel(master=self, textvariable=self.chosen_directory,
                                                   wraplength=225, fg_color='transparent', text_color='#34abeb',
                                                   font=ctk.CTkFont(family='Helvetica', size=14), justify='center')
        self.chosen_directory_label.place(relx=0.5, rely=0.34, relwidth=0.98, relheight=0.12, anchor='center')

        self.filename = ''
        self.filename_input_string = ctk.StringVar(value='')
        self.filename_input_string.trace('w', self.update_filename)

        self.filename_entry = ctk.CTkEntry(master=self, textvariable=self.filename_input_string, state=ctk.DISABLED,
                                           font=ctk.CTkFont(family='Helvetica', size=14))
        self.filename_entry.place(relx=0.5, rely=0.46, relwidth=0.98, relheight=0.06, anchor='center')

        self.filename_help_label = ctk.CTkLabel(master=self, text='Filename', fg_color='transparent',
                                                font=ctk.CTkFont(family='Helvetica', size=14), justify='center')
        self.filename_help_label.place(relx=0.5, rely=0.54, relwidth=0.98, relheight=0.08, anchor='center')

        self.filename_label = ctk.CTkLabel(master=self, text='', fg_color='transparent', wraplength=225,
                                           text_color='yellow',
                                           font=ctk.CTkFont(family='Helvetica', size=14), justify='center')
        self.filename_label.place(relx=0.5, rely=0.59, relwidth=0.98, relheight=0.06, anchor='center')

        self.add_resources_var = ctk.StringVar(value="add_resources")
        self.add_resources_check = ctk.CTkCheckBox(self, text="Add relevant resources to file",
                                                   variable=self.add_resources_var, onvalue='add_resources',
                                                   offvalue='no_add_resources')
        self.add_resources_check.place(relx=0.04, rely=0.65, relwidth=0.92, relheight=0.06, anchor='nw')

        self.save_button = ctk.CTkButton(master=self, text='Save', fg_color='green', state=ctk.DISABLED,
                                         hover_color='darkgreen',
                                         font=ctk.CTkFont(family='Helvetica', size=16), command=self.save)
        self.save_button.place(relx=0.5, rely=0.78, relwidth=0.6, relheight=0.06, anchor='center')

        self.save_button_clicked = ctk.BooleanVar(value=False)
        self.save_info_label = ctk.CTkLabel(master=self, text='', fg_color='transparent', wraplength=225,
                                            text_color='#34eb43',
                                            font=ctk.CTkFont(family='Helvetica', size=14), justify='center')
        self.save_info_label.place(relx=0.5, rely=0.86, relwidth=0.98, relheight=0.06, anchor='center')
        self.successful_save = True

        options_config_json = open(resource_path('options_config.json'), 'r')
        options_config = json.load(options_config_json)
        options_config_json.close()
        self.menu_language = options_config["menu_language_var"]
        if self.menu_language == "bulgarian":
            self.change_language("bulgarian")

    def choose_directory(self):
        self.chosen_directory.set(fd.askdirectory(initialdir=str(Path.home())))
        if self.chosen_directory.get():
            self.filename_entry.configure(state=ctk.NORMAL)
            self.save_button.configure(state=ctk.NORMAL)
            self.save_info_label.configure(text='')

    def update_filename(self, *args):
        if self.filename_input_string.get():
            self.filename = self.filename_input_string.get().replace(' ', '_') + '.txt'
            self.filename_label.configure(text=self.filename)
        else:
            self.filename_label.configure(text='')

    def save(self):
        self.save_button_clicked.set(True)
        if self.successful_save:
            if self.menu_language == "english":
                self.save_info_label.configure(text='Chat conversation saved successfully!', text_color='#34eb43')
            elif self.menu_language == "bulgarian":
                self.save_info_label.configure(text='Чат разговорът беше запазен успешно!', text_color='#34eb43')
        else:
            if self.menu_language == "english":
                self.save_info_label.configure(text='ERROR! Chat conversation was NOT saved!', text_color='#fc2a05')
            elif self.menu_language == "bulgarian":
                self.save_info_label.configure(text='ГРЕШКА! Чат разговорът НЕ беше запазен!', text_color='#fc2a05')
        self.after(5000, lambda: self.save_info_label.configure(text=''))

    def change_language(self, new_language):
        self.menu_language = new_language
        if new_language == "english":
            self.help_info_label.configure(text="You can save the current chat conversation as a text (*.txt) file in "
                                                "a chosen directory.")
            self.choose_directory_button.configure(text="Choose directory")
            self.filename_help_label.configure(text="Filename")
            self.add_resources_check.configure(text="Add relevant resources to file")
            self.save_button.configure(text="Save")

        elif new_language == "bulgarian":
            self.help_info_label.configure(text="Може да запазите текущия чат разговор като текстов (*.txt) файл в "
                                                "избрана директория.")
            self.choose_directory_button.configure(text="Избери директория")
            self.filename_help_label.configure(text="Име на файла")
            self.add_resources_check.configure(text="Добави релевантните ресурси\nкъм файла")
            self.save_button.configure(text="Запази")


class LogFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(expand=True, fill='both')
        self.text_font = ctk.CTkFont(family='Helvetica', size=14)
        self.log_help_label = ctk.CTkLabel(master=self, text='Here you can see what happens in the background of the '
                                                             'app.',
                                           fg_color='transparent', text_color='yellow', font=self.text_font,
                                           justify='left', wraplength=225)
        self.log_textbox = ctk.CTkTextbox(master=self, font=self.text_font, wrap='word', state=ctk.DISABLED)
        self.log_relevant_chunks_var = ctk.StringVar(value="log_chunks")
        self.log_relevant_chunks_check = ctk.CTkCheckBox(self, text="Show relevant chunks",
                                                         variable=self.log_relevant_chunks_var, onvalue='log_chunks',
                                                         offvalue='no_log_chunks')

        self.log_help_label.place(relx=0.01, rely=0, relwidth=0.98, relheight=0.06, anchor='nw')
        self.log_relevant_chunks_check.place(relx=0.04, rely=0.07, relwidth=0.96, relheight=0.06, anchor='nw')
        self.log_textbox.place(relx=0, rely=0.14, relwidth=1, relheight=0.85, anchor='nw')

        options_config_json = open(resource_path('options_config.json'), 'r')
        options_config = json.load(options_config_json)
        options_config_json.close()
        self.menu_language = options_config["menu_language_var"]
        if self.menu_language == "bulgarian":
            self.change_language("bulgarian")

        text_handler = TextHandler(self.log_textbox)
        logging.basicConfig(filename=resource_path('rag_log.log'), filemode='a', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logger = logging.getLogger()
        logger.addHandler(text_handler)

    def change_language(self, new_language):
        self.menu_language = new_language
        if new_language == "english":
            self.log_help_label.configure(text="Here you can see what happens in the background of the app.")
            self.log_relevant_chunks_check.configure(text="Show relevant chunks")
        elif new_language == "bulgarian":
            self.log_help_label.configure(text="Тук може да видите какво се случва зад кадър в системата.")
            self.log_relevant_chunks_check.configure(text="Покажи релевантните откъси")

    def log(self, text):
        self.log_textbox.configure(state=ctk.NORMAL)
        self.log_textbox.insert(ctk.END, text + '\n')
        self.log_textbox.configure(state=ctk.DISABLED)

    def delete_log(self):
        self.log_textbox.configure(state=ctk.NORMAL)
        self.log_textbox.delete(0.0, 'end')
        self.log_textbox.configure(state=ctk.DISABLED)


class HelpFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(expand=True, fill='both')
        self.text_font = ctk.CTkFont(family='Helvetica', size=14)
        self.help_textbox = ctk.CTkTextbox(master=self, font=self.text_font, wrap='word', state=ctk.DISABLED)
        self.help_textbox.place(relx=0, rely=0, relwidth=1, relheight=1, anchor='nw')

        options_config_json = open(resource_path('options_config.json'), 'r')
        options_config = json.load(options_config_json)
        options_config_json.close()
        self.menu_language = options_config["menu_language_var"]
        if self.menu_language == "bulgarian":
            self.add_bulgarian_help()
        elif self.menu_language == "english":
            self.add_english_help()

    def add_text(self, text):
        self.help_textbox.configure(state=ctk.NORMAL)
        self.help_textbox.insert(ctk.END, text + '\n')
        self.help_textbox.configure(state=ctk.DISABLED)

    def change_language(self, new_language):
        self.menu_language = new_language
        self.help_textbox.configure(state=ctk.NORMAL)
        self.help_textbox.delete(0.0, 'end')
        if new_language == "bulgarian":
            self.add_bulgarian_help()
        elif self.menu_language == "english":
            self.add_english_help()
        self.help_textbox.configure(state=ctk.DISABLED)

    def add_bulgarian_help(self):
        self.add_text("Добре дошли в приложението BG RAG!\n"
                      "Това е RAG система, която работи добре както с ресурси на български, така и на английски език.\n"
                      "Използва LLM модела „INSAIT-Institute/BgGPT-7B-Instruct-v0.2-GGUF“, създаден от INSAIT.\n"
                      "За да използвате приложението в пълния му капацитет, трябва да инсталирате llama.cpp и "
                      "CUDA Toolkit на вашия компютър.\n\n"
                      "Обща информация за системата:\n\n"
                      "Потребителят може да копира отговор на системата или заявка, като кликне върху текста в чата."
                      "Ако показаният ресурс е url, може също да се щракне върху него и връзката ще се отвори в "
                      "браузъра по подразбиране на потребителя. \n\n"
                      "Системата RAG е основно система, която използва LLM за генериране на отговори на потребителски "
                      "запитвания, "
                      "но подобрява фактологията и достоверността на отговорите, като дава контекст на LLM (под форма "
                      "на текстови откъси) от релевантни ресурси, извлечени от база данни с текстови документи.\n"
                      "Най-важният аспект на системата RAG е процесът на извличане, където релевантни към "
                      "потребителската заявка откъси от текст трябва да бъдат намерени от база данни с текстови "
                      "документи. В тази система за тази цел са използвани както векторно търсене, така и търсене по "
                      "ключови думи, като резултатите се комбинират с реципрочния ранг алгоритъма Reciprocal Rank "
                      "Fusion (RRF) за определяне на крайните релевантни откъси към заявката. Може също да се "
                      "използват множество различни техники за подобряване на извличането на релевантни ресурси, "
                      "налични в раздела 'Опции'.\n\n"
                      "От редица експерименти може да се сподели, че най-добрите техники, които трябва да се използват "
                      "за по-добро извличане, са RAG Fusion и HyDE заедно. Ако потребителят иска да използва само една "
                      "техника, за да намали времето, необходимо за генериране на отговор, тогава се препоръчва RAG "
                      "Fusion. Опцията за генерирани подзаявки може да се използва за по-сложни заявки в зависимост от "
                      "предпочитанията на потребителя (може също да се комбинира с другите техники). Препоръчва се "
                      "модифицирането на заявка въз основа на чат разговора да бъде включено. Препоръчителният "
                      "диапазон за топ n най-релевантни откъси е от 3 до 5. Използването на LLM за проверка на "
                      "ресурсите може да бъде полезно за по-обстойно филтриране на върнатите ресурси от системата RAG "
                      "и може да подобри резултатите. Преводът на заявката трябва да се използва за "
                      "многоезични ресурси в базата данни от качени файлове от потребителя или ако потребителят е "
                      "избрал да използва и двете бази данни и качените файлове са на английски език (базата данни по "
                      "подразбиране е на български и превеждането на заявките е необходимо за търсенето по ключови "
                      "думи).\n\n\n"
                      "По-подробни инструкции и полезна информация за използването на приложението и менюто:\n\n\n"
                      "Раздел 'Качи':\n\n"
                      "В този раздел потребителят може да качва свои собствени файлове, които могат да се използват "
                      "като база данни за RAG системата.\n"
                      "Първо, потребителят трябва да избере файловете, като кликне върху 'Избиране на файлове'. "
                      "Възможните видове файлове за избиране са pdf, doc, docx, json, txt. След като потребителят "
                      "избере файловете, те ще бъдат показани в секцията 'Избрани'. Ако потребителят е направил грешка "
                      "с някои от избраните файлове, може да щракне върху 'Изчисти', за да премахне избраните файлове "
                      "и да избере нови.\n"
                      "На второ място, след като потребителят е доволен от избраните файлове, трябва да щракне върху "
                      "бутона 'Качи', за да могат файловете да бъдат качени в базата данни от качени файлове. Ако "
                      "импортирането е успешно, файловете ще станат видими в секцията 'Качени'. Ако има проблем с "
                      "някои от файловете (напр. те не могат да бъдат прочетени), проблемните файлове ще бъдат "
                      "игнорирани и няма да бъдат импортирани. Останалите файлове само ще бъдат качени в базата.\n"
                      "СЛЕД ИМПОРТИРАНЕТО ПОТРЕБИТЕЛЯТ ТРЯБВА ДА ИЗБЕРЕ БАЗАТА ДАННИ ОТ КАЧЕНИ ФАЙЛОВЕ В РАЗДЕЛА "
                      "'ОПЦИИ', ЗА ДА СЕ ИЗПОЛЗВАТ КАЧЕНИТЕ ФАЙЛОВЕ ОТ RAG СИСТЕМАТА.\n"
                      "Базата данни с качените файлове се запазва и ще бъде заредена при следващото стартиране на "
                      "приложението.\n"
                      "Потребителят може да изтрие базата данни, създадена от качените файлове, като щракне върху "
                      "'Изтриване на база данни' и потвърди изтриването.\n"
                      "Качените файлове могат да бъдат както на български, така и на английски език. Ако има файлове "
                      "от двата езика, тогава потребителят трябва да избере опцията 'Превод на заявка за многоезични "
                      "ресурси' в раздела 'Опции'. Тази опция трябва да се избере и ако са избрани и двете бази данни "
                      "да се използват едновременно и качените файлове са на английски език.\n\n\n"
                      "Раздел 'Запази':\n\n"
                      "В този раздел потребителят може да запише текущия разговор в чата във файл в избрана "
                      "директория."
                      "Първо, потребителят трябва да избере директория, в която ще бъде разположен файлът. "
                      "Второ, потребителят трябва да посочи името на файла. Интервалите в името се заменят с '_' "
                      "(знак за долна черта) и типът на файла за запис е .txt. Ако името на файла, въведено от "
                      "потребителя, е същото име като съществуващ файл в избраната директория, тогава чат разговорът "
                      "ще се добави в края на съществуващия файл. По този начин текущият разговор в чата може да "
                      "се добави към запис на предишен чат разговор. Освен това, тъй като системата взема предвид "
                      "запазените чат итерации, ако потребителят, след редица нови чат итерации, иска да запази "
                      "разширеният чат отново, само новите и незапазените итерации на чата ще бъдат запазени."
                      "Тази функция е направена с идеята, че потребителят може да иска да разшири текущия запис на "
                      "чата с продължението на досегашния разговор без да се презаписва отново записания вече "
                      "разговор.\n"
                      "Налична е опция 'Добавяне на релевантни ресурси', която може да бъде включена, ако потребителят "
                      "иска да запази релевантните ресурси, върнати от системата RAG за всяка заявка и отговор на "
                      "системата.\n\n\n"
                      "Раздел 'Лог':\n\n"
                      "Този раздел е посочен като разделът по подразбиране, когато приложението се стартира, тъй като "
                      "дава информация за процесът на зареждане на базите данни, ембединг (embedding) модела и LLM. "
                      "По време на зареждането потребителят не може да изпраща съобщения до системата. След като "
                      "зареждането приключи, подходящо съобщение се показва в раздела 'Лог' и потребителят може да "
                      "използва системата нормално. В този раздел потребителят може да "
                      "види какво се случва във фонов режим на приложението въз основа на избраните техники и опции. "
                      "Там също ще бъдат показани грешки и изключения, ако такива са възникнали по време на "
                      "използването на системата."
                      "Освен това съдържанието, показано в раздела, се съхранява като лог файл с името 'rag_log.log' в "
                      "инсталационна директория на системата. Има опция 'Покажи релевантните откъси', която отпечатва "
                      "в лога откъсите от текст, които RAG системата е извлекла от базата данни като релевантни към "
                      "потребителската заявка."
                      "По подразбиране тази опция е изключена.\n\n\n"
                      "Раздел 'Опции':\n\n"
                      "Това е разделът, който позволява на потребителя да използва различни техники за извличане на "
                      "релевантни ресурси (откъси) за заявката, както и да зададе предпочитаната база данни или бази "
                      "данни.\n\n"
                      "Секция 'Усъвършенстван RAG':\n"
                      "Потребителят може да посочи кои техники за извличане да използва.\n"
                      "RAG Fusion е техника, която използва LLM, за да генерира 4 заявки, подобни на оригиналната, "
                      "и за всяка от тях (включително оригиналната) системата намира подходящи ресурси (използвайки "
                      "както векторно търсене, така и търсене по ключови думи). След това резултатите се комбинират с "
                      "алгоритъма Reciprocal Rank Fusion (RRF) за извличане на релевантните текстови откъси. "
                      "Това обикновено е най-добрата единична техника, която да се използва за подобряване на "
                      "процеса на извличане на релевантни ресурси в приложението.\n\n"
                      "HyDE е техника, която също използва LLM, но по различен начин. LLM отговаря на потребителското "
                      "запитване без никакъв контекст и след това използва отговора, за да извлече релевантни откъси "
                      "(ресурси) за заявката, използвайки векторно търсене и търсене по ключови думи. В тази система, "
                      "когато опцията HyDE е включена, векторно търсене и търсене по ключови думи се използва за "
                      "извличането на подходящи откъси и за оригиналната заявка."
                      "Най-подходящите откъси, намерени за отговора на HyDE от LLM и за оригиналната заявка, се "
                      "комбинират с алгоритъма RRF за извличане на последните топ n подходящи текстови откъси. Тази "
                      "техника може да бъде поставена на второ място като единична техника за подобряване на процеса "
                      "на извличане. За най-добри резултати при процеса на извличане е препоръчително да се използват "
                      "както HyDE, така и RAG Fusion, дори ако за това е необходимо малко повече време "
                      "за генерирането на отговор от системата. Те могат дори да се комбинират със следващата опция - "
                      "техника на генериране на подзаявки.\n\n"
                      "Опцията за генерирани подзаявки също използва LLM, за да раздели сложна заявка на няколко "
                      "по-малки и самодостатъчни заявки, за да се намерят най-добрите ресурси за всичките отделни "
                      "части на заявката. Ако опцията е включена, тогава се извършва проверка дали потребителската "
                      "заявка е сложна или не (тя е проста) и ако е сложна, LLM разделя потребителската заявка на "
                      "няколко по-прости заявки. Те се модифицират, за да бъдат самодостатъчни и да запазят контекста, "
                      "така че всяка една от подзаявките да може да се използва в процеса на извличане за намиране на "
                      "релевантни ресурси."
                      "Най-релевантните откъси за всяка подзаявка, както и оригиналната, се комбинират с "
                      "RRF алгоритъма за получаване на окончателните релевантни откъси, които се предоставят на LLM "
                      "като контекст, който да бъде използван за генерирането на отговор на потребителското запитване. "
                      "Личен избор е дали да се използва тази опция или не, тъй като понякога може да не предоставя "
                      "достатъчно контекст в подзаявките. Следете генерираните подзаявки в раздела 'Лог' и решете дали "
                      "да използвате опцията или не.\n\n"
                      "'База данни':\n"
                      "Потребителят може да избере базата данни по подразбиране (създадена от българската Уикипедия и "
                      "българския новинарски сайт focus news) или база данни от качени файлове на потребителя. "
                      "Опцията за база данни с качени файлове е достъпна само ако потребителят е импортирал файлове в "
                      "системата.\n"
                      "Базата данни по подразбиране и базата данни с качени файлове могат да се използват заедно. "
                      "При използването на двете бази данни, ако качените файлове са на английски език, трябва да се "
                      "избере опцията 'Превод на заявка за многоезични ресурси'. Тя трябва да се избере и когато "
                      "качените файлове от потребителя са едновременно на български и английски език.\n\n"
                      "'Други':\n"
                      "'Показване на ресурсите' – дали да се показват извлечените подходящи ресурси под отговора на "
                      "заявката или не. Ако опцията е изключена, тогава релевантните ресурси могат да се видят в "
                      "раздела 'Лог' и лог файла. Ако опцията е включена, съответните ресурси, "
                      "намерени за потребителската заявка ще бъдат показани под отговора на заявката. Ако ресурсът е "
                      "url линк от базата данни по подразбиране, върху него може да се щракне и уеб страницата на "
                      "ресурса ще се зареди и ще се покаже в уеб браузъра по подразбиране на потребителя.\n\n"
                      "'Само LLM (без RAG)' – опция за използване само на LLM, без да му се предоставя контекст от "
                      "релевантни откъси за заявката от базата данни с текстови документи. Тази опция позволява на "
                      "потребителите да използват само LLM, без системата RAG. Когато тази опция е включена , RAG "
                      "опциите са недостъпни, тъй като не са необходими. Ако опцията е изключена, "
                      "RAG опциите са отново достъпни и са конфигурирани по начина, по който са били преди включването "
                      "на опцията.\n\n"
                      "'Винаги RAG' – позволява на потребителя да зададе опцията за използване на RAG за всяка заявка. "
                      "Ако опцията е изключена, тогава системата ще реши дали RAG е необходима за заявката или не. "
                      "RAG така или иначе се използва почти винаги, независимо от това обаче, ако системата реши, че "
                      "има достатъчно знания, може да не използва RAG. Най-важното е, че RAG няма да се използва, "
                      "ако въведеният текст от потребителя се разпознае като общ коментар (като 'Благодаря', 'Здравей' "
                      "и т.н.), а не като запитване. Препоръчително е да бъде изключена тази опция, освен ако "
                      "за някоя заявка системата реши, че няма нужда от RAG, а потребителят желае да използва RAG.\n\n"
                      "'Промени заявката на база чата' – ако е включена, потребителската заявка се променя "
                      "въз основа на текущия чат разговор, за да предостави на потребителската заявка достатъчен "
                      "контекст, необходим за процеса на извличане на релевантни откъси. Ако заявката се счете за "
                      "самодостатъчна и имаща достатъчно контекст от системата, тогава заявката не се променя. Опцията "
                      "е силно препоръчително да бъде включена и да се изключва само ако потребителят не харесва "
                      "модификацията на определена заявка и иска да я изпрати непроменена към системата RAG. Следете "
                      "модифицираната заявка в раздела 'Лог'.\n\n"
                      "'Преведи заявката за многоезични източници' – превежда потребителската заявка на другия език"
                      "(на английски, ако заявката е на български и обратно) за многоезични ресурси (ако "
                      "качените файлове от потребителя са и на двата езика или потребителят иска да използва както "
                      "основната, така и базата данни от качени файлове, но последната има документи на английски). "
                      "Преводът също се извършва за всяка генерирана заявка в процеса на извличане от избраните "
                      "техники (RAG Fusion, HyDE, генерирани подзаявки) и преводите могат да се видят в раздела 'Лог'. "
                      "Тези преводи са необходими за търсене по ключови думи на релевантни откъси (те също помагат и "
                      "за векторното търсене, но само малко, тъй като ембединг (embedding) модела, използван в "
                      "системата, е многоезичен). Не забравяйте, че RAG системата, внедрена в приложението, използва "
                      "както търсене по ключови думи, така и векторно търсене за извличане на релевантни "
                      "откъси и комбинира резултатите с алгоритъма Reciprocal Rank Fusion (RRF).\n\n"
                      "'Провери релевантните източници с LLM' – тази опция използва LLM за филтриране на извлечените "
                      "откъси от системата RAG. LLM решава дали извлечените откъси наистина съдържат "
                      "информация, свързана със заявката и ако някой откъс не съдържа такава, той се отхвърля и не се "
                      "използва като контекст за генерирането на отговор на заявката. По този начин откъсите се "
                      "филтрират и само най-релевантните откъси се използват като контекст за генериране на отговор "
                      "на заявка от LLM.\n\n"
                      "'Брой релевантни откъси' – този плъзгач може да се използва за промяна на броя "
                      "топ релевантни откъси, връщани от процеса на извличане. Диапазонът е от 1 до 7, стойността по "
                      "подразбиране е 3. Предпочитаният диапазон за стойността е от 3 до 5. Променете го въз основа на "
                      "лични предпочитания и опит с приложението.\n\n"
                      "'Език на менюто' - избиране на езика на менюто (български или английски).")

    def add_english_help(self):
        self.add_text("Welcome to BG RAG app!\n"
                      "This is a RAG system that works well with both Bulgarian and English language resources.\n"
                      "It uses the 'INSAIT-Institute/BgGPT-7B-Instruct-v0.2-GGUF' LLM model created by INSAIT.\n"
                      "In order to use the app in its full capacity, you need to install llama.cpp and "
                      "CUDA Toolkit on your computer.\n\n"
                      "General information about the system:\n\n"
                      "The user can copy an answer of the system or a query by clicking on the text in the chat. "
                      "If the resource displayed is a url, it can also be clicked and the link will be opened in the "
                      "default browser of the user. \n\n"
                      "The RAG system is basically a system which utilises an LLM to generate answers to user queries "
                      "but improves the factuality and truthfulness of the answers by giving the LLM context (in the "
                      "form of text chunks) from relevant resources retrieved from a database of text documents.\n"
                      "The most important aspect of the RAG system is the retrieval process where relevant chunks "
                      "to the user query need to be found from a database of text documents. In this system vector and "
                      "keyword search are used for this part and the results are combined with the Reciprocal Rank "
                      "Fusion (RRF) algorithm to determine the final relevant chunks to the query. There can also be a "
                      "variety of different techniques, available in the 'Options' tab.\n\n"
                      "From a number of experiments, the best techniques to be used for better retrieval are RAG "
                      "Fusion and HyDE together. If the user wants to use only one technique in order to reduce the "
                      "time needed for the answer generation, then RAG Fusion is recommended. Generated subqueries "
                      "option can be used for more complex queries depending on the preference of the user (it can "
                      "also be combined with the other techniques). Modifying query based on chat is recommended to be "
                      "on as well. Recommended range of the top n relevant "
                      "chunks is from 3 to 5. Using the LLM for checking of the resources may be helpful for further "
                      "filtering the returned resources by the RAG system and may improve the results. The translation "
                      "of the query should be used for multilingual resources in the user "
                      "uploaded files database or if the user has chosen to use both databases and the uploaded "
                      "(imported) files are in English language (the default database is in Bulgarian and translation "
                      "is needed for the keyword search).\n\n\n"
                      "More detailed instructions and useful info for the app usage and the menu:\n\n\n"
                      "'Import' tab:\n\n"
                      "In this tab the user can import their own files, which can be used as a database for the RAG "
                      "system.\n"
                      "First of all, the user has to choose the files by clicking on 'Choose files'. The possible file "
                      "types are pdf, doc, docx, json, txt. After the selecting of the files, they will be displayed "
                      "in the 'Selected' section. If the user has made a mistake with some of the selected files, they "
                      "can click on 'Clear' in order to remove the selected files and choose new ones.\n"
                      "Second of all, after the user "
                      "is happy with the selected files, 'Import' button needs to be clicked in order for the files to "
                      "be imported in the uploaded files database. If the import is successful, the files will be "
                      "made visible in the 'Imported' section. If there is a problem with some of the files (e.g. they "
                      "cannot be read), they will be ignored and will not be imported.\n"
                      "AFTER THE IMPORT THE USER MUST SELECT THE UPLOADED FILES DATABASE IN THE 'OPTIONS' TAB IN ORDER "
                      "FOR THE IMPORTED DATABASE TO BE USED BY THE RAG SYSTEM.\n"
                      "The imported files database is saved and will be loaded in the next starting of the app.\n"
                      "The user can delete the database created from the imported files by clicking 'Delete database' "
                      "and confirming the deletion.\n"
                      "The imported files can be both in Bulgarian and in English. If there are files from both of the "
                      "languaged, then the user must select the option 'Translate query for multilingual resources' in "
                      "the 'Options' tab.\n\n\n"
                      "'Save' tab:\n\n"
                      "In this tab the user can save the current chat conversation to a file in a chosen directory. "
                      "Firstly, the user must choose a directory where the file is going to be located. "
                      "Secondly, the user has to specify the filename. The spaces in the name are replaced with '_' "
                      "(underscore sign) and the type of the save file is .txt. If the filename written by the user "
                      "is the same name as an existing file in the chosen directory, then the chat conversation will "
                      "be appended to the end of the existing file. In this way, the current chat conversation could "
                      "be appended to a save of a prevoius chat conversation. Moreover, as the system takes account of "
                      "the saved chat iterations, if the user, after a number of new chat iterations, wants to save "
                      "the extended chat again, only the new and unsaved iterations of the chat will be saved. "
                      "This feature is made with the idea that a user may want to extend the current chat save with "
                      "the continuation of the conversation without saving again the already saved previous chat "
                      "conversation. \n"
                      "An option 'Add relevant resources' is available to be turned on if the user wants to save the "
                      "relevant resources returned by the RAG system for each query and answer of the system.\n\n\n"
                      "'Log' tab:\n\n"
                      "This tab is the default tab when the app is starting as it gives information about "
                      "the loading process of the databases, embedding model and LLM. During the loading, the user "
                      "cannot send messages to the system. After the loading is finished, a proper message is "
                      "displayed in the 'Log' tab and the user can use the system normally. In this tab, the user can "
                      "see what happens in the background of the app based on the techniques and options chosen. It "
                      "will also show errors and exceptions if such occurred during the usage of the system. "
                      "The content shown in the tab is also stored as a log file with the name 'rag_log.log' in the "
                      "installation directory of the system. There is an option 'Show relevant chunks' which prints "
                      "in the log the chunks of text that the RAG system has decided as relevant to the user query. "
                      "By default, this option is turned off.\n\n\n"
                      "'Options' tab:\n\n"
                      "This is the tab that enables the user to use different techniques for the retrieval of relevant "
                      "information about the query, as well as set the preferred database or databases.\n\n"
                      "'Advanced RAG' section:\n"
                      "The user can specify which retrieval techniques to use.\n"
                      "RAG Fusion is a technique that utilises the LLM to make 4 similar queries to the original one "
                      "and for each of them (including the original one) the system finds relevant resources (using "
                      "both vector and keyword search). The results are then combined with"
                      " Reciprocal Rank Fusion (RRF) algorithm to retrieve the relevant "
                      "chunks. It generally is the best single technique to be used for improving the retrieval "
                      "process in the app.\n\n"
                      "HyDE is a technique that also utilises the LLM but differently. The LLM answers the user query "
                      "without any context and then it uses the answer to retrieve relevant chunks (resources) for the "
                      "query, using both vector and keyword search. In this system, when HyDE option is on, "
                      "vector and keyword search is also used for retrieval of relevant chunks for the original query. "
                      "The top relevant chunks found for the HyDE LLM answer and the original query are combined with "
                      "the RRF algorithm to retrieve the final top n relevant chunks. This technique can be placed "
                      "at second place as a single technique for improving the retrieval process. For best retrieval "
                      "results, it is recommended to use both HyDE and RAG Fusion even if it costs a little more time "
                      "for the generation of the answer. They can be even combined with the next option - generated "
                      "subqueries technique.\n\n"
                      "The generated subqueries option also utilises the LLM to split a complex query into several "
                      "smaller and self-sufficient queries in order to find the best resources for all parts of the "
                      "query. If the option is turned on, then a check takes place whether the user query is complex "
                      "not (it is simple) and if it is complex, the LLM splits the user query into several simpler "
                      "queries. They are modified to be self-sufficient in order to retain the context so that each "
                      "one of the subqueries can be used in the retrieval process for finding relevant resources. "
                      "The top relevant chunks for each subquery as well as the original one are combined with the "
                      "RRF algorithm to get the final relevant chunks that are provided to the LLM as context to be "
                      "used for answering the user query. It is personal choice whether to use this option or not "
                      "as it may sometimes not provide enough context in the subqueries. Keep an eye of the "
                      "generated subqueries in the 'Log' tab and decide whether to use the option or not.\n\n"
                      "'Database':\n"
                      "The user can choose the default database (created from Bulgarian wikipedia and bulgarian "
                      "news site focus news) or uploaded files database created from imported files of the user. "
                      "The uploaded files database option is only available if the user has imported files to the "
                      "system.\n"
                      "Both the default and the uploaded files database can be used together. If the imported files, "
                      "however, are in English language, then turn the 'Translate query for multilingual resources' "
                      "option when using both databases or when the imported files are both in Bulgarian and "
                      "English language.\n\n"
                      "'Other':\n"
                      "'Display resources' - whether to display the retrieved relevant resources below the answer of "
                      "the query or not. If the option is turned off, then the relevant resources can be seen in the "
                      "'Log' section and the log file. If the option is turned on, the relevant resources found for "
                      "the user query will be displayed below the answer of the query. If the resource is a url from "
                      "the default database, it can be clicked and the webpage of the resource will be loaded and "
                      "displayed in the default web browser of the user.\n\n"
                      "'Use just the LLM (not RAG)' - option to use only the LLM without providing it context of "
                      "relevant chunks to the query from the database of text documents. This options enables users to "
                      "use just the LLM, without the RAG system. When this option is turned "
                      "on, the RAG options are made unavailable as they are not needed. If the option is turned off, "
                      "RAG options are made available again and they are configured the way they were before the "
                      "checking of the option.\n\n"
                      "'Use RAG always' - enables the user to set the option to use RAG on each query. If the option "
                      "is turned off, then the system will decide whether RAG is needed for the query or not. "
                      "It is made to use RAG almost always regardless, however, if the system decides that it has "
                      "enough knowledge it may not use RAG. Most importantly, RAG will not be used if the user input "
                      "is detected as a general comment (like 'Thank you', 'Hi', etc.) and not a query. It is "
                      "recommended for the option to be turned off, although it can be turned on for a query that "
                      "the system has decided that RAG is not needed, but the user wants to use RAG for it.\n\n"
                      "'Modify query based on chat' - if turned on, the user query is modified based on the ongoing "
                      "chat conversation to provide the user query with enough context needed for the retrieval of "
                      "relevant chunks process. If the query is regarded as a self-sufficient and as having enough "
                      "context as it is by the system, then the query is not modified. The option is strongly advised "
                      "to be turned on and only turned off if the user does not like the modification of a certain "
                      "query and wants to send it unchanged to the RAG system. Keep an eye on the modified query in "
                      "the 'Log' tab.\n\n"
                      "'Translate query for multilingual resources' - translates the user query to the other language "
                      "(to English if the query is in Bulgarian and vice versa) for multilingual resources (if the "
                      "imported files are in both languages or the user wants to use both the default and the uploaded "
                      "files database but the latter has documents in English). The translation is also done for each "
                      "generated query in the retrieval process from the techniques chosen (RAG Fusion, HyDE, "
                      "generated subqueries) and the translations could be seen in the 'Log' tab. These translations "
                      "are needed for the keyword search of relevant chunks (it also helps the vector search but only "
                      "a little as the embedding model used in the system is multilingual). Remember that the RAG "
                      "system implemented in the app uses both keyword and vector search for the retrieval of relevant "
                      "chunks and combines the results with the Reciprocal Rank Fusion (RRF) algorithm.\n\n"
                      "'Check relevant resources by LLM' - this option utilises the LLM to filter the retrieved "
                      "chunks by the RAG system. The LLM decides whether the retrieved chunks really contain "
                      "information relevant to the query and if a chunks does not, it is discarded and is not used for "
                      "the answer generation as context. In this way, the chunks are filtered and only the most "
                      "relevant chunks are used as context for the query answer generation.\n\n"
                      "'Top n relevant chunks for context' - this slider can be used to modify the number of top "
                      "relevant chunks returned by the retrieval process. The range is 1 to 7, the default is 3. The "
                      "preferred range for the value is from 3 to 5. Change it based on personal preference and "
                      "experience.\n\n"
                      "Menu Language - select the language of the menu (Bulgarian or English).")
