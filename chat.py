import json
import os

import customtkinter as ctk
import webbrowser
import validators
from rag import resource_path


class ChatInput(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color='transparent', corner_radius=0)
        self.grid(row=1, column=1, rowspan=1, sticky='nsew')
        self.input_text = ctk.StringVar(value='')
        self.clear_chat_button_clicked = ctk.BooleanVar(value=False)
        self.entry_textbox_font = ctk.CTkFont(family='Helvetica', size=20)
        self.send_button_font_english = ctk.CTkFont(family='Helvetica', size=20, weight="bold")
        self.send_button_font_bulgarian = ctk.CTkFont(family='Helvetica', size=18, weight="bold")

        self.clear_chat_button_text_font = ctk.CTkFont(family='Helvetica', size=16, weight="bold")
        self.entry_textbox = ctk.CTkTextbox(master=self, font=self.entry_textbox_font, wrap='word', state=ctk.NORMAL,
                                            undo=True)
        self.entry_textbox.place(relx=0.05, rely=0.95, relwidth=0.75, relheight=0.9, anchor='sw')
        self.send_button = ctk.CTkButton(master=self, text='Send', font=self.send_button_font_english,
                                         state=ctk.DISABLED, command=self.send_input)
        self.send_button.place(relx=0.84, rely=0.4, relwidth=0.12, relheight=0.3, anchor='sw')
        self.clear_chat_button = ctk.CTkButton(master=self, text='Clear\nchat', fg_color='orange',
                                               hover_color='darkorange', state=ctk.DISABLED,
                                               font=self.clear_chat_button_text_font, command=self.clear_chat)
        self.clear_chat_button.place(relx=0.84, rely=0.9, relwidth=0.12, relheight=0.3, anchor='sw')

        options_config_json = open(resource_path('options_config.json', app_generated_file=True), 'r')
        options_config = json.load(options_config_json)
        options_config_json.close()
        if options_config["menu_language_var"] == "bulgarian":
            self.change_language("bulgarian")

    def send_input(self):
        self.input_text.set(self.entry_textbox.get('0.0', 'end-1c'))
        self.entry_textbox.delete('0.0', ctk.END)
        self.send_button.configure(state=ctk.DISABLED)
        self.clear_chat_button.configure(state=ctk.DISABLED)

    def clear_chat(self):
        self.clear_chat_button_clicked.set(True)
        self.clear_chat_button.configure(state=ctk.DISABLED)

    def change_language(self, new_language):
        if new_language == "bulgarian":
            self.send_button.configure(text="Изпрати", font=self.send_button_font_bulgarian)
            self.clear_chat_button.configure(text="Изчисти\nчата")
        if new_language == "english":
            self.send_button.configure(text="Send", font=self.send_button_font_english)
            self.clear_chat_button.configure(text="Clear\nchat")


class ChatOutput(ctk.CTkScrollableFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color='transparent', corner_radius=0)
        self.grid(row=0, column=1, rowspan=1, sticky='nsew')
        self.text_font = ctk.CTkFont(family='Helvetica', size=18)
        self.rag_response = ctk.StringVar()
        self.time_to_stop_waiting_animation = False
        self.dots_str = ctk.StringVar(value='...')
        self.waiting_rag_label = ctk.CTkLabel(master=self, padx=10, pady=10, fg_color=('#60a0a6', '#4d4545'),
                                              font=ctk.CTkFont(family='Helvetica', size=20, weight='bold'),
                                              textvariable=self.dots_str,
                                              width=70, corner_radius=8, justify='center', wraplength=100)

        options_config_json = open(resource_path('options_config.json', app_generated_file=True), 'r')
        options_config = json.load(options_config_json)
        options_config_json.close()
        self.menu_language = options_config["menu_language_var"]

    def add_text_label(self, text, is_user, relevant_resources=None):
        fg_color_label = ('#3B8ED0', '#1F6AA5') if is_user else ('#60a0a6', '#4d4545')
        label = ctk.CTkLabel(master=self, padx=10, pady=10, text=text, fg_color=fg_color_label, corner_radius=8,
                             font=self.text_font, justify='left', wraplength=500, cursor="hand2")
        label.bind("<Button-1>", lambda e: self.copy_label_text(text))

        if is_user:
            self.dots_str.set('...')
            label.pack(side=ctk.TOP, anchor='e', padx=20, pady=10, expand=True)
            self.waiting_rag_label.pack(side=ctk.TOP, anchor='w', padx=20, pady=10, expand=True)
            self.time_to_stop_waiting_animation = False
            self._parent_canvas.yview_moveto(1.0)
            self.animate_waiting_label()
        else:
            label.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)
            if relevant_resources is not None and len(relevant_resources) > 0:
                info_label = ctk.CTkLabel(master=self, padx=10, pady=10, text="Resources:", fg_color=fg_color_label,
                                          corner_radius=8, font=self.text_font, justify='left', wraplength=500)
                if self.menu_language == "bulgarian":
                    info_label.configure(text="Източници:")

                info_label.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)

                if validators.url(relevant_resources[0]):
                    resource_label_1 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[0],
                                                    fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                    justify='left', wraplength=500, cursor='hand2')
                    resource_label_1.bind("<Button-1>", lambda e: webbrowser.open(relevant_resources[0]))
                    resource_label_1.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)
                else:
                    resource_label_1 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[0],
                                                    fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                    justify='left', wraplength=500, cursor='hand2')
                    resource_label_1.bind("<Button-1>", lambda e: os.startfile(relevant_resources[0]))
                    resource_label_1.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)

                if len(relevant_resources) >= 2:
                    if validators.url(relevant_resources[1]):
                        resource_label_2 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[1],
                                                        fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                        justify='left', wraplength=500, cursor='hand2')
                        resource_label_2.bind("<Button-1>", lambda e: webbrowser.open(relevant_resources[1]))
                        resource_label_2.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)
                    else:
                        resource_label_2 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[1],
                                                        fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                        justify='left', wraplength=500, cursor='hand2')
                        resource_label_2.bind("<Button-1>", lambda e: os.startfile(relevant_resources[1]))
                        resource_label_2.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)

                if len(relevant_resources) >= 3:
                    if validators.url(relevant_resources[2]):
                        resource_label_3 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[2],
                                                        fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                        justify='left', wraplength=500, cursor='hand2')
                        resource_label_3.bind("<Button-1>", lambda e: webbrowser.open(relevant_resources[2]))
                        resource_label_3.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)
                    else:
                        resource_label_3 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[2],
                                                        fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                        justify='left', wraplength=500, cursor='hand2')
                        resource_label_3.bind("<Button-1>", lambda e: os.startfile(relevant_resources[2]))
                        resource_label_3.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)

                if len(relevant_resources) >= 4:
                    if validators.url(relevant_resources[3]):
                        resource_label_4 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[3],
                                                        fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                        justify='left', wraplength=500, cursor='hand2')
                        resource_label_4.bind("<Button-1>", lambda e: webbrowser.open(relevant_resources[3]))
                        resource_label_4.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)
                    else:
                        resource_label_4 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[3],
                                                        fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                        justify='left', wraplength=500, cursor='hand2')
                        resource_label_4.bind("<Button-1>", lambda e: os.startfile(relevant_resources[3]))
                        resource_label_4.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)

                if len(relevant_resources) >= 5:
                    if validators.url(relevant_resources[4]):
                        resource_label_5 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[4],
                                                        fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                        justify='left', wraplength=500, cursor='hand2')
                        resource_label_5.bind("<Button-1>", lambda e: webbrowser.open(relevant_resources[4]))
                        resource_label_5.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)
                    else:
                        resource_label_5 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[4],
                                                        fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                        justify='left', wraplength=500, cursor='hand2')
                        resource_label_5.bind("<Button-1>", lambda e: os.startfile(relevant_resources[4]))
                        resource_label_5.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)

                if len(relevant_resources) >= 6:
                    if validators.url(relevant_resources[5]):
                        resource_label_6 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[5],
                                                        fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                        justify='left', wraplength=500, cursor='hand2')
                        resource_label_6.bind("<Button-1>", lambda e: webbrowser.open(relevant_resources[5]))
                        resource_label_6.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)
                    else:
                        resource_label_6 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[5],
                                                        fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                        justify='left', wraplength=500, cursor='hand2')
                        resource_label_6.bind("<Button-1>", lambda e: os.startfile(relevant_resources[5]))
                        resource_label_6.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)

                if len(relevant_resources) >= 7:
                    if validators.url(relevant_resources[6]):
                        resource_label_7 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[6],
                                                        fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                        justify='left', wraplength=500, cursor='hand2')
                        resource_label_7.bind("<Button-1>", lambda e: webbrowser.open(relevant_resources[6]))
                        resource_label_7.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)
                    else:
                        resource_label_7 = ctk.CTkLabel(master=self, padx=10, pady=10, text=relevant_resources[6],
                                                        fg_color=fg_color_label, corner_radius=8, font=self.text_font,
                                                        justify='left', wraplength=500, cursor='hand2')
                        resource_label_7.bind("<Button-1>", lambda e: os.startfile(relevant_resources[6]))
                        resource_label_7.pack(side=ctk.TOP, anchor='w', padx=20, pady=1, expand=True)
        self._parent_canvas.yview_moveto(1.0)

    def animate_waiting_label(self, i=1):
        if i == 2:
            self._parent_canvas.yview_moveto(1.0)
        if self.time_to_stop_waiting_animation:
            self._parent_canvas.yview_moveto(1.0)
            return

        if len(self.dots_str.get()) == 3:
            self.dots_str.set('.')
        else:
            self.dots_str.set(self.dots_str.get() + '.')
        i += 1
        self.after(1000, lambda: self.animate_waiting_label(i))

    def rag_response_text_callback(self, rag_response, relevant_resources, *args):
        self.waiting_rag_label.pack_forget()
        self.time_to_stop_waiting_animation = True
        self.add_text_label(text=rag_response, is_user=False, relevant_resources=relevant_resources)
        self._parent_canvas.yview_moveto(1.0)

    def update_waiting_rag_label(self, dots_str):
        self.waiting_rag_label.configure(text=dots_str)

    def copy_label_text(self, text):
        self.clipboard_clear()
        self.clipboard_append(text, format='unicode')
