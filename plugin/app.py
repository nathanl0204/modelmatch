import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import requests
import json
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from classifier_pipeline import ClassifierPipeline

class ModelMatchApp:
    """
    Classe principale pour l'application ModelMatch.
    Fournit une interface utilisateur pour interagir avec des modèles de langage
    et détecter en temps réel un changement de modèle par rapport au modèle
    attendu, ou analyser un texte fourni, sans identifier le modèle de remplacement.
    """
    def __init__(self, root):
        """Initialise l'application, la fenêtre principale et les composants de base."""
        self.root = root
        self.root.title("ModelMatch Plugin")
        self.root.geometry("800x600")

        self.conversation_history = []
        self.pipeline = ClassifierPipeline()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.chat_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.chat_tab, text="Chat en direct")
        self.notebook.add(self.analysis_tab, text="Analyse de texte")

        self.setup_chat_tab()

        self.setup_analysis_tab()
    
    def setup_chat_tab(self):
        """Configure l'interface de l'onglet 'Chat en direct'."""
        config_frame = ttk.LabelFrame(self.chat_tab, text="Configuration", padding="10")
        config_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(config_frame, text="Modèle attendu:").grid(row=0, column=0, sticky="w")
        self.supported_models = [
            "Qwen/Qwen2-7B-Instruct", "RedHatAI/Qwen2-7B-Instruct-quantized.w8a16",
            "elinas/Llama-3-13B-Instruct", "google/gemma-7b-it",
            "meta-llama/Meta-Llama-3-8B-Instruct", "unsloth/gemma-7b-it-bnb-4bit",
            "deepseek-ai/deepseek-llm-7b-chat", "gpt-4o",
            "microsoft/Phi-3-mini-128k-instruct", "mistralai/Mistral-7B-Instruct-v0.1"
        ]
        self.model_var = tk.StringVar(value=self.supported_models[0])
        self.model_menu = ttk.Combobox(config_frame, textvariable=self.model_var, values=self.supported_models, state="readonly")
        self.model_menu.grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(config_frame, text="API Endpoint:").grid(row=1, column=0, sticky="w")
        self.api_url_var = tk.StringVar(value="http://localhost:8000/v1/chat/completions")
        self.api_url_entry = ttk.Entry(config_frame, textvariable=self.api_url_var)
        self.api_url_entry.grid(row=1, column=1, sticky="ew", padx=5)

        ttk.Label(config_frame, text="API Key (Optionnel):").grid(row=2, column=0, sticky="w")
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(config_frame, textvariable=self.api_key_var, show="*")
        self.api_key_entry.grid(row=2, column=1, sticky="ew", padx=5)

        config_frame.columnconfigure(1, weight=1)

        chat_frame = ttk.LabelFrame(self.chat_tab, text="Chat", padding="10")
        chat_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.chat_display = scrolledtext.ScrolledText(chat_frame, state="disabled", wrap=tk.WORD)
        self.chat_display.pack(fill="both", expand=True)
        self.chat_display.tag_configure("bold", font=("Segoe UI", 9, "bold"))

        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill="x", pady=5)

        self.prompt_entry = ttk.Entry(input_frame)
        self.prompt_entry.pack(fill="x", expand=True, side="left", padx=(0, 5))
        self.prompt_entry.bind("<Return>", self.send_prompt)

        self.send_button = ttk.Button(input_frame, text="Envoyer", command=self.send_prompt)
        self.send_button.pack(side="right")

        verify_frame = ttk.Frame(self.chat_tab, padding="10")
        verify_frame.pack(fill="x")

        self.verify_button = ttk.Button(verify_frame, text="Détecter un changement", command=self.run_verification)
        self.verify_button.pack(side="left")

        self.visualize_button = ttk.Button(verify_frame, text="Visualiser les métriques", command=self.open_visualization_window)
        self.visualize_button.pack(side="left", padx=5)

        self.reset_button = ttk.Button(verify_frame, text="Réinitialiser", command=self.reset_chat)
        self.reset_button.pack(side="left", padx=5)

        self.result_label = ttk.Label(verify_frame, text="Résultat: En attente de détection...", font=("Segoe UI", 10, "bold"))
        self.result_label.pack(side="left", padx=10)
    
    def setup_analysis_tab(self):
        """Configure l'interface de l'onglet 'Analyse de texte'."""
        prompt_frame = ttk.LabelFrame(self.analysis_tab, text="Prompt utilisateur (contexte)", padding="10")
        prompt_frame.pack(fill="x", padx=10, pady=5)
        self.analysis_prompt_text = scrolledtext.ScrolledText(prompt_frame, height=5, wrap=tk.WORD)
        self.analysis_prompt_text.pack(fill="x", expand=True)

        response_frame = ttk.LabelFrame(self.analysis_tab, text="Réponse du LLM à analyser", padding="10")
        response_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.analysis_response_text = scrolledtext.ScrolledText(response_frame, height=15, wrap=tk.WORD)
        self.analysis_response_text.pack(fill="both", expand=True)

        analysis_verify_frame = ttk.Frame(self.analysis_tab, padding="10")
        analysis_verify_frame.pack(fill="x")

        self.analysis_verify_button = ttk.Button(analysis_verify_frame, text="Analyser le texte", command=self.run_text_analysis_verification)
        self.analysis_verify_button.pack(side="left")

        self.analysis_result_label = ttk.Label(analysis_verify_frame, text="Résultat: En attente d'analyse...", font=("Segoe UI", 10, "bold"))
        self.analysis_result_label.pack(side="left", padx=10)
    
    def open_visualization_window(self):
        """Ouvre une nouvelle fenêtre pour visualiser l'évolution des métriques."""
        if not self.conversation_history:
            messagebox.showinfo("Info", "Aucune conversation à analyser. Veuillez d'abord discuter avec le modèle.")
            return
        
        vis_window = tk.Toplevel(self.root)
        vis_window.title("Visualisation des métriques")
        vis_window.geometry("800x600")

        control_frame = ttk.Frame(vis_window, padding="10")
        control_frame.pack(fill="x")

        ttk.Label(control_frame, text="Chosiir une métrique:").pack(side="left")

        metric_names = self.pipeline.get_metric_names()
        self.metric_to_plot = tk.StringVar()

        # Barre de recherche avec autocomplétion
        metric_combo = ttk.Combobox(control_frame, textvariable=self.metric_to_plot, values=metric_names)
        metric_combo.pack(side="left", fill="x", expand=True, padx=5)
        metric_combo.bind("<<ComboboxSelected>>", lambda event: self.update_plot(vis_window))

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=vis_window)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        vis_window.ax = ax
        vis_window.canvas = canvas
        vis_window.metric_var = self.metric_to_plot

        self.update_plot(vis_window) # Affiche le graphique initial
    
    def update_plot(self, window):
        """Met à jour le graphique avec la métrique sélectionnée."""
        metric_name = window.metric_var.get()
        if not metric_name or not self.conversation_history:
            return
        
        fingerprints_df = self.pipeline._create_fingerprints(self.conversation_history)

        if metric_name not in fingerprints_df.columns:
            messagebox.showerror("Erreur", f"La métrique '{metric_name}' n'a pas pu être calculée.")
            return
        
        values = fingerprints_df[metric_name]
        turns = range(1, len(values) + 1)

        window.ax.clear()
        window.ax.plot(turns, values, marker='o', linestyle='-')
        window.ax.set_title(f"Évolution de: {metric_name}")
        window.ax.set_xlabel("Tour de conversation")
        window.ax.set_ylabel("Valeur de la métrique")
        window.ax.grid(True)
        window.canvas.draw()
    
    def add_text_to_chat(self, sender, text):
        """Ajoute du texte à la fenêtre de chat."""
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, f"{sender}:", ("bold",))
        self.chat_display.insert(tk.END, f" {text}\n\n")
        self.chat_display.config(state="disabled")
        self.chat_display.see(tk.END)
    
    def send_prompt(self, event=None):
        """Gère l'envoi d'un prompt par l'utilisateur."""
        prompt = self.prompt_entry.get()
        if not prompt:
            return
        
        self.add_text_to_chat("Vous", prompt)
        self.prompt_entry.delete(0, tk.END)

        self.send_button.config(state="disabled")
        threading.Thread(target=self.get_model_response, args=(prompt,)).start()
    
    def get_model_response(self, prompt):
        """Envoie le prompt à l'API du modèle et affiche la réponse."""
        api_url = self.api_url_var.get()
        model_name = self.model_var.get()
        api_key = self.api_key_var.get()

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        messages = []
        for turn in self.conversation_history:
            messages.append({"role": "user", "content": turn["prompt"]})
            messages.append({"role": "assistant", "content": turn["response"]})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": model_name,
            "messages": messages,
            "max_tokens":512,
            "temperature": 0.7
        }

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=60)
            response.raise_for_status()
            response_json = response.json()
            model_response = response_json["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            model_response = f"Erreur de connexion à l'API: {e}"
        except (KeyError, IndexError):
            model_response = f"Réponse inattendue de l'API: {response.text}"

        self.conversation_history.append({"prompt": prompt, "response": model_response})
        self.root.after(0, self.add_text_to_chat, "Modèle", model_response)
        self.root.after(0, self.send_button.config, {"state": "normal"})
    
    def run_verification(self):
        """Lance la détection de changement pour la conversation en cours."""
        if not self.conversation_history:
            messagebox.showinfo("Info", "Veuillez d'abord converser avec le modèle avant de lancer une détection.")
            return
        
        self.verify_button.config(state="disabled")
        self.result_label.config(text="Résultat: Détection en cours...")
        threading.Thread(target=self.verification_thread, args=(self.conversation_history, self.verify_button, self.result_label)).start()
    
    def reset_chat(self):
        """Réinitialise l'historique de la conversation et l'affichage."""
        if messagebox.askyesno("Réinitialiser la conversation", "Êtes-vous sûr de vouloir effacer la conversation actuelle ?"):
            self.conversation_history.clear()
            self.chat_display.config(state="normal")
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state="disabled")
            self.result_label.config(text="Résultat: En attente de détection...")
    
    def run_text_analysis_verification(self):
        """Lance la vérification pour le texte collé dans l'onglet d'analyse."""
        response_text = self.analysis_response_text.get("1.0", tk.END).strip()
        prompt_text = self.analysis_prompt_text.get("1.0", tk.END).strip()

        if not response_text:
            messagebox.showinfo("Info", "Veuillez coller une réponse de LLM à analyser.")
            return
        
        if not prompt_text:
            messagebox.showwarning("Avertissement", "Aucun prompt n'a été fourni. L'analyse sera moins précise car les métriques d'interaction ne seront pas calculées.")

        temp_history = [{"prompt": prompt_text, "response": response_text}]

        self.analysis_verify_button.config(state="disabled")
        self.analysis_result_label.config(text="Résultat: Analyse en cours...")
        threading.Thread(target=self.verification_thread, args=(temp_history, self.analysis_verify_button, self.analysis_result_label)).start()
    
    def verification_thread(self, history, button, label):
        """
        Exécute la pipeline de classification dans un thread séparé
        pour ne pas bloquer l'interface utilisateur.

        Le résultat est exprimé en détection d'anomalie : un écart entre le
        modèle classé et le modèle attendu signale un changement, sans
        révéler l'identité du modèle de remplacement.
        """
        expected_model = self.model_var.get()
        predicted_model, _ = self.pipeline.verify(history)

        if predicted_model == expected_model:
            result_text = "Résultat: Aucun changement détecté (cohérent avec le modèle attendu)."
        else:
            result_text = "Résultat: CHANGEMENT DE MODÈLE DÉTECTÉ ! Le modèle en cours ne correspond plus à celui attendu."

        self.root.after(0, button.config, {"state": "normal"})
        self.root.after(0, label.config, {"text": result_text})


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelMatchApp(root)
    root.mainloop()