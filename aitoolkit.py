from transformers import pipeline, AutoTokenizer
import tkinter as tk
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        self.sentiment_classifier = pipeline('sentiment-analysis')
        self.text_generator_gpt2 = pipeline('text-generation', model='gpt2')
        self.text_generator_gpt_neo = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                     tokenizer=self.tokenizer)

    def run_sentiment_analysis(self, text):
        try:
            result = self.sentiment_classifier(text)
            return result
        except Exception as e:
            raise Exception("Error in run_sentiment_analysis") from e

    def generate_text_based_on_sentiment(self, prompt, sentiment_label, model_name):
        try:
            generator = self.text_generator_gpt2 if model_name == "gpt2" else self.text_generator_gpt_neo
            prompt = f"I am feeling {sentiment_label.lower()} because {prompt}"
            output = generator(prompt, max_length=150, num_return_sequences=1, temperature=0.8)
            generated_text = output[0]['generated_text'] if output else ""
            return generated_text
        except Exception as e:
            raise Exception("Error in generate_text_based_on_sentiment") from e

    def classify_entities(self, text):
        try:
            ner_results = self.ner_pipeline(text)
            threshold = 0.9985464733273534
            entities_expected = ["ORG", "PER", "LOC", "MISC", "O"]
            entities_found = [entity["entity"] for entity in ner_results]
            correct = sum(1 for entity_found, entity_expected in zip(entities_found, entities_expected)
                          if entity_found == entity_expected)
            accuracy = correct / len(entities_expected) if len(entities_expected) > 0 else 0.0
            return ner_results, accuracy
        except Exception as e:
            raise Exception("Error in classify_entities") from e


class View:
    def __init__(self, root):
        self.controller = None

        self.root = root
        self.root.geometry("500x400")
        self.root.title("Text Analysis App")

        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(pady=10)

        self.label = tk.Label(self.input_frame, text="Enter your text:")
        self.label.pack(side=tk.LEFT)

        self.input_entry = tk.Entry(self.input_frame, width=40)
        self.input_entry.pack(side=tk.LEFT)

        self.options_frame = tk.Frame(self.root)
        self.options_frame.pack(pady=10)

        self.selected_option = tk.StringVar(self.root)
        self.selected_option.set("Sentiment Analysis")

        self.option_menu = tk.OptionMenu(self.options_frame, self.selected_option, "Sentiment Analysis", "GPT-2",
                                         "EleutherAI/gpt-neo-1.3B", "NER")
        self.option_menu.pack(side=tk.LEFT)

        self.analyze_button = tk.Button(self.root, text="Analyze")
        self.analyze_button.pack(pady=10)

        self.output_text = tk.Text(self.root, height=10, width=50)
        self.output_text.pack()

    def set_controller(self, controller):
        self.controller = controller
        self.analyze_button.config(command=self.controller.analyze)


    def display_sentiment_results(self, text, sentiment_results):
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"Input: {text}\n\n")
        self.output_text.insert(tk.END, "Sentiment Analysis:\n")
        for result in sentiment_results:
            self.output_text.insert(tk.END, f"{result['label']}: {result['score']}\n")

    def plot_sentiment(self, sentiment_results):
        try:
            labels = [result['label'] for result in sentiment_results]
            scores = [result['score'] for result in sentiment_results]

            plt.bar(labels, scores)
            plt.xlabel('Sentiment')
            plt.ylabel('Score')
            plt.title('Sentiment Analysis')
            plt.show()
        except Exception as e:
            raise Exception("Error in plot_sentiment") from e

    def generate_text(self, prompt, model_name):
        try:
            sentiment_results = self.controller.model.run_sentiment_analysis(prompt)
            sentiment_label = sentiment_results[0]['label'] if sentiment_results else ""
            generated_text = self.controller.model.generate_text_based_on_sentiment(prompt, sentiment_label, model_name)

            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Input: {prompt}\n\n")
            self.output_text.insert(tk.END, f"Sentiment: {sentiment_label}\n")
            self.output_text.insert(tk.END, f"Generated Text: {generated_text}\n\n")
        except Exception as e:
            raise Exception("Error in generate_text") from e

    def classify_entities(self, text):
        try:
            ner_results, accuracy = self.controller.model.classify_entities(text)

            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Input: {text}\n\n")
            self.output_text.insert(tk.END, "NER Results:\n")
            for entity in ner_results:
                self.output_text.insert(tk.END,
                                        f"Word: {entity['word']}\tEntity: {entity['entity']}\tScore: {entity['score']}\n")
            self.output_text.insert(tk.END, f"\nAccuracy: {accuracy}\n")
        except Exception as e:
            raise Exception("Error in classify_entities") from e

    def display_error(self, error_message):
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"Error: {error_message}\n")


class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def analyze(self):
        try:
            text = self.view.input_entry.get().strip()
            if not text:
                raise ValueError("Input text is empty.")

            if self.view.selected_option.get() == "Sentiment Analysis":
                sentiment_results = self.model.run_sentiment_analysis(text)
                self.view.display_sentiment_results(text, sentiment_results)
                self.view.plot_sentiment(sentiment_results)
            elif self.view.selected_option.get() == "GPT-2":
                self.view.generate_text(text, "gpt2")
            elif self.view.selected_option.get() == "EleutherAI/gpt-neo-1.3B":
                self.view.generate_text(text, "EleutherAI/gpt-neo-1.3B")
            elif self.view.selected_option.get() == "NER":
                self.view.classify_entities(text)
            else:
                raise ValueError("Invalid option selected.")
        except Exception as e:
            self.view.display_error(str(e))


class TextAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.model = Model()
        self.view = View(self.root)
        self.controller = Controller(self.model, self.view)
        self.view.set_controller(self.controller)

    def run(self):
        self.root.mainloop()





if __name__ == "__main__":
    app = TextAnalysisApp()
    app.run()
