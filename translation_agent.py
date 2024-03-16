import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


class TranslationAgent:
    def __init__(self):
        self.model_name = "facebook/mbart-large-50-many-to-many-mmt"
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.model_name)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def translate(self, phrase, src_lang, tgt_lang):
        # Ensure the model is in evaluation mode
        self.model.eval()
        with torch.no_grad():
            # Tokenize the source phrase
            inputs = self.tokenizer(
                phrase, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            # Add the source language ID at the beginning of the input IDs
            src_lang_id = torch.tensor(
                [[self.tokenizer.lang_code_to_id[src_lang]]], device=self.device)
            inputs['input_ids'] = torch.cat(
                [src_lang_id, inputs['input_ids']], dim=-1)
            # Update the attention mask
            inputs['attention_mask'] = torch.cat(
                [torch.ones(1, 1, device=self.device), inputs['attention_mask']], dim=-1)
            # Generate translation
            translated = self.model.generate(**inputs,
                                             forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
                                             decoder_start_token_id=None)
            # Decode the translation
            translation = self.tokenizer.batch_decode(
                translated, skip_special_tokens=True)
        return translation[0]


class TranslationConsole:
    def __init__(self):
        self.agent = TranslationAgent()
        self.languages = {
            '1': 'fr_XX',  # French
            '3': 'de_DE',  # German
            '4': 'it_IT',  # Italian
        }
        self.language_names = {
            'fr_XX': 'French',
            'de_DE': 'German',
            'it_IT': 'Italian',
        }

    def display_languages(self):
        for key, value in self.languages.items():
            print(f"{key}: {self.language_names[value]}")

    def start(self):
        print("Welcome to the translation console!")
        while True:
            print("\nPlease enter a phrase to translate (or 'exit' to quit):")
            phrase = input()
            if phrase.lower() == 'exit':
                break

            print("\nPlease choose a target language:")
            self.display_languages()
            tgt_lang = input()

            if tgt_lang not in self.languages:
                print("Invalid language selection.")
                continue

            translation = self.agent.translate(
                phrase, 'en_XX', self.languages[tgt_lang])
            print(f"\nTranslated phrase: {translation}")


if __name__ == "__main__":
    console = TranslationConsole()
    console.start()
