import language_tool_python
import torch

class Postprocessor:
    def __init__(self, tokenizer, model):
        """
        Initialize the PostProcessor with a tokenizer and model.
        Args:
            tokenizer: Tokenizer for the model
            model: Model for text correction
        """
        # Initialize the LanguageTool client for Ukrainian
        self.tool = language_tool_python.LanguageTool('uk-UA')

        #load model and tokenizer
        self.tokenizer = tokenizer
        self.model = model

    def clean_formatting(self, text):
        # Basic formatting fixes, like extra spaces
        text = text.replace(" ,", ",").replace(" .", ".")
        text = text.replace("  ", " ").strip()
        return text


    def correct_spelling_and_grammar(self, text):
        # Find mistakes using LanguageTool
        matches = self.tool.check(text)
        # Apply corrections to the text
        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text

    def correct_text(self, text, correction_type="pl"):
      divider = "------"
      if correction_type == "pl":
          divider = "\n"
      elif correction_type == "pp":
          divider = "\n\n"

      corrected = []

      for line in text.split(divider):
          if line:
              # Tokenize the input text
              inputs = self.tokenizer(line, return_tensors="pt", truncation=True, padding=True)

              # Run the model to get the corrected text
              with torch.no_grad():
                  outputs = self.model.generate(**inputs)

              # Decode the output tokens to string
              corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

              corrected.append(corrected_text)
          else:
            corrected.append('')
      return divider.join(corrected)



    def process(self, text):
        text = self.clean_formatting(text)
        text = self.correct_text(text)
        text = self.correct_spelling_and_grammar(text)
        return text
