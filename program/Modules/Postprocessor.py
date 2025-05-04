import language_tool_python
import torch

class Postprocessor:
    def __init__(self, tokenizer, model):
        self.tool = language_tool_python.LanguageTool('uk-UA')

        self.tokenizer = tokenizer
        self.model = model

    def clean_formatting(self, text):
        text = text.replace(" ,", ",").replace(" .", ".")
        text = text.replace("  ", " ").strip()
        return text


    def correct_spelling_and_grammar(self, text):
        matches = self.tool.check(text)
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
              inputs = self.tokenizer(line, return_tensors="pt", truncation=True, padding=True)

              with torch.no_grad():
                  outputs = self.model.generate(**inputs)

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
