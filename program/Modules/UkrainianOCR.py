import numpy as np
import pandas as pd
import os
import pytesseract
import matplotlib.pyplot as plt
from typing import Tuple, Union, List
from Modules.Preprocessor import Preprocessor
from Modules.Postprocessor import Postprocessor
from PIL import Image

class UkrainianOCR:
    def __init__(self,
                tesseract_path: str = None,
                lang: str = "ukr+eng",
                preprocessor: Preprocessor = None,
                postprocessor: Postprocessor = None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path


        self.lang = lang

        # Check if Ukrainian language data is available
        try:
            langs = pytesseract.get_languages()
            if "ukr" not in langs:
                print(f"Warning: Ukrainian language pack not found in Tesseract. "
                      f"Available languages: {langs}")
                print("Please install the Ukrainian language pack for Tesseract.")

                # Fall back to English if Ukrainian is not available
                if "eng" in langs:
                    self.lang = "eng"
                    print("Falling back to English OCR.")
        except:
            print("Could not verify language availability. Make sure Tesseract is properly installed.")

        # Create a default preprocessor if none provided
        self.preprocessor = preprocessor if preprocessor else Preprocessor()
        self.postprocessor = postprocessor if postprocessor else Postprocessor()

        self.custom_config = None

    def set_custom_config(self, config: str):
        self.custom_config = config

    def _get_tesseract_config(self) -> str:
        if self.custom_config:
            return self.custom_config

        config = f'--psm 3 --oem 3 -l {self.lang}'
        return config

    def recognize_text(self, image: Union[str, np.ndarray]) -> str:
        if isinstance(image, str):
            processed_image = self.preprocessor.process_image(image)
        else:
            processed_image = image

        config = self._get_tesseract_config()

        pil_image = Image.fromarray(processed_image)

        text = pytesseract.image_to_string(pil_image, config=config)

        return self.postprocessor.process(text)

    def recognize_to_data(self, image: Union[str, np.ndarray]) -> pd.DataFrame:
        if isinstance(image, str):
            processed_image = self.preprocessor.process_image(image)
        else:
            processed_image = image

        config = self._get_tesseract_config()

        pil_image = Image.fromarray(processed_image)

        data = pytesseract.image_to_data(
            pil_image,
            config=config,
            output_type=pytesseract.Output.DATAFRAME
        )

        data['text'] = data['text'].astype(str)

        data = data[data['text'].str.strip().astype(bool)]
        
        data["text"] = data["text"].apply(lambda x: self.postprocessor.process(x))

        return data

    def recognize_file(self, file_path: str) -> str:
        return self.recognize_text(file_path)
