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
    """
    Class for performing OCR on Ukrainian documents using Tesseract.
    """

    def __init__(self,
                tesseract_path: str = None,
                lang: str = "ukr+eng",
                preprocessor: Preprocessor = None,
                postprocessor: Postprocessor = None):
        """
        Initialize the OCR engine.

        Args:
            tesseract_path: Path to Tesseract executable
            lang: Language code for Tesseract (ukr for Ukrainian)
            preprocessor: Image preprocessor instance
        """
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

        # Custom configuration
        self.custom_config = None

    def set_custom_config(self, config: str):
        """
        Set custom configuration for Tesseract.

        Args:
            config: Tesseract configuration string
        """
        self.custom_config = config

    def _get_tesseract_config(self) -> str:
        """
        Get the Tesseract configuration.

        Returns:
            Tesseract configuration string
        """
        if self.custom_config:
            return self.custom_config

        # Default configuration optimized for Ukrainian/Cyrillic text
        config = f'--psm 3 --oem 3 -l {self.lang}'

        # Page segmentation modes (PSM):
        # 1 = Auto page segmentation with OSD
        # 3 = Fully automatic page segmentation, but no OSD (default)
        # 4 = Assume a single column of text of variable sizes
        # 6 = Assume a single uniform block of text

        # OCR Engine modes (OEM):
        # 1 = Neural nets LSTM engine only - better for complex scripts like Cyrillic
        # 3 = Default, based on what is available (LSTM + Legacy)

        return config

    def recognize_text(self, image: Union[str, np.ndarray]) -> str:
        """
        Perform OCR on an image to extract Ukrainian text.

        Args:
            image: Image path or numpy array

        Returns:
            Extracted text as string
        """
        # Process the image if it's a path
        if isinstance(image, str):
            processed_image = self.preprocessor.process_image(image)
        else:
            # Assume it's already a processed numpy array
            processed_image = image

        # Get Tesseract configuration
        config = self._get_tesseract_config()

        # Convert to PIL Image for better compatibility
        pil_image = Image.fromarray(processed_image)

        # Perform OCR
        text = pytesseract.image_to_string(pil_image, config=config)

        return self.postprocessor.process(text)

    def recognize_to_data(self, image: Union[str, np.ndarray]) -> pd.DataFrame:
        """
        Extract text with position and confidence information.

        Args:
            image: Image path or numpy array

        Returns:
            DataFrame with text, positions, and confidence scores
        """
        # Process the image if it's a path
        if isinstance(image, str):
            processed_image = self.preprocessor.process_image(image)
        else:
            # Assume it's already a processed numpy array
            processed_image = image

        # Get Tesseract configuration
        config = self._get_tesseract_config()

        # Use a PIL image for better compatibility
        pil_image = Image.fromarray(processed_image)

        # Extract detailed OCR data
        data = pytesseract.image_to_data(
            pil_image,
            config=config,
            output_type=pytesseract.Output.DATAFRAME
        )

        # Convert text column to string to handle float values
        data['text'] = data['text'].astype(str)

        # Filter out empty text and non-text entries
        data = data[data['text'].str.strip().astype(bool)]
        
        data["text"] = data["text"].apply(lambda x: self.postprocessor.process(x))

        return data

    def recognize_file(self, file_path: str) -> str:
        """
        Process a file and perform OCR.

        Args:
            file_path: Path to the image file

        Returns:
            Extracted text
        """
        return self.recognize_text(file_path)
