from fastapi import FastAPI
from Modules.UkrainianOCR import UkrainianOCR
from Modules.Preprocessor import Preprocessor
from Modules.Postprocessor import Postprocessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from fastapi import UploadFile, File
import os

TESSERACT_PATH = "/opt/homebrew/opt/tesseract/bin/tesseract"
MODEL_PATH = "qordon/uk_gec_model_2"
LORA_PATH = "Volplayed/ukr-document-gec"


def load_model():
    """
    Load the tokenizer and model for text correction.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model = PeftModel.from_pretrained(model, LORA_PATH)
    return tokenizer, model

def create_ocr_instance():
    """
    Create an instance of the UkrainianOCR class with preprocessor and postprocessor.
    """
    tokenizer, model = load_model()
    preprocessor = Preprocessor()
    postprocessor = Postprocessor(tokenizer, model)
    ocr_instance = UkrainianOCR(tesseract_path=TESSERACT_PATH,
                                 preprocessor=preprocessor,
                                 postprocessor=postprocessor)
    return ocr_instance

app = FastAPI()
ocr_instance = create_ocr_instance()

@app.get("/")
def read_root():
    """
    Root endpoint to check if the server is running.
    """
    return {"message": "Ukrainian OCR server is running."}

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Endpoint to extract text from an uploaded image file.
    Args:
        file: Uploaded image file
    Returns:
        Extracted text from the image
    """
    # Check file extension
    allowed_extensions = {"png", "jpg", "jpeg"}
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in allowed_extensions:
        return {"error": "Unsupported file type. Only PNG, JPG, and JPEG files are allowed."}

    # Save the uploaded file temporarily
    try:
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Process the image and perform OCR
        extracted_text = ocr_instance.recognize_file(temp_file_path)
        return {"text": extracted_text}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
