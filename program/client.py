import gradio as gr
import requests

# Define the API endpoint
API_URL = "http://127.0.0.1:8000/extract"  # Replace with your actual API endpoint

def extract_text_from_image(image):
    # Open the image file in binary mode
    with open(image, "rb") as img_file:
        files = {"file": img_file}
        response = requests.post(API_URL, files=files)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json().get("text", "No text found")
    else:
        return f"Error: {response.status_code}, {response.text}"

# Create the Gradio interface
interface = gr.Interface(
    fn=extract_text_from_image,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="OCR Україномовних Документів",
    description="Завантажте зображення документа, щоб витягти текст.",
    theme="default",
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()