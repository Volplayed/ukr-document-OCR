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

submit_btn = gr.Button(
    "Витягнути текст",
    variant="primary",
)

clear_btn = gr.Button(
    "Очистити",
    variant="secondary",
    elem_id="clear_btn",
)

image_input = gr.Image(
    type="filepath",
    label="Зображення документа",
    webcam_options=None,
    show_download_button=False,
    show_share_button=False,
    streaming=False,
    placeholder="Завантажте зображення документа",
    sources=["upload"],
)

text_output = gr.Textbox(
    label="Результат тексту",
    placeholder="Тут з'явиться витягнутий текст",
    lines=10,
    interactive=False,
    visible=True,
)

interface = gr.Interface(
    fn=extract_text_from_image,
    inputs=image_input,
    outputs=gr.Textbox(label="Результат тексту"),
    title="OCR Україномовних Документів",
    description="Завантажте зображення документа, щоб витягти текст.",
    allow_flagging="never",
    examples=None,
    submit_btn=submit_btn,
    clear_btn=clear_btn,
    theme="default",
    live=False,
    analytics_enabled=False,
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()