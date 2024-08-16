import gradio as gr
from gradio import ChatMessage
from PIL import Image
import base64
from io import BytesIO
import requests
import json

def image_to_base64(image=None):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def mock_vlm(image, text_prompt):
    response_text = f"mock answer"
    return image, response_text

def vlm(text_prompt, image=None):
    if image is not None:
        img_base64 = image_to_base64(image)
        img_url = f"data:image/jpeg;base64,{img_base64}"

        messages = [{
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": text_prompt
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": img_url
                  }
                }
              ]
            }]

    else:
        img_base64 = None
        messages = [{
              "role": "user",
              "content": text_prompt
            }]

    data = {
      "model": "beni1.6b",
      "messages": messages,
      "max_tokens": 384,
      "temperature": 0.7,
      "top_k": 64,
      "do_sample": True
    }

    headers = {"Content-Type":"application/json"}
    res = requests.post("http://localhost:5001/", headers=headers, data=json.dumps(data))
    res = res.json()

    return img_base64, res['choices'][0]['message']['content']


# gradio stuff
theme = gr.themes.Soft(primary_hue="zinc", secondary_hue="green", neutral_hue="green",
                      text_size=gr.themes.sizes.text_sm)   
theme = 'JohnSmith9982/small_and_pretty'
#theme = None

CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""

# Gradio interface components
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# 🦙 Beni vision-language demo")

    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(type="pil", label="Upload Image")
            text_prompt = gr.Textbox(lines=2, placeholder="Enter a text prompt related to the image", label="Text Prompt")
            send_button = gr.Button("Send")
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat with Beni", type="messages") #, elem_id = "chatbot")
            clear_button = gr.Button("Clear chat")

        def clear_messages(chatbot):
            return []

        def add_message(image, text_prompt, chatbot, dims=250):
            img_base64, response_text = vlm(text_prompt, image)
            
            # Convert the image to a base64 string
            if img_base64 is not None:
                img_html = f'<img src="data:image/png;base64,{img_base64}" alt="User Image" style="max-width: {dims}px; max-height: {dims}px;"/>'
                # Append the image and response text to the chatbot
                chatbot.append(
                    ChatMessage(
                        role = "user",
                        content = img_html,
                    ),
                )

            chatbot.append(
                ChatMessage(
                    role = "user",
                    content = text_prompt,
                ),
            )
            chatbot.append(
                ChatMessage(
                    role = "assistant",
                    content = response_text,
                ),
            )

            return chatbot, "", None

        send_button.click(fn=add_message, inputs=[image_input, text_prompt, chatbot], outputs=[chatbot, text_prompt, image_input])
        clear_button.click(fn=clear_messages, inputs=[chatbot], outputs=[chatbot])
        image_input.clear(fn=add_message)

# Launch the Gradio app
demo.launch(debug=True, share=True)

