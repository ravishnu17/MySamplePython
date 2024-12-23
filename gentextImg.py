from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
import google.generativeai as genai
from pydantic import BaseModel
import requests, os, io
from PIL import Image

genai.configure(api_key="AIzaSyCvu0wgSdbOx3THmFt9pxFkJW_L81djbdo")
model = genai.GenerativeModel('gemini-1.5-flash-latest')

app = FastAPI(
    title="Generation API",
    description="Generate Text and image",
)

@app.get("/")
async def root():
    return {"message": "Generate API"}

# Generate madlib phrase using google
prefix_prompt= """
Generate a madlib phrase with a single word missing, which can be a noun, verb, adjective, or adverb. Provide three suggested words separated by commas in the following format:
phrase: {generated_sentence}
options: {suggestions}
Ensure the sentence is meaningful and has no unnecessary spaces.
"""

@app.get("/ai/generatetext")
async def text():
    print("generating Text")
    chat = model.start_chat(
    history=[
        {"role": "user", "parts": "Hello"},
        {"role": "model", "parts": "Great to meet you. What would you like to know?"},
        {"role": "user", "parts": prefix_prompt},
        {"role": "model", "parts": "phrase: The fluffy cat sat on the {adjective} mat.\noptions: grumpy, fuzzy, striped"},
        ]
    )
    response= chat.send_message("Generate New phrase")
    generated_text =  response.text.split("\n")

    options= generated_text[1].split("options: ")[1].strip().split(",")
    generated_text= {"phrase": generated_text[0].split("phrase: ")[1].strip(), "options": [i.strip() for i in options]}

    return {"status": True, "data": generated_text}

headers = {"Authorization": "Bearer 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"}

class Prompt(BaseModel):
    prompt: str

@app.post("/ai/generateimage")
def generateImage(prompt: Prompt):
    print("generating Image")
    response = requests.post("https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large", headers=headers, json={"inputs":prompt + "generate like cartoon style"})
    if response.status_code == 500:
        print(response.content)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error generating image")
    image = Image.open(io.BytesIO(response.content))
    filename= f"{prompt.prompt.replace(' ', '_')[0:10]}.png"
    if filename in os.listdir("images"):
        filename= f"{prompt.prompt.replace(' ', '_')[0:10]}_{len(os.listdir('images'))}.png"
    path= f"images/{filename}"
    image.save(path)
    return {"status": True, "data": filename}

@app.get("/images")
def getImages(filename: str):
    if not os.path.exists(f"images/{filename}"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return FileResponse(f"images/{filename}", media_type="image/png")
