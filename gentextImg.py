from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
import google.generativeai as genai
from pydantic import BaseModel
import requests, os, io, json
from PIL import Image
from openai import OpenAI

genai.configure(api_key="AIzaSyCvu0wgSdbOx3THmFt9pxFkJW_L81djbdo")
model = genai.GenerativeModel('gemini-1.5-flash-latest')

grok_key="xai-1cxIrCKdRACgyIutAuwrz6Aqei10eYCR09UoiNLhPklGA2PmvzXqWCwtA8fMSITLZlrgbBJ82J1vPihG"

app = FastAPI(
    title="Generation API",
    description="Generate Text and image",
)

@app.get("/")
async def root():
    return {"message": "Generate API"}

prefix_prompt= """
Generate a madlib phrase with a single word missing, which can be a noun, verb, adjective, or adverb. Provide three suggested words separated by commas in the following format:
Example: {"phrase": "The {noun} jumped over the fence.", "options": "dog, cat, rabbit"}
Ensure the sentence is meaningful and has no unnecessary spaces.
"""

# Generate madlib phrase using google
@app.get("/ai/generatetext")
async def text():
    print("generating Text")
    chat = model.start_chat(
    history=[
            {"role": "user", "parts": "Hello"},
            {"role": "model", "parts": "Great to meet you. What would you like to know?"},
            {"role": "user", "parts": prefix_prompt},
            {"role": "model", "parts": '{"phrase": "The fluffy cat sat on the {adjective} mat.","options": "grumpy, fuzzy, striped"}'},
        ]
    )
    response= chat.send_message("Generate New phrase")
    try:
        generated_text= json.loads(response.text)
        generated_text['options'] = [i.strip() for i in generated_text['options'].split(",")] if type(generated_text['options']) == str else [i.strip() for i in generated_text['options']]
    except:
        print(response.text)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error generating text")

    # generated_text =  response.text.split("\n")

    # options= generated_text[1].split("options: ")[1].strip().split(",")
    # generated_text= {"phrase": generated_text[0].split("phrase: ")[1].strip(), "options": [i.strip() for i in options]}

    return {"status": True, "data": generated_text}

promptx="""
Create a sentence with one missing word, indicated by its part of speech in curly braces (e.g., {noun}, {verb}, {adjective}). 
Provide three options to fill the blank. Ensure the sentence is simple and suitable for kids. 
Example json : {"phrase": "The {adjective} cat climbed the tree.", "options": "fluffy, small, happy"}

Generate a unique sentence each time.
"""

# Generate madlib phrase using grok
@app.get('/xtext')
def xtext():
    client = OpenAI( api_key= grok_key, base_url="https://api.x.ai/v1")
    
    response = client.chat.completions.create(
    model="grok-2-latest",
    messages=[
        {"role": "system", "content": "How can I help you."},
        {"role": "user", "content": promptx},
    ],
    response_format={ "type": "json_object" },
    temperature=0.7,
    max_tokens=100,
    )
    try:
        generated_text= json.loads(response.choices[0].message.content)
        generated_text['options'] = [i.strip() for i in generated_text['options'].split(",")]
    except:
        print(response.choices)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate text")

    return {"status": True, "data": generated_text}


class Prompt(BaseModel):
    prompt: str

# url= "https://m4rh4wg8-8000.inc1.devtunnels.ms/ai/images/"
url= "http://localhost:8000/ai/images/"


def gen_file_name(prompt):
    filename= f"{prompt.replace(' ', '_')[0:10]}.png"
    if filename in os.listdir("images"):
        filename= f"{prompt.replace(' ', '_')[0:10]}_{len(os.listdir('images'))}.png"
    return filename

# Generate image using huggingface inference request - 3 per minute
@app.post("/ai/generateimage")
def generateImage(prompt: Prompt):
    print("generating Image using huggingface")
    headers = {"Authorization": "Bearer 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"}
    response = requests.post(
        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large", 
        headers=headers, 
        json={"inputs":prompt.prompt + "generate like cartoon style"}
    )
    if response.status_code == 500:
        print(response.content)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error generating image")
    image = Image.open(io.BytesIO(response.content))
    filename= gen_file_name(prompt.prompt)
    path= f"images/{filename}"
    image.save(path)
    return {"status": True, "image": f"{url}{filename}" }

# Generate image using stability.ai - 3 credits per request 1 credit = 0.01$ - free 25 credits
@app.post('/ai/generateimage/stabilityai')
def generateImage(prompt: Prompt):
    print("generating image using stability.ai")
    response = requests.post(
        "https://api.stability.ai/v2beta/stable-image/generate/core",
        headers={
            "authorization": f"Bearer sk-ES1hkdefH8ZBIRdcNuAro7cJrc8M0CZG7RzeL3acxLvC0otT",
            "accept": "image/*"
        },
        files={"none": ''},
        data={
            "prompt": prompt.prompt,
            "output_format": "png",
        },
    )
    filename= gen_file_name(prompt.prompt)
    if response.status_code == 200:
        with open(f"images/{filename}", 'wb') as file:
            file.write(response.content)
    else:
        print(str(response.json()))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate image")

    return {"status": True, "image":  f"{url}{filename}"}

@app.post

# Image service
@app.get("/ai/images/{filename}")
def getImages(filename: str):
    if not os.path.exists(f"images/{filename}"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return FileResponse(f"images/{filename}", media_type="image/png")
