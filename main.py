import os
import re
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# third-party SDKs
import google.generativeai as genai
import openai

load_dotenv()  # read .env

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not GEMINI_KEY or not OPENAI_KEY:
    raise RuntimeError("Set GEMINI_API_KEY and OPENAI_API_KEY in backend/.env")

genai.configure(api_key=GEMINI_KEY)
openai.api_key = OPENAI_KEY

app = FastAPI(title="Story Generator Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in prod restrict to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoryRequest(BaseModel):
    idea: str
    genre: str = "Fantasy"
    tone: str = "Lighthearted"
    audience: str = "Teens"

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate_story")
async def generate_story(req: StoryRequest):
    """
    Returns JSON: { "scenes": ["scene 1 text", "scene 2 text", ...] }
    Uses Gemini to generate a 4-scene story and tries to return valid JSON.
    """
    # Instruct the model to output JSON with key "scenes"
    system_prompt = (
        "You are a creative short-story generator. "
        "Given the idea and constraints, produce exactly a JSON object with key 'scenes' "
        "whose value is an array of 3-5 short scene strings. Output ONLY valid JSON.\n\n"
    )
    user_prompt = (
        f"Idea: {req.idea}\n"
        f"Genre: {req.genre}\n"
        f"Tone: {req.tone}\n"
        f"Audience: {req.audience}\n\n"
        "Format: {\"scenes\": [\"scene1...\",\"scene2...\", ...]}"
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(system_prompt + user_prompt)
        text = resp.text.strip()

        # Try parse JSON first
        try:
            parsed = json.loads(text)
            scenes = parsed.get("scenes", [])
            if not isinstance(scenes, list) or len(scenes) == 0:
                raise ValueError("No scenes in JSON")
        except Exception:
            # fallback: split by blank lines or "Scene" markers
            parts = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
            if len(parts) >= 1:
                scenes = parts
            else:
                scenes = [text]

        # Normalize scenes to max 5
        scenes = scenes[:5]
        return {"scenes": scenes}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Story generation failed: {str(e)}")

@app.post("/generate_image")
async def generate_image(req: ImageRequest):
    """
    Returns { "image_url": "..." } by calling OpenAI image API.
    Uses the scene text as the prompt; you can refine this later to generate
    a dedicated image prompt first.
    """
    prompt_text = req.prompt
    # add a style hint so images are consistent
    image_prompt = f"{prompt_text}. Illustrative, cinematic composition, soft lighting, digital art, high detail."

    try:
        # Using OpenAI images generate (if your SDK version differs, adapt accordingly)
        result = openai.images.generate(
            model="gpt-image-1",
            prompt=image_prompt,
            size="1024x1024"
        )
        url = result.data[0].url
        return {"image_url": url}
    except Exception as e:
        # Provide a fallback placeholder image (so frontend still shows)
        placeholder = "https://via.placeholder.com/1024x768?text=Image+Unavailable"
        return {"image_url": placeholder}
