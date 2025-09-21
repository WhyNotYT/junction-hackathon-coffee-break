from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import ollama  # Make sure ollama Python package is installed

app = FastAPI(title="My Ollama API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    model: str = "gemma3:4b"

class ChatResponse(BaseModel):
    response: str
    model: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "https://nordxgpt.vercel.app",
        "https://qbhhpr-ip-130-231-176-211.tunnelmole.net",
        "*"  # For testing; remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Ollama FastAPI server is running!"}

@app.get("/models")
async def list_models():
    """Return only gemma3:4b to avoid KeyErrors"""
    try:
        # Optionally verify that gemma3:4b exists
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        available_models = [line.split()[0] for line in result.stdout.splitlines()[1:]]
        if "gemma3:4b" in available_models:
            return {"models": ["gemma3:4b"]}
        else:
            return {"models": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ollama(request: ChatRequest):
    """Chat with Ollama gemma3:4b"""
    try:
        response = ollama.chat(
            model="gemma3:4b",
            messages=[{"role": "user", "content": request.message}]
        )
        return ChatResponse(
            response=response['message']['content'],
            model="gemma3:4b"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

@app.post("/generate")
async def generate_text(request: ChatRequest):
    """Generate text using Ollama gemma3:4b"""
    try:
        response = ollama.generate(model="gemma3:4b", prompt=request.message)
        return {"response": response['response'], "model": "gemma3:4b"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
