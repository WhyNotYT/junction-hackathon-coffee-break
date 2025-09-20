from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama

app = FastAPI(title="My Ollama API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-oss:20b"

class ChatResponse(BaseModel):
    response: str
    model: str

# Fix CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "https://nordxgpt.vercel.app",  # Remove trailing slash
        "https://irnfnv-ip-130-231-176-211.tunnelmole.net",
        "http://localhost:3000",  # For local testing
        "*"  # Allow all origins for testing (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Ollama FastAPI server is running!"}

@app.get("/models")
async def list_models():
    """Get list of available Ollama models"""
    try:
        models = ollama.list()
        return {"models": [model['name'] for model in models['models']]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ollama(request: ChatRequest):
    """Chat with Ollama model"""
    try:
        response = ollama.chat(model=request.model, messages=[
            {
                'role': 'user',
                'content': request.message,
            },
        ])
        
        return ChatResponse(
            response=response['message']['content'],
            model=request.model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

@app.post("/generate")
async def generate_text(request: ChatRequest):
    """Generate text using Ollama"""
    try:
        response = ollama.generate(model=request.model, prompt=request.message)
        return {"response": response['response'], "model": request.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")