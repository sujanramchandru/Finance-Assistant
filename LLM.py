from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import Optional

try:
    from gpt4all import GPT4All
except ImportError:
    GPT4All = None

app = FastAPI()

MODEL_NAME = "mistral-7b-openorca.Q2_K.gguf"
MODEL_PATH = "./models"
MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)

model: Optional[object] = None
model_load_error: Optional[str] = None

if GPT4All is None:
    model_load_error = "gpt4all package is not installed. Please install it in your environment."
elif not os.path.isdir(MODEL_PATH):
    model_load_error = f"Model directory does not exist: {MODEL_PATH}"
elif not os.path.isfile(MODEL_FILE):
    model_load_error = f"Model file does not exist: {MODEL_FILE}"
else:
    try:
        model = GPT4All(MODEL_NAME, model_path=MODEL_PATH, allow_download=False)
    except Exception as e:
        model_load_error = f"Failed to load model: {str(e)}"

class QueryRequest(BaseModel):
    user_query: str
    retrieved_docs: list[str]

@app.post("/generate/")
async def generate_response(request: QueryRequest):
    if model_load_error:
        raise HTTPException(status_code=500, detail=f"Model not available: {model_load_error}")
    if not request.retrieved_docs:
        raise HTTPException(status_code=400, detail="No documents provided for context.")
    try:
        documents = "\n\n".join(doc.strip() for doc in request.retrieved_docs if doc.strip())
        prompt = f"""
You are a knowledgeable and concise financial assistant.
Respond to the user's question using only the information provided in the retrieved documents.
If the answer is not present in those documents, clearly respond with:
\"I'm sorry, I couldn't find that information in the provided data.\"
Keep your answers natural, well-structured, and easy to understand. Use full sentences and avoid listing raw data unless it's relevant to the user’s query.

Documents:
{documents}

User Query:
{request.user_query}

Answer:
"""
        output = model.generate(prompt, max_tokens=300, temp=0.4)
        return {"response": output.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
