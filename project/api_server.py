# api_server.py
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import uvicorn

app = FastAPI(title="Data Analysis Assistant API")

# Allow browser clients from other origins; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set to your site origin(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str
    session_id: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze")
async def analyze(q: Query):
    # TODO: hook into your analysis pipeline or models
    return {"reply": f"Received: {q.message}"}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        f.write(await file.read())
    # TODO: trigger ingestion or analysis of uploaded file
    return {"filename": file.filename, "saved": True}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            text = await ws.receive_text()
            await ws.send_text(f"Echo: {text}")
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
