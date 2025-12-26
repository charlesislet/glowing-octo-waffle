from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from rag_pipeline import RAGPipeline

app = FastAPI(title="RAG QA Service")
app.mount("/static", StaticFiles(directory="static"), name="static")

pipeline: RAGPipeline


@app.on_event("startup")
def startup() -> None:
    global pipeline
    pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


@app.get("/", response_class=HTMLResponse)
def serve_frontend() -> str:
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/query")
def query_rag(payload: QueryRequest) -> JSONResponse:
    question = payload.query.strip()
    if not question:
        return JSONResponse({"status": "error", "error": "請輸入查詢內容"}, status_code=400)

    try:
        result = pipeline.query(question, top_k=payload.top_k)
        return JSONResponse(result)
    except Exception as exc:  # 捕捉後端錯誤，避免洩漏堆疊
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("main_rag:app", host="0.0.0.0", port=8002, reload=True)
