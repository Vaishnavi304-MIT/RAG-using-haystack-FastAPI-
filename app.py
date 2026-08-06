from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from dotenv import load_dotenv

from QASystem.Retrival_and_generation import get_result

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("Import Successfully")

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@app.post("/get_answer")
async def get_answer(question: str = Form(...)):
    try:
        answer = get_result(question)

        return JSONResponse(
            content={
                "answer": str(answer)
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "answer": str(e)
            }
        )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )