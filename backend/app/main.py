from fastapi import FastAPI

app = FastAPI(title="Invoice Diploma API")

@app.get("/health")
def health():
    return {"status": "ok"}