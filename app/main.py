from fastapi import FastAPI

app = FastAPI(title="Aegis v2")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
