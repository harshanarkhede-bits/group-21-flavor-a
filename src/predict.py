"""Backward compatibility bridge for FastAPI serving application."""
from src.serving.api import app

if __name__ == "__main__":
    import uvicorn
    from src.config import config
    uvicorn.run("src.serving.api:app", host=config.serving_host, port=config.serving_port, reload=True)
