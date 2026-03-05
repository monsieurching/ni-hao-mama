import os

port = int(os.environ.get("PORT", 8080))
os.execlp(
    "gunicorn", "gunicorn", "app:app",
    "--bind", f"0.0.0.0:{port}",
    "--timeout", "120",
    "--workers", "1",
)
