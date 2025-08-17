from fastapi.middleware.cors import CORSMiddleware

def add_cors(app, origins: list[str]):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
