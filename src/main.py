from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from agents.agentic_rag_service import AgenticRAGService
from middlewares.auth_middleware import AuthMiddleware
from routes import  documents_router, projects_router, query_router, auth_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print("🚀 App is starting up! Initializing resources...")

    services = await AgenticRAGService.create()
    app.state.supervisor_agent = services.get_service("supervisor_agent")
    app.state.embedding_service = services.get_service("embedding_service")
    app.state.vector_store = services.get_service("vector_store")

    print("✅ Resources initialized successfully.")

    yield

    # --- Shutdown ---

    print("👋 App shutdown complete. Goodbye!")


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware class
app.add_middleware(AuthMiddleware)
# --- Include Routers ---
 
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(projects_router)
app.include_router(query_router)
 

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "message": "App is running smoothly!"}
    )