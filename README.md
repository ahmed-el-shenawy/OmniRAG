# MultiRAG Chatbot

**MultiRAG Chatbot** is an intelligent query system leveraging **Retrieval-Augmented Generation (RAG)** to answer queries over documents, structured databases, and web sources. It implements a **supervisor agent** to orchestrate multiple worker agents and ensures **secure access** with authentication and authorization.

* **LLM**: Generation via `qwen/qwen3-32b` through Groq API.
* **Embeddings**: `nomic-embed-text` locally via Ollama.
* **Database**: PostgreSQL with `pgvector` for vector storage (via Docker).
* **Supervisor Agent**: Routes queries to the appropriate agent (`rag_agent`, `sql_agent`, `web_agent`) and combines results.
* **Authentication**: Only authenticated users can access API endpoints.
* **Authorization**: Users can query only projects they are authorized for; admins can manage data and permissions.
* **Tokens**: Uses short-lived **access tokens** and long-lived **refresh tokens**.
* **Flexibility**: Models can be swapped as long as they follow OpenAI API standards (configured via `.env`).

## 1. Setup

### 1.1 Install dependencies

* Install **Docker**.

  * On Windows, install **WSL2** first.

* Create a Python virtual environment (example using Conda):

```bash
conda create -n istud python=3.12
conda activate istud
```

* Copy environment files and update credentials:

```bash
mv src.env.example .env
mv docker.env.example .env
# Edit .env to set API keys, DB URLs, and other values
```

* Install Python requirements:

```bash
cd src
pip install -r requirements.txt
```

### 1.2 Start Docker services

```bash
cd docker
docker compose up
```

### 1.3 Apply database migrations

```bash
cd src
alembic upgrade head
```

### 1.4 Start FastAPI app

```bash
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

App will run at `http://localhost:5000`.

---

## 2. API Routes

### 2.1 Authentication & Authorization (`/auth`)

Manages user accounts, roles, tokens, and permissions.

| Endpoint       | Method | Description                         |
| -------------- | ------ | ----------------------------------- |
| `/signup`      | POST   | Register a new user                 |
| `/login`       | POST   | Authenticate user & generate tokens |
| `/refresh`     | POST   | Refresh access token                |
| `/logout`      | POST   | Logout & invalidate tokens          |
| `/authorize`   | POST   | Grant project permissions (admin)   |
| `/deauthorize` | POST   | Revoke project permissions (admin)  |
| `/update-role` | POST   | Update user role (admin only)       |

> **Note**: Only admins can modify data or roles; users can query only projects they are authorized for.

---

### 2.2 Documents (`/documents`)

Upload, process, manage, and search project documents.

| Endpoint                 | Method | Description                             |
| ------------------------ | ------ | --------------------------------------- |
| `/upload/{project_name}` | POST   | Upload files to a project               |
| `/process`               | POST   | Generate embeddings for uploaded docs   |
| `/flush`                 | POST   | Remove embeddings from the system       |
| `/delete`                | POST   | Delete specific documents               |
| `/`                      | GET    | List documents for a project            |
| `/search`                | POST   | Search a document by project & filename |

---

### 2.3 Query (`/query`)

Send queries to the supervisor agent, which orchestrates worker agents (`rag_agent`, `sql_agent`, `web_agent`) and combines results.

```
Authorization: Users must be authorized for the project to query its documents.
```

| Endpoint | Method | Description                           |
| -------- | ------ | ------------------------------------- |
| `/`      | POST   | Send query and get structured results |

**Example Request:**

```json
POST /query
{
  "query": "How many users are in the system?"
}
```

**Example Response:**

```json
{
  "final_answer": "There are 12 users in the system.",
  "agents_used": ["sql_agent"]
}
```

> The `agents_used` field shows which agents were called to generate the answer. Multiple agents may be called and their results combined.

---

## 3. Supervisor Agent Workflow

1. **Analyze query**: Determine if it is about documents, database, or web.
2. **Call appropriate agents**:

   * `rag_agent`: Document-based queries (e.g., Ahmed’s resume).
   * `sql_agent`: Database queries (users, orders, products).
   * `web_agent`: Fallback for up-to-date web info.
3. **Combine results**: Aggregate multiple agent responses into a short, precise answer.
4. **Return structured output**:

```json
{
  "final_answer": "<answer_text>",
  "agents_used": ["rag_agent", "sql_agent"]
}
```

---

## 4. Notes

* Ensure `.env` contains correct model, API keys, and DB URLs.
* Docker is required for PostgreSQL with `pgvector`.
* Models and embeddings can be replaced as long as they are OpenAI-compatible.
* Users can only query projects they are authorized for; admins manage permissions.

---

