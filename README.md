# Expense Manager (Microservices + Docker)

This project runs a containerized Expense Manager app using only open-source components:

- **PostgreSQL** (`postgres:16-alpine`) for data storage
- **FastAPI** backend for REST CRUD APIs
- **Streamlit** frontend for browser UI

## Run in your browser

### 1) Start services

```bash
docker compose up --build
```

### 2) Open the app

- Frontend (UI): http://localhost:8501
- Backend docs (Swagger): http://localhost:8000/docs

### 3) Stop services

```bash
docker compose down
```

If you also want to remove DB data volume:

```bash
docker compose down -v
```

## Features

- Add Expense (Date, Category, Amount)
- View Expenses (Table)
- Spending visuals (Bar chart by category)

## Service layout

```text
.
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py
```
