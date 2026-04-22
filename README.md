# Pet Adoption Project - Setup Guide

## Estudiantes:
Ariana Víquez
Fabián Barquero 

## Running the Project

### Step 1: Run Notebooks
Navigate to the `notebooks` folder and run all notebooks in the order specified in each file:

1. Open Notebooks
   ```bash
   cd notebooks
   ```

2. Execute each notebook in the `notebooks` folder following the order indicated in the filenames

3. Wait for all notebooks to complete execution

### Step 2: Run the Backend
Once all notebooks have finished executing, start the backend server:

1. Run the application with the correct path:
   ```bash
   python backend/app.py
   ```

2. The server will start and serve the frontend index

### Step 3: Access the Frontend
Once `app.py` is running, the frontend index from the `frontend` folder will be accessible through the running server.

## Project Structure
```
pet_adoption-fabian-analisis/
├── notebooks/          # Data processing and analysis notebooks
├── backend/            # Flask/API backend (app.py)
└── frontend/           # Frontend files (index.html, etc.)
```

## Notes
- Ensure all notebook executions complete successfully before running `app.py`
- Follow the order specified in each notebook filename
- The backend server must be running to access the frontend
