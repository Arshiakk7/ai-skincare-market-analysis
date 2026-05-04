 # AI Powered Skincare Market Analysis and Prediction Tool

A full-stack market analytics dashboard with ML-based sales prediction and an AI chatbot analyst powered by Groq (Llama 3), built with FastAPI.

---

## Project Structure

```
skincare_app/
├── main.py                  ← FastAPI backend (all routes + AI chat)
├── model.py                 ← Train the ML model (run once)
├── model.pkl                ← Trained Random Forest model
├── skincare_sales_data.csv  ← Raw sales data
├── requirements.txt         ← Python dependencies
├── .env.example             ← Copy to .env and add your API key
├── templates/
│   └── index.html           ← Full dashboard UI
└── static/
    └── css/
        └── placeholder.css
```

---

## Setup — Step by Step

### Step 1 — Open folder in VS Code
```
File → Open Folder → select skincare_app
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
```

Activate it:
- **Windows:**   `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Get your FREE grok API key

1. Go to  https://console.groq.com
2. Sign up (free account)
3. Click "API Keys" → "Create Key"
4. Copy the key (starts with sk-ant-...)

### Step 5 — Set the API key

** — Paste directly in main.py**

Open main.py, find this line:
```python
GROQ_API_KEY = = os.environ.get("GROQ_API_KEY" , "YOUR_API_KEY_HERE")
```
Replace `YOUR_API_KEY_HERE` with your actual key.

### Step 6 — Run the server
```bash
uvicorn main:app --reload
```

### Step 7 — Open in browser
```
http://localhost:8000
```

---

## Features

| Page | Description |
|------|-------------|
| Overview | KPI cards + monthly/country/product charts |
| Predict Sales | ML model predicts revenue for any scenario |
| Analytics | Full product + salesperson breakdown |
| Leaderboard | Rankings with visual bars |
| AI Analyst | Chat with Claude about your sales data |
| Model Info | R² score, MAE, feature importance |

---

## Retrain the Model (optional)
```bash
python model.py
```
This regenerates model.pkl from the CSV.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| AI chat returns 401 | API key is wrong or not set |
| AI chat not working | Make sure FastAPI is running on port 8000 |
| Charts not loading | Check browser console for API errors |
