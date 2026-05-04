from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, Response
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
from groq import Groq
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

jinja_env = Environment(loader=FileSystemLoader("templates"))

# ── Load ML model ──────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    payload = pickle.load(f)

model            = payload["model"]
FEATURES         = payload["features"]
product_list     = payload["product_list"]
country_list     = payload["country_list"]
salesperson_list = payload["salesperson_list"]
r2_score_val     = payload["r2_score"]
mae_val          = payload["mae"]
importances      = payload["importances"]

# ── Load raw CSV ────────────────────────────────────────────────
raw = pd.read_csv("skincare_sales_data.csv")
raw.columns = (raw.columns.str.strip().str.lower()
               .str.replace(" ", "_")
               .str.replace("(", "").str.replace(")", "")
               .str.replace("$", "dollar"))
raw["date"]  = pd.to_datetime(raw["date"])
raw["month"] = raw["date"].dt.month

# ── Groq client (FREE) ──────────────────────────────────────────
# Get your FREE API key from: https://console.groq.com
# Step 1: Sign up at https://console.groq.com (no credit card needed)
# Step 2: Go to API Keys → Create API Key
# Step 3: Set it below OR as environment variable GROQ_API_KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
ai_client = Groq(api_key=GROQ_API_KEY)

# ── AI system prompt (full sales data baked in) ─────────────────
SALES_SYSTEM_PROMPT = """You are an expert AI Sales Analyst for a skincare company. 
Answer questions based ONLY on the data below. Be concise and cite specific numbers.
Give actionable business recommendations when relevant. Keep answers to 3-6 sentences 
unless the user asks for a full report.

=== SALES DATA ===
OVERVIEW:
- Total Revenue: $2,909,104.12
- Total Orders: 374
- Period: Jan 2022 – Aug 2022 (8 months)
- Avg Order Value: $7,778.35

REVENUE BY PRODUCT:
1. Tea Tree Moisturizer: $260,905.44 (8,319 boxes)
2. Hydrating Face Serum: $250,323.33 (7,977 boxes)
3. Hair Repair Oil: $232,864.77 (6,983 boxes)
4. Anti-Aging Serum: $232,248.00 (7,423 boxes)
5. Body Butter Cream: $222,923.58 (7,065 boxes)
6. SPF 50 Sunscreen: $218,576.97 (7,018 boxes)
7. Vitamin C Cream: $218,210.84 (7,182 boxes)
8. Aloe Vera Gel: $202,901.75 (6,941 boxes)
9. Face Sheet Masks: $197,882.89 (5,893 boxes)
10. Under Eye Cream: $195,936.57 (6,184 boxes)
11. Lip Balm Pack: $170,100.69 (6,281 boxes)
12. Rose Water Toner: $151,324.68 (4,455 boxes)
13. Niacinamide Toner: $141,841.47 (4,257 boxes)
14. Salicylic Acid Cleanser: $110,329.72 (3,613 boxes)
15. Charcoal Face Wash: $102,733.42 (3,562 boxes)

REVENUE BY COUNTRY:
1. USA: $628,487.86 (21.6%)
2. New Zealand: $557,059.85 (19.1%)
3. Australia: $505,497.64 (17.4%)
4. UK: $497,061.54 (17.1%)
5. Canada: $374,562.31 (12.9%)
6. India: $346,434.92 (11.9%)

REVENUE BY SALESPERSON:
1. Olivia D'Souza: $387,405.91
2. Sophia Nair: $319,887.82
3. Isabella Roy: $302,087.60
4. Ethan Reddy: $298,595.61
5. Lucas Verma: $295,166.91
6. Ananya Gupta: $293,204.67
7. Noah Mehta: $272,188.08
8. Liam Patel: $270,960.55
9. Ava Sharma: $246,174.28
10. Mason Kapoor: $223,432.69

TOP COUNTRY-PRODUCT COMBOS:
- USA + Anti-Aging Serum: $113,821.81
- Australia + Hair Repair Oil: $91,002.87
- New Zealand + SPF 50 Sunscreen: $87,897.03
- New Zealand + Body Butter Cream: $86,730.86
- UK + Hydrating Face Serum: $75,719.24

ML MODEL: Random Forest, R²=0.6885, MAE=$2,417.46
Most important feature: Boxes Shipped (77.4% importance)
"""

# ── Pydantic models ─────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]


# ── Routes ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    template = jinja_env.get_template("index.html")
    html = template.render(
        products     = product_list,
        countries    = country_list,
        salespersons = salesperson_list,
        r2_score     = r2_score_val,
        mae          = mae_val,
    )
    return HTMLResponse(content=html)


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.post("/chat")
def chat(req: ChatRequest):
    """AI Sales Analyst chatbot endpoint — powered by Groq (free)."""
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        response = ai_client.chat.completions.create(
            model      = "llama-3.3-70b-versatile",
            messages   = [{"role": "system", "content": SALES_SYSTEM_PROMPT}] + messages,
            max_tokens = 1024,
        )
        reply = response.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err.lower():
            return JSONResponse(status_code=401, content={
                "error": "Invalid Groq API key. Get your free key at https://console.groq.com"
            })
        return JSONResponse(status_code=500, content={"error": err})

@app.get("/predict")
def predict(month: int, product: str, country: str,
            salesperson: str, boxes: int = 100):
    try:
        product_code     = product_list.index(product)
        country_code     = country_list.index(country)
        salesperson_code = salesperson_list.index(salesperson)
        quarter          = (month - 1) // 3 + 1

        input_df = pd.DataFrame([{
            "month":             month,
            "quarter":           quarter,
            "dayofweek":         2,
            "product_code":      product_code,
            "country_code":      country_code,
            "sales_person_code": salesperson_code,
            "boxes_shipped":     boxes,
        }])

        prediction = model.predict(input_df)[0]
        return {
            "predicted_sales": round(float(prediction), 2),
            "product":         product,
            "country":         country,
            "month":           month,
            "boxes":           boxes,
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/analytics/kpi")
def kpi():
    return {
        "total_sales":  round(float(raw["amount_dollar"].sum()), 2),
        "total_orders": int(len(raw)),
        "total_boxes":  int(raw["boxes_shipped"].sum()),
        "avg_order":    round(float(raw["amount_dollar"].mean()), 2),
        "top_product":  str(raw.groupby("product")["amount_dollar"].sum().idxmax()),
        "top_country":  str(raw.groupby("country")["amount_dollar"].sum().idxmax()),
        "model_r2":     r2_score_val,
        "model_mae":    mae_val,
    }


@app.get("/analytics/by-month")
def by_month():
    grouped = raw.groupby("month")["amount_dollar"].sum()
    names   = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
    return {
        "labels": [names[m-1] for m in grouped.index],
        "values": [round(v, 2) for v in grouped.values.tolist()]
    }


@app.get("/analytics/by-country")
def by_country():
    g = raw.groupby("country")["amount_dollar"].sum().sort_values(ascending=False)
    return {"labels": g.index.tolist(),
            "values": [round(v, 2) for v in g.values.tolist()]}


@app.get("/analytics/by-product")
def by_product():
    g = raw.groupby("product")["amount_dollar"].sum().sort_values(ascending=False)
    return {"labels": g.index.tolist(),
            "values": [round(v, 2) for v in g.values.tolist()]}


@app.get("/analytics/by-salesperson")
def by_salesperson():
    g = raw.groupby("sales_person")["amount_dollar"].sum().sort_values(ascending=False)
    return {"labels": g.index.tolist(),
            "values": [round(v, 2) for v in g.values.tolist()]}


@app.get("/analytics/importance")
def importance():
    s = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    labels = [k.replace("_code","").replace("_"," ").title() for k,_ in s]
    values = [round(v*100, 2) for _,v in s]
    return {"labels": labels, "values": values}
