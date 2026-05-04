import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# Load & clean
data = pd.read_csv("skincare_sales_data.csv")
data.columns = (data.columns.str.strip().str.lower()
                .str.replace(" ", "_")
                .str.replace("(", "").str.replace(")", "")
                .str.replace("$", "dollar"))

print("Columns:", data.columns.tolist())

data['date']      = pd.to_datetime(data['date'])
data['month']     = data['date'].dt.month
data['quarter']   = data['date'].dt.quarter
data['dayofweek'] = data['date'].dt.dayofweek

# Encode categories
for col in ['product', 'country', 'sales_person']:
    data[col + '_code'] = data[col].astype('category').cat.codes

product_list     = list(data['product'].astype('category').cat.categories)
country_list     = list(data['country'].astype('category').cat.categories)
salesperson_list = list(data['sales_person'].astype('category').cat.categories)

# Features & target
FEATURES = ['month', 'quarter', 'dayofweek', 'product_code',
            'country_code', 'sales_person_code', 'boxes_shipped']
X = data[FEATURES]
y = data['amount_dollar']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R2 Score : {r2:.4f}")
print(f"MAE      : ${mae:.2f}")

# Feature importance
importances = dict(zip(FEATURES, model.feature_importances_))

# Save
payload = {
    "model":            model,
    "features":         FEATURES,
    "product_list":     product_list,
    "country_list":     country_list,
    "salesperson_list": salesperson_list,
    "r2_score":         round(r2, 4),
    "mae":              round(mae, 2),
    "importances":      importances,
}
with open("model.pkl", "wb") as f:
    pickle.dump(payload, f)

print("model.pkl saved!")