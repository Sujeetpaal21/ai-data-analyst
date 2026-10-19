"""Generate deliberately messy sample datasets for demos and tests.  python sample_data/generate_sample.py"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1500
out = Path(__file__).parent
cities = {"Delhi": "North", "Lucknow": "North", "Mumbai": "West", "Pune": "West", "Chennai": "South", "Kolkata": "East"}
products = {"Laptop": ("Electronics", 55000), "Phone": ("Electronics", 22000), "Headphones": ("Accessories", 2500),
            "Mouse": ("Accessories", 600), "Desk": ("Furniture", 9000), "Chair": ("Furniture", 6500)}
city = rng.choice(list(cities), n)
prod = rng.choice(list(products), n, p=[.12, .22, .2, .22, .12, .12])
dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 540, n), unit="D")
qty = rng.integers(1, 6, n)
price = np.array([products[p][1] for p in prod]) * rng.normal(1, 0.05, n)
disc = rng.choice([0, 0.05, 0.1, 0.15], n)
rev = qty * price * (1 - disc)
rev = rev * (1 + 0.0006 * (dates - dates.min()).days.values)  # gentle growth
df = pd.DataFrame({"Order_ID": [f"ORD{1000 + i}" for i in range(n)], "Order_Date": dates.strftime("%Y-%m-%d"), "Customer_ID": [f"C{rng.integers(1, 400):04d}" for _ in range(n)],
                   "City": city, "Region": [cities[c] for c in city], "Product": prod, "Category": [products[p][0] for p in prod],
                   "Quantity": qty, "Unit_Price": price.round(2), "Discount": disc, "Revenue": rev.round(2),
                   "Churned": (rng.random(n) < 0.12 + 0.2 * (disc == 0)).astype(int)})
# --- inject realistic problems
df.loc[rng.choice(n, 40, replace=False), "Revenue"] = np.nan
df.loc[rng.choice(n, 25, replace=False), "City"] = np.nan
df.loc[rng.choice(n, 30, replace=False), "City"] = "  delhi "
df.loc[rng.choice(n, 20, replace=False), "Product"] = "LAPTOP"
df.loc[rng.choice(n, 5, replace=False), "Quantity"] = -1
df.loc[rng.choice(n, 6, replace=False), "Revenue"] = df["Revenue"].max() * 3
df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)
df.to_csv(out / "sales_sample.csv", index=False)
with pd.ExcelWriter(out / "multi_sheet_sample.xlsx") as xw:
    df.head(300).to_excel(xw, sheet_name="Sales", index=False)
    df.groupby("Region", as_index=False)["Revenue"].sum().to_excel(xw, sheet_name="By_Region", index=False)
print("written", len(df), "rows")
