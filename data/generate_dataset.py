import pandas as pd
import numpy as np

np.random.seed(42)

n=1000
data=pd.DataFrame({
    "DSCR": np.random.uniform(0.8, 3.0, n),
    "LLCR": np.random.uniform(1.0, 4.0, n),
    "PLCR": np.random.uniform(1.0, 5.0, n),
    "Leverage": np.random.uniform(20, 90, n),
    "GDP_Growth": np.random.uniform(-5, 10, n),
    "Inflation": np.random.uniform(1, 15, n),
    "Construction_Progress": np.random.uniform(10, 100, n),
    "Contractor_Score": np.random.uniform(30, 100, n),
    "Country_Rating": np.random.randint(1, 10, n)
})

data["Default"] = (
    (data["DSCR"] < 1.2) |
    (data["Leverage"] > 70) |
    (data["Inflation"] > 10)
).astype(int)

data.to_csv("data/infrarisk_dataset.csv", index=False)

print("Dataset Generated Successfully")