# %%
import pandas as pd

df = pd.read_csv("./results/DefiNet_low_density_WSe2.csv")
for col in df.columns[1:]:
    print(f"{col}: ", df[col].mean().round(6))
# %%

