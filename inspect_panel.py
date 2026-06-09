import pandas as pd

panel = pd.read_csv("data/panel.csv")
modelling = panel[panel["in_modelling_set"]]

print("=== Modelling rows by year ===")
print(modelling.groupby("report_year").agg(
    n=("worsened_next_year", "size"),
    n_worsened=("worsened_next_year", "sum"),
    rate=("worsened_next_year", "mean"),
).round(3))

print("\n=== Top 10 departments in modelling set ===")
print(modelling["department"].value_counts().head(10))
