import pandas as pd

df = pd.read_csv("powermeasure.csv", header=None, names=['12v_back_a',
    '12v_back_v','12v_aux_a', '12v_aux_v'], usecols=[0,1,12,13])

df["Sum"] = (df["12v_back_v"] * df["12v_back_a"]) + (df["12v_aux_v"] *
        df["12v_aux_a"])

print(df.head())
print("Average:", df["Sum"].mean())
print("Max:", df["Sum"].max())
