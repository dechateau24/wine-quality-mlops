import pandas as pd

df = pd.read_csv("data/raw/wine_quality.csv")

print("Taille :", df.shape)
print("Doublons :", df.duplicated().sum())

print("\nRepartition de la note (quality) :")
print(df["quality"].value_counts().sort_index())

print("\nCorrelation avec quality :")
print(df.corr()["quality"].sort_values(ascending=False).round(2))