import os
from sklearn.datasets import fetch_openml

wine = fetch_openml("wine_quality", version=1, as_frame=True)
df = wine.frame

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/wine_quality.csv", index=False)

print("Fichier cree : data/raw/wine_quality.csv")
print("Taille :", df.shape)