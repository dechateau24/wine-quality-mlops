\# Prediction de la qualite du vin (MLOps)



Pipeline MLOps complet pour predire la note de qualite (3 a 9) d'un vin

portugais a partir de 11 mesures physico-chimiques (acidite, pH, alcool...).

Probleme de \*\*regression\*\*.



Outils : Git, DVC, MLflow, FastAPI, Docker, Evidently.



\## Donnees



\- Dataset `wine\_quality` (OpenML), 6497 vins, rouges et blancs melanges.

\- 1179 doublons supprimes dans `src/prepare.py` : un meme vin present dans

&#x20; le train et le test fausserait le score.

\- Split 80 / 20 : 4254 vins en train, 1064 en test.

\- Les donnees sont versionnees avec DVC (jamais dans Git).



\## Resultats



| Modele | RMSE | MAE | R2 |

|---|---|---|---|

| ridge | 0.7139 | 0.5520 | 0.3206 |

| random\_forest | \*\*0.6755\*\* | \*\*0.5127\*\* | \*\*0.3918\*\* |

| gradient\_boosting | 0.6816 | 0.5182 | 0.3808 |



Le meilleur modele (RMSE la plus basse) recoit l'alias `production` dans le

registry MLflow. Les correlations avec la qualite sont faibles (alcool : 0.44),

donc un R2 autour de 0.4 est attendu sur ce dataset.



\## Structure



```

data/raw/            donnees brutes (suivies par DVC)

src/prepare.py       nettoyage et split train/test

src/train.py         entrainement de 3 modeles, tracking MLflow, registry

src/download\_data.py telechargement du dataset depuis OpenML

api/main.py          API FastAPI (/health, /predict, /drift-report)

api/drift.py         rapport de derive Evidently

params.yaml          parametres (split, hyperparametres)

dvc.yaml             pipeline : prepare -> train

Dockerfile           image de l'API

```



\## Reproduire



```bash

git clone <url-du-depot>

cd wine-quality-mlops

python -m venv venv

venv\\Scripts\\activate          # Mac/Linux : source venv/bin/activate

pip install -r requirements.txt



python src/download\_data.py    # si le remote DVC n'est pas disponible

dvc repro                      # lance prepare puis train

dvc status                     # doit afficher : up to date

```



Resultats MLflow :



```bash

set MLFLOW\_ALLOW\_FILE\_STORE=true

mlflow ui --backend-store-uri ./mlruns

```



Puis http://localhost:5000, experience `wine\_quality`, onglet Models.



\## API et Docker



```bash

docker build -t wine-api .

docker run -p 8000:8000 wine-api

```



| Route | Role |

|---|---|

| `GET /health` | renvoie `{"status": "ok"}` |

| `POST /predict` | renvoie la note predite |

| `GET /drift-report` | rapport HTML Evidently |



Exemple pour `/predict` :



```json

{"fixed\_acidity": 7.4, "volatile\_acidity": 0.7, "citric\_acid": 0.0,

&#x20;"residual\_sugar": 1.9, "chlorides": 0.076, "free\_sulfur\_dioxide": 11,

&#x20;"total\_sulfur\_dioxide": 34, "density": 0.9978, "pH": 3.51,

&#x20;"sulphates": 0.56, "alcohol": 9.4}

```



Reponse attendue : `{"predicted\_quality": 5.13}` (note reelle : 5).



\## Monitoring



`/drift-report` compare les donnees d'entrainement a un echantillon de 500 vins

dont `alcohol` (+1.5) et `volatile.acidity` (x1.4) ont ete decales a la main.

Evidently detecte la derive sur ces 2 features sur 11. La derive est

\*\*simulee\*\* : ce n'est pas du trafic reel.



\## Choix et limites



\- L'API charge `models/model.joblib` (export de `train.py`) plutot que le

&#x20; registry MLflow : le stockage par dossier ecrit des chemins absolus Windows

&#x20; qui n'existent pas dans le conteneur.

\- Le remote DVC est un dossier local (`../dvc\_storage`).

\- Le modele se trompe d'environ 0.7 point sur la note, et il predit mal les

&#x20; notes extremes (3 et 9 sont tres rares).

