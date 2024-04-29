from flask import Flask, render_template, request
import joblib
from textblob import TextBlob


# Initialisation de Flask
app = Flask(__name__)

# Chargement du modèle et du vectoriseur sauvegardés
vectoriseur = joblib.load("Save_Train/tfidf_vectorizer_IMDB_Dataset.pkl")
model = joblib.load("Save_Train/logistic_regression_model_IMDB_Dataset.pkl")

# Fonction de prétraitement
def preprocessor_texte(texte):
    texte = TextBlob(str(texte)).lower().string.strip()  # Prétraitement de base
    return texte

# Route pour la page d'accueil
@app.route("/")
def index():
    return render_template("index.html")

# Route pour afficher le résultat avec la couleur appropriée
@app.route("/result", methods=["POST"])
def result():
    texte = request.form["texte"]  # Obtenir le texte du formulaire
    
     # Vérifier si le texte est vide
    if not texte.strip():
        # Si le texte est vide, retourner une réponse appropriée
        print("le text est un vide..")
    
    texte_pretraité = preprocessor_texte(texte)
    vecteur = vectoriseur.transform([texte_pretraité])
    sentiment_predit = model.predict(vecteur)[0]

    # Détermine la classe CSS en fonction du sentiment
    if sentiment_predit == "positive":
        sentiment_classe = "positive"
    elif sentiment_predit == "negative":
        sentiment_classe = "negative"
    else:
        sentiment_classe = "positive"  # Par défaut, pas de couleur spécifique

    return render_template(
        "result.html", 
        texte=texte, 
        sentiment_predit=sentiment_predit, 
        sentiment_classe=sentiment_classe
    )

# Démarrer le serveur Flask
if __name__ == "__main__":
    app.run(debug=True)
    
#Pour  lancer application en utileser cette commde---->python C:\Users\otman\Desktop\Flask\app.py    
