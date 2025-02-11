import requests
import numpy as np

# Liste des serveurs exposés via ngrok
model_urls = [
 
    "https://5763-89-30-29-68.ngrok-free.app/predict",
    "https://2535-89-30-29-68.ngrok-free.app/predict" # URL du modèle 1
]

def get_prediction_from_models(features):
    predictions = []
    
    for url in model_urls:
        # Convertir les features en chaîne séparée par des virgules
        features_str = ",".join(map(str, features))
        
        # Construire les paramètres de la requête
        params = {'features': features_str}  # Pas de paramètre 'model' dans l'URL
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            # On suppose que la réponse JSON contient la prédiction dans une clé spécifique
            prediction = response.json()
            predictions.append(prediction.get("model_1", None))  # Ajoute la prédiction pour ce modèle
        else:
            print(f"Erreur de connexion au modèle {url}")
    
    return predictions

def get_consensus_prediction(features):
    predictions = get_prediction_from_models(features)
    
    # Moyenne des prédictions (choisir la classe la plus fréquente parmi les modèles)
    consensus = []
    for feature_predictions in zip(*predictions):
        # Compter la fréquence de chaque prédiction
        prediction_count = {}
        for prediction in feature_predictions:
            if prediction in prediction_count:
                prediction_count[prediction] += 1
            else:
                prediction_count[prediction] = 1
        
        # Trouver la prédiction la plus fréquente
        consensus_prediction = max(prediction_count, key=prediction_count.get)
        consensus.append(consensus_prediction)
    
    return consensus

# Exemple d'utilisation
features = [5.1, 3.5, 1.4, 0.2]  # Exemple de caractéristiques

consensus_prediction = get_consensus_prediction(features)
print(f"Prédiction consensuelle: {consensus_prediction}")