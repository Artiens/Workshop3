import json
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
model_1 = RandomForestClassifier(random_state=42)
#model_2 = LogisticRegression(max_iter=200)
#model_3 = SVC(probability=True, random_state=42)

model_1.fit(X_train, y_train)
#model_2.fit(X_train, y_train)
#model_3.fit(X_train, y_train)

# Initialize JSON database for balances
json_db = "balances.json"
def init_database():
    data = {
        "models": {
            "model_1": {"balance": 1000, "weight": 0.5},
        }
    }
    with open(json_db, "w") as f:
        json.dump(data, f, indent=4)

# Load database
def load_database():
    with open(json_db, "r") as f:
        return json.load(f)

# Save database
def save_database(data):
    with open(json_db, "w") as f:
        json.dump(data, f, indent=4)

# Initialize the database if it doesn't exist
try:
    load_database()
except FileNotFoundError:
    init_database()

@app.route('/')
def home():
    return "Welcome to the Decentralized Prediction System! Use endpoints like /predict"

@app.route('/predict', methods=['GET'])
def predict():
    # Récupérer les features depuis les paramètres d'URL
    features = request.args.get("features", "")
    
    # Si features est vide, on peut définir des valeurs par défaut
    if features == "":
        features = [5.1, 3.5, 1.4, 0.2]  # Par exemple, des valeurs par défaut
    else:
        features = [float(x) for x in features.split(",")]

    db = load_database()

    predictions = {}
    prediction = model_1.predict([features])[0]
    # On associe le nom de la fleur à l'indice de la classe
    flower_name = data.target_names[prediction]
    predictions["model_1"] = flower_name
    return jsonify(predictions)




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
