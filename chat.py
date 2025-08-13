from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import random
import json

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from remove_accents import remove_accents

# Setup Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for React frontend

# Load intents
with open("intents.json", encoding="utf-8") as json_data:
    intents = json.load(json_data)

# Load trained model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Γραμματεία"

@app.route("/chat", methods=["POST"])
def chat():
    data_json = request.get_json()
    sentence = data_json.get("message")
    if not sentence:
        return jsonify({"response": "Παρακαλώ πληκτρολογήστε κάτι."})
    
    # Preprocess
    sentence = remove_accents(sentence.lower())
    X = bag_of_words(tokenize(sentence), all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Inference
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Confidence score
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return jsonify({"response": random.choice(intent["responses"])})
    
    return jsonify({"response": "Δεν κατάλαβα... Προσπαθήστε ξανά."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
