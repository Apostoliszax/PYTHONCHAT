from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import random
import json

from model import NeuralNet
from nltk_utils import BagOfWords, tokenize
from removeAccents import RemoveAccents

app = Flask(__name__)
CORS(app)

with open("intents.json", encoding="utf-8") as json_data:
    intents = json.load(json_data)
FILE = "data.pth"
data = torch.load(FILE)

InputSize = data["input_size"]
HiddenSize = data["hidden_size"]
OutputSize = data["output_size"]
AllWords = data["all_words"]
tags = data["tags"]
ModelState = data["model_state"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(InputSize, HiddenSize, OutputSize, dropout=0.0).to(device)
model.load_state_dict(ModelState)
model.eval()

def PreprocessSentence(sentence):
    sentence = RemoveAccents(sentence.lower())
    return BagOfWords(tokenize(sentence), AllWords)

def GetResponseFromModel(sentence):
    X = PreprocessSentence(sentence)
    X = X.reshape(1, X.shape[0])
    X_tensor = torch.from_numpy(X).float().to(device)
    output = model(X_tensor)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(f"DEBUG Input: {sentence}, Predicted Tag: {tag}, Confidence: {prob.item()}") # debugging
    ProbLimit = 0.75
    if prob.item() > ProbLimit:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
    return None

@app.route("/chat", methods=["POST"])
def chat():
    dataJson = request.get_json(force=True, silent=True) or {}
    sentence = dataJson.get("message", "").strip()
    response = (
        GetResponseFromModel(sentence) or
        "Δεν κατάλαβα... Προσπαθήστε ξανά."
    )
    response = (
        "Παρακαλώ πληκτρολογήστε κάτι." if not sentence else response
    )
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
