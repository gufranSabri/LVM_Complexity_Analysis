from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import argparse
import torch
import torch.nn as nn
from torchvision import models
from utils import * 
import webbrowser

app = Flask(__name__)
app.secret_key = 'your_secret_key' 
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return render_template("index.html")

def hook_fn(module, input, output):
    input_size = input[0].shape
    layer_name = str(module.__class__).split(".")[-1].split("'")[0]
    socketio.emit("layer_size", {"layer": layer_name, "size": str(input_size)})

@socketio.on("request_model")
def handle_request_model(data):
    model_type = data.get("model")
    if model_type not in ["vit", "deit", "swin", "resnet"]:
        emit("model_response", {"error": "Invalid model type"})
        return
    
    model = get_model(model_type)
    
    for layer in model.children():
        layer.register_forward_hook(hook_fn)

    dummy_input = torch.randn(1, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        model(dummy_input)

    model_str = str(model)
    emit("model_response", {"model": model_str})

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:3000")
    socketio.run(app, host="0.0.0.0", port=3000, debug=True)
    