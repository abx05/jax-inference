import torch
import transformers
from transformers import AutoTokenizer, GPTJForCausalLM
from flask import Flask, request, jsonify


num_gpus = torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "./model"
tokenizer = "EleutherAI/gpt-j-6b"

model = GPTJForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer)

def predict(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=1024, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict_route():
    prompt = request.json['prompt']
    return jsonify(predict(prompt))

