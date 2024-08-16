from flask import Flask, request, jsonify
import json
import requests 

app = Flask(__name__)

@app.route("/", methods = ["GET"])
def hello():
  return "hello, world\n"


@app.route("/api/v1/chat/completions/", methods = ["POST"])
def infer():
  data = request.data
  headers = {"Content-Type":"application/json"}
  res = requests.post("http://localhost:5001/", headers=headers, data=json.dumps(data))

  return jsonify(res.json())

if __name__ == "__main__":
  app.run(host = '0.0.0.0', port = '5000', debug = True)
