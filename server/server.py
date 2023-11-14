from flask import Flask, request
from flask_cors import CORS
from util import *

app = Flask(__name__)
CORS(app)

@app.route('/get_text', methods=['POST'])
def get_text():
    data = request.get_json()
    # print(data['text'])
    response_data = get_text_prediction(data['text']).tolist()
    print(response_data)
    # response_data = {'message': 'Data Success'}
    return response_data

@app.route('/hello')
def hello():
    return "Hey!"

if __name__ == "__main__":
    print("Starting the python server")
    app.run(port=5000, debug=True)