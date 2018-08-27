from flask import Flask, abort, render_template, jsonify, request
from api import get_similar

app = Flask('StrainRecommender')

@app.route('/recommender', methods=['POST'])
def do_recommendation():
    if not request.json:
        abort(400)
    data = request.json

    response = get_similar(data)

    return jsonify(response)

@app.route('/')
def index():
    return render_template('index.html')

app.run(debug=True)
