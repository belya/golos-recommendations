# -*- coding: UTF-8 -*-
from flask import Flask, jsonify, request
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
  user_id = uid_to_idx[float(request.args.get("user"))]
  prediction = model.recommend(user_id, user_item, N=20)
  result = [{idx_to_mid[item]: rating} for item, rating in prediction]
  return jsonify(result)

# TODO replace toarray() method
@app.route('/history', methods=['GET'])
def history():
  user_id = uid_to_idx[float(request.args.get("user"))]
  positive_user_item = user_item > 0
  return jsonify([idx_to_mid[i] for i in positive_user_item[user_id].indices])

if __name__ == '__main__':
  uid_to_idx = joblib.load('./model3/uid_to_idx.pkl')
  idx_to_mid = joblib.load('./model3/idx_to_mid.pkl')
  user_item = joblib.load('./model3/matrix.pkl')
  print(user_item.shape)
  model = joblib.load('./model3/model.pkl')
  app.run(port=8080)
