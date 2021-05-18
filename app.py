
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('button.html')
@app.route('/predict1',methods=['POST'])
def predict1():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model1.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('regular_student.html', prediction_text='Package should be {} LPA and Accuracy of the model is 99%.'.format(output))
@app.route('/predict_api1',methods=['POST'])
def predict_api1():
    '''
    For direct API calls trought request
    '''
    data = request1.get_json(force=True)
    prediction = model1.predict1([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
@app.route('/predict2',methods=['POST'])
def predict2():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model2.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('diploma_student.html', prediction_text='Package should be {} LPA and Accuracy of the model is 100%.'.format(output))
@app.route('/predict_api2',methods=['POST'])
def predict_api2():
    '''
    For direct API calls trought request
    '''
    data = request2.get_json(force=True)
    prediction = model2.predict2([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
@app.route('/regular_student')
def regular_student():
    return render_template('regular_student.html')
@app.route('/diploma_student')
def diploma_student():
    return render_template('diploma_student.html')
if __name__ == "__main__":
    app.run(debug=True)

