from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('catboost_model-2.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        RestingBP = int(request.form['RestingBP'])
        Cholesterol = int(request.form['Cholesterol'])
        Oldpeak = float(request.form['Oldpeak'])
        FastingBS = int(request.form['FastingBS'])
        MaxHR = int(request.form['MaxHR'])
        prediction = model.predict([[Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak
                                     ]])

        if prediction[0] == 1:
            return render_template('index.html', prediction_text="Kindly make an appointment with the doctor!")

        else:
            return render_template('index.html', prediction_text="You are well. No worries :)")

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080, debug=True)
    # gunicorn - -bind 0.0.0.0: 8080 app: app
