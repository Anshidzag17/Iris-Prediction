from flask import Flask, render_template, request
import pickle
import numpy as np

#initialize the flask app
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html',prediction_text='')

@app.route('/predict',methods=['POST'])
def predict():
    SL = float(request.form['Sepal_Length'])
    SW = float(request.form['Sepal_Width'])
    PL = float(request.form['Petal_Length'])
    PW = float(request.form['Petal_Width'])
    input = np.array([[SL,SW,PL,PW]])
    model = pickle.load(open("model.pkl",'rb'))
    result = model.predict(input)
    if result == 0:
        result='setosa'
    elif result == 1:
        result='versicolor'
    elif result == 2:
        result='virginica'
    else:
        result='error'
    return render_template('index.html',prediction_text= result)

if __name__ == "__main__":
    app.run(debug=True)
    

