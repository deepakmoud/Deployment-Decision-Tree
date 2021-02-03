import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
dataset= pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(sc.transform(final_features))
    print(prediction)
    if prediction==[1]:
        output='Yes'
    else:
        output='No'
        
    return render_template('index.html', prediction_text='Model has prected Item purchase status is: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)