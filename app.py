from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 19)
    print(to_predict)
    loaded_dict = pickle.load(open("./model/churn-scaler.pickle", "rb"))
    loaded_model = loaded_dict['model']
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/')
def home():
    # Use the data from the pickle file in your home route
    return render_template('./views/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the data from the POST request.
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = 'The Customer is Churn'
        else:
            prediction = 'The Customer is No Churn'
        return render_template('./views/result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)