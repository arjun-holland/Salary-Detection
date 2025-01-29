from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('salary.pkl', 'rb'))

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    # Get the input value from the form
    data1 = float(request.form['a'])  # Convert to float
    arr = np.array([[data1]])    # Reshape for model input
    pred = model.predict(arr)    # Make prediction
    pred_int = int(pred[0])       # Convert prediction to integer
    return render_template('after.html', data=pred_int)  # Pass integer prediction to template

if __name__ == "__main__":
    app.run(debug=True)