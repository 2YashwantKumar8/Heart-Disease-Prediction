from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('xgboost_heart_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features in the same order as training
        features = [float(request.form[key]) for key in request.form]
        data = np.array([features])

        # Make prediction
        prediction = model.predict(data)[0]

        # Convert numeric prediction to label
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return render_template('result.html', prediction=result)

    except Exception as e:
        # Send error message to template
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    # debug=True -> auto-reload, detailed error pages
    app.run(debug=True)
