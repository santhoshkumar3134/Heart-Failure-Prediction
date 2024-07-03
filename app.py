from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    output = int(prediction[0])
    
    return render_template('index.html', prediction_text='Dead Event Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
