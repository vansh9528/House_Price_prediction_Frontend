from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained regression model
with open('regmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        data = request.form
        square_footage = float(data['squareFootage'])
        bedrooms = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])
        lot_size = float(data['lotSize'])
        garage_size = float(data['garageSize'])
        year_built = int(data['yearBuilt'])
        neighborhood_quality = int(data['neighborhoodQuality'])

        # Prepare features for prediction
        features = np.array([
            square_footage, bedrooms, bathrooms, lot_size, garage_size, year_built, neighborhood_quality
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Return prediction as JSON
        return jsonify({'predicted_price': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
