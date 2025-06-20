import pickle
import requests
from flask import Flask, request, render_template

API_KEY = 'your_ibm_cloud_api_key'  # Replace with your actual IBM Cloud API key

# Generate access token for IBM ML API
token_response = requests.post(
    'https://iam.cloud.ibm.com/identity/token',
    data={
        "apikey": API_KEY,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }
)

mltoken = token_response.json()["access_token"]
header = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + mltoken
}

# Initialize Flask app
app = Flask(__name__)

# Load your trained ML model
model = pickle.load(open('CKD.pkl', 'rb'))

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('indexnew.html')

@app.route('/Home', methods=['GET', 'POST'])
def my_home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input values from form
    blood_urea = request.form['blood_urea']
    blood_glucose_random = request.form['blood glucose random']
    coronary_artery_disease = request.form['coronary_artery_disease']
    anemia = request.form['anemia']
    pus_cell = request.form['pus_cell']
    red_blood_cells = request.form['red_blood_cells']
    diabetesmellitus = request.form['diabetesmellitus']
    pedal_edema = request.form['pedal_edema']

    # Convert categorical variables to numeric
    c1 = 1 if coronary_artery_disease == 'Yes' else 0
    a1 = 1 if anemia == 'Yes' else 0
    p1 = 1 if pus_cell == 'Normal' else 0
    r1 = 1 if red_blood_cells == 'Normal' else 0
    d1 = 1 if diabetesmellitus == 'Yes' else 0
    pe = 1 if pedal_edema == 'Yes' else 0

    # Prepare input for model
    input_features = [[
        int(blood_urea),
        int(blood_glucose_random),
        c1, a1, p1, r1, d1, pe
    ]]
    print("Input features:", input_features)

    # Scoring payload for IBM Cloud (optional if you're using local model)
    payload_scoring = {
        "input_data": [{
            "fields": [
                "blood_urea",
                "blood glucose random",
                "coronary_artery_disease",
                "anemia",
                "pus_cell",
                "red blood cells",
                "diabetesmellitus",
                "pedal_edema"
            ],
            "values": input_features
        }]
    }

    # Uncomment below if using IBM Watson endpoint for prediction:
    # response = requests.post(
    #     'your_ibm_cloud_model_endpoint_url',
    #     json=payload_scoring,
    #     headers=header
    # )
    # pred = response.json()['predictions'][0]['values'][0][0]

    # Use local model for prediction
    pred = model.predict(input_features)[0]

    return render_template('result.html', pred=pred)

if __name__ == '__main__':
    app.run(debug=True)