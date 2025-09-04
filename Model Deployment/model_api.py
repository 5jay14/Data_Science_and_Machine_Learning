from flask import Flask, jsonify, request
import joblib
import pandas as pd

# Create Flask APP
app = Flask(__name__)


# Connect the function to POST API call using decorator
@app.route('/predict', methods=['POST'])
# Connect POST API call ---> Predict Function
def predict():
    # Get JSON request
    feat_data = request.json

    # Convert JSON ==> PD DF
    df = pd.DataFrame(feat_data)
    df.reindex(columns=col_names)

    # Predict
    prediction = list(model.predict(df))

    # Return the prediction
    return jsonify({'prediction': str(prediction)})

# Load the model and column names

if __name__ == '__main__':
    col_names = joblib.load(r'C:\Users\vijay\Data Science and Machine Learning\Model Deplloyment\Col_names.pkl')
    model = joblib.load(r'C:\Users\vijay\Data Science and Machine Learning\Model Deplloyment\Final_model.pkl')
    app.run(debug=True)