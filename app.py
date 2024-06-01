from flask import Flask, request, render_template
import pandas as pd
import pickle
import openai

app = Flask(__name__)

# Loading trained models
with open('credit_worthiness_model.pkl', 'rb') as f:
    credit_model = pickle.load(f)

with open('fraud_detect_ion_model.pkl', 'rb') as f:
    fraud_model = pickle.load(f)

openai.api_key = "sk-proj-6gNUSv7V5wnKEZYg9xc0T3BlbkFJkZABRj2L65yhKnmXqyth"  # Keep this confidential

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Salary': [float(request.form['Salary'])],
        'Assets': [float(request.form['Assets'])],
        'ExistingLoans': [int(request.form['ExistingLoans'])],
        'CreditScore': [float(request.form['CreditScore'])],
        'NumAccounts': [int(request.form['NumAccounts'])],
        'PropertyOwnership': [int(request.form['PropertyOwnership'])],
        'VoterRegistration': [int(request.form['VoterRegistration'])],
        'TotalIncome': [float(request.form['TotalIncome'])],
        'Fraud': [0],  
        'Creditworthy': [0] 
    }

    df = pd.DataFrame(data)

    # Predicting using loaded models
    creditworthy = credit_model.predict(df)[0]
    fraud = fraud_model.predict(df)[0]

    # Generating explanation using GPT-3.5 Turbo
    explanation = generate_gpt3_explanation(data, creditworthy, fraud)

    return render_template('results.html', creditworthy=creditworthy, fraud=fraud, explanation=explanation)


def generate_gpt3_explanation(data, creditworthy, fraud):
    prompt = f"""
    Based on the following financial data:
    - Salary: {data['Salary'][0]}
    - Assets: {data['Assets'][0]}
    - Existing Loans: {data['ExistingLoans'][0]}
    - Credit Score: {data['CreditScore'][0]}
    - Number of Accounts: {data['NumAccounts'][0]}
    - Property Ownership: {'Yes' if data['PropertyOwnership'][0] == 1 else 'No'}
    - Voter Registration: {'Yes' if data['VoterRegistration'][0] == 1 else 'No'}
    - Total Income: {data['TotalIncome'][0]}
    
    The creditworthiness score is: {'Creditworthy' if creditworthy else 'Not Creditworthy'}
    The fraud detection result is: {'Fraudulent' if fraud else 'Not Fraudulent'}
    
    Provide a detailed explanation for these results.
    """

    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150
    )

    explanation = response['choices'][0]['text'].strip()
    return explanation


if __name__ == '__main__':
    app.run(debug=True)
