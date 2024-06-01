from tempfile import TemporaryFile
from io import BytesIO

from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
from fpdf import FPDF
import openai

app = Flask(__name__)

fraud_model = joblib.load('models/fraud_model.pkl')
scaler = joblib.load('models/scaler.pkl')  # Updated scaler with new features
label_encoders = joblib.load('models/label_encoders.pkl')

creditworthiness_model = joblib.load('models/creditworthiness_model.pkl')
creditworthiness_scaler = joblib.load('models/creditworthiness_scaler.pkl')  # Updated scaler with new features
creditworthiness_label_encoders = joblib.load('models/creditworthiness_label_encoders.pkl')

openai.api_key = 'sk-proj-6gNUSvkZABRj2L65yhKnmXqyth'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    form_data = request.form

    input_data = pd.DataFrame([form_data])

    # Preprocessing the input data
    for column, le in label_encoders.items():
        input_data[column] = le.transform(input_data[column])
    input_data = scaler.transform(input_data)  

    # Predicting the fraud
    fraud_prediction = fraud_model.predict(input_data)[0]

    # Generates reports (combined section)
    if fraud_prediction == 1:
        fraud_features = form_data.to_dict()
        fraud_features['Fraud'] = 'High Risk'

    # Predicting creditworthiness
    creditworthiness_prediction = creditworthiness_model.predict(input_data)[0]
    credit_score_category = categorize_credit_score(form_data['Credit Score'])

    # Generating Creditworthiness Report
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=f"Generate a report for a loan application with a credit score category of {credit_score_category}.",
        max_tokens=100
    )
    gpt_response = response.choices[0].text.strip()

    legit_features = form_data.to_dict()
    legit_features['Creditworthiness'] = gpt_response
    report_text = f"**Creditworthiness Assessment:** {legit_features['Creditworthiness']}"

    return render_template('result.html', report_text=report_text)

#Below can be intialized to directly download pdf generated report

# def generate_pdf(data, is_fraud, output_stream):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)

#     # Header with bold formatting (workaround)
#     pdf.cell(200, 10, txt=f"Loan Application Report", ln=True, align='C')
#     pdf.set_text_color(255, 0, 0)  # Set red color for bold effect
#     pdf.write(5, txt="**")  # Write double asterisks for visual bold representation
#     pdf.set_text_color(0, 0, 0)  # Reset text color to black
    

#     # Applicant Information
#     pdf.cell(100, 10, txt=f"Applicant Name:", ln=False)
#     applicant_name = data.get('Applicant Name', 'abc')  # Handle missing key with default value
#     pdf.cell(100, 10, txt=f"{applicant_name}", ln=True)

#     pdf.cell(100, 10, txt=f"Email:", ln=False)
#     pdf.cell(100, 10, txt=f"{data.get('Email', 'gmail')}", ln=True)  # Handle missing key with default value

#     # Loan Details (example)
#     pdf.cell(100, 10, txt=f"Loan Amount:", ln=False)
#     pdf.cell(100, 10, txt=f"{data.get('Loan Amount', '202')}", ln=True)  # Handle missing key with default value
#     pdf.cell(100, 10, txt=f"Loan Purpose:", ln=False)
#     pdf.cell(100, 10, txt=f"{data.get('Loan Purpose', 'just')}", ln=True)  # Handle missing key with default value

#     # Fraud Report Content (if applicable)
#     if is_fraud:
#         pdf.cell(200, 10, txt=f"Fraud Prediction: {data['Fraud']}", ln=True)
#         # Add more details about fraud indicators here (replace with your logic)
#         pdf.cell(200, 10, txt=f"Suspicious activity detected in loan application", ln=True)

#     # Creditworthiness Report Content (if applicable)
#     else:
#         pdf.cell(200, 10, txt=f"Creditworthiness Assessment: {data['Creditworthiness']}", ln=True)
#         # Add more details about creditworthiness analysis here (replace with your logic)
#         pdf.cell(200, 10, txt=f"Based on your credit score and financial history, you seem to be a creditworthy borrower.")

#     # Write PDF content directly to the temporary file in memory (using recommended solution)
#         pdf.set_encoding('UTF-8')
#     with output_stream as temp_file:
#         pdf.output(output_stream)

def categorize_credit_score(credit_score):
    credit_score = int(credit_score)
    if 300 <= credit_score <= 499:
        return 'Poor'    
    elif 500 <= credit_score <= 649:
        return 'Average'
    elif 650 <= credit_score <= 749:
        return 'Good'
    elif 750 <= credit_score <= 900:
        return 'Excellent'
    else:
        return 'Unknown'

if __name__ == '__main__':
    app.run(debug=True)
