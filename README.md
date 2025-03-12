![prototype](https://github.com/user-attachments/assets/7edb39a8-f576-4f4e-85b1-7de803a7de3f)



https://github.com/krakken190/AI-Loan-Underswriting-Adviser/assets/72223902/bdff8b32-ef0c-41d8-a7c3-ca49f96082ff




# AI-Loan-Underswriting-Adviser

## Introduction
This document provides clear step-by-step instructions on building, deploying, and using the AI Loan Underwriting Advisor application. This application leverages machine learning to predict loan approvals and detect potential fraud based on user input.

## Prerequisites
Python 3.x
Flask
Scikit-learn
Pandas

## Project Structure
app.py: Main Flask application file.
templates/: Contains HTML templates (index.html, result.html).
static/: Contains the CSS file (style.css).
model/: Contains the pre-trained machine learning model.
ML File/: Contains python code to train models.
requirements.txt: List of required Python packages.

## Step-by-Step Instructions
1. Clone the Repository
Clone the repository from GitHub or download the source files directly.

git clone <repository-url>
cd AI-Loan-Underwriting-Advisor

2. Set Up Virtual Environment
Create and activate a virtual environment.

python -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`

## 3. Install Dependencies

Install the required Python packages.
pip install -r requirements.txt

## 4. Train the Machine Learning Model (if not using pre-trained model)

Generate synthetic data and train the model. This step assumes you have a script (train_model.py) to handle this.
python train_model.py
Save the trained model in the model/ directory.

## 5. Run the Flask Application
Start the Flask application.
python app.py
The application should now be running on your chrome(Local Host)

## 6. Using the Application

Fill out the loan application form with the required details.
Submit the form to get the loan approval result and fraud detection analysis.
