import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

data = pd.read_csv('data/new_loan.csv')

# Feature engineering and preprocessing
features = ['Age', 'Employment Status', 'Annual Income', 'Education Level', 'Credit Score', 'Credit Utilization Ratio', 
            'Delinquencies', 'Credit History Length', 'Loan Amount', 'Loan Purpose', 'Collateral Value', 
            'Monthly Expenses', 'Debt-to-Income Ratio', 'Identity Verification', 'Application Inconsistencies']
target = 'Fraud (Target)'

X = data[features]
y = data[target]

# Encoding categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X.loc[:, column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Scaling numerical variables
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Scale the entire dataframe (including new features)

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
fraud_model = RandomForestClassifier(random_state=42)
fraud_model.fit(X_train, y_train)

# Evaluating the model
y_pred = fraud_model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(fraud_model, 'models/fraud_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')  # Save the updated scaler
joblib.dump(label_encoders, 'models/label_encoders.pkl')
