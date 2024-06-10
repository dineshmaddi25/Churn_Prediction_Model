import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import streamlit as st

# Load the dataset
data = pd.read_csv('telco_churn_cleaned.csv')

# Preprocessing
X = data.drop('Churn', axis=1)
y = data['Churn']
X = pd.get_dummies(X, drop_first=True)
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {name}: {accuracy:.2f}")
    print(f"Classification Report of {name}:\n{classification_report(y_test, y_pred)}")

    # Save the trained model
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")

# Streamlit UI
st.title('Telco Customer Churn Prediction')

gender = st.radio("Select gender", ('Male', 'Female'))
SeniorCitizen = st.radio("Is Senior Citizen?", ('Yes', 'No'))
Partner = st.radio("Has Partner?", ('Yes', 'No'))
Dependents = st.radio("Has Dependents?", ('Yes', 'No'))
tenure = st.slider('Tenure (months)', min_value=0, max_value=72)
MonthlyCharges = st.slider('Monthly Charges ($)', min_value=0.0, max_value=120.0)
TotalCharges = st.slider('Total Charges ($)', min_value=0.0, max_value=9000.0)
InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic'])
Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaymentMethod = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])


input_data = {
    'tenure': tenure,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
    'InternetService_DSL': 1 if InternetService == 'DSL' else 0,
    'Contract_One year': 1 if Contract == 'One year' else 0,
    'Contract_Two year': 1 if Contract == 'Two year' else 0,
    'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0,
    'PaymentMethod_Bank transfer (automatic)': 1 if PaymentMethod == 'Bank transfer (automatic)' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0,
    'gender_Female': 1 if gender == 'Female' else 0,
    'SeniorCitizen_Yes': 1 if SeniorCitizen == 'Yes' else 0,  
    'Partner_Yes': 1 if Partner == 'Yes' else 0,
    'Dependents_Yes': 1 if Dependents == 'Yes' else 0,
    'Dependents_No': 1 if Dependents == 'No' else 0,  
    
}
input_df = pd.DataFrame([input_data])


def predict_churn(model, data):
    prediction = model.predict(data)
    return prediction
if st.button('Predict with Logistic Regression'):
    logreg_model = joblib.load('logistic_regression_model.pkl')
    prediction = predict_churn(logreg_model, input_df)
    if prediction[0] == 1:
        st.error("Churn prediction: Yes")
    else:
        st.success("Churn prediction: No")

if st.button('Predict with Random Forest'):
    rf_model = joblib.load('random_forest_model.pkl')
    prediction = predict_churn(rf_model, input_df)
    if prediction[0] == 1:
        st.error("Churn prediction: Yes")
    else:
        st.success("Churn prediction: No")

if st.button('Predict with Gradient Boosting'):
    gb_model = joblib.load('gradient_boosting_model.pkl')
    prediction = predict_churn(gb_model, input_df)
    if prediction[0] == 1:
        st.error("Churn prediction: Yes")
    else:
        st.success("Churn prediction: No")