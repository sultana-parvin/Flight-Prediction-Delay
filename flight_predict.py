import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
import requests
import base64

# Set page configuration
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

# Function to get base64 encoded image from URL
def get_base64_of_bin_file(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode()
    except Exception as e:
        st.warning(f"Failed to load image from URL: {e}")
        return None

# Function to set background image
def set_bg_hack(main_bg_url):
    bin_str = get_base64_of_bin_file(main_bg_url)
    if bin_str:
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
        st.markdown(page_bg_img, unsafe_allow_html=True)
    else:
        st.warning("Failed to set background image. Using default background.")

# Set background image from URL
background_url = "https://storage.googleapis.com/kaggle-datasets-images/1957837/3228623/8a6745b0b43eadd0e5a7806bb7ad2e38/dataset-cover.jpg?t=2022-02-25-18-08-22"
set_bg_hack(background_url)

# Custom CSS
custom_css = """
<style>
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
        color: #FFFFFF;
        text-shadow: 2px 2px 4px #000000;
    }
    .medium-font {
        font-size: 30px !important;
        font-weight: bold;
        color: #FFFFFF;
        text-shadow: 1px 1px 2px #000000;
    }
    .small-font {
        font-size: 20px !important;
        color: #FFFFFF;
        text-shadow: 1px 1px 2px #000000;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        border: 2px solid #4CAF50;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: white;
        color: black;
    }
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.8);
        color: black;
        border: 2px solid #4CAF50;
        border-radius: 5px;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Load data, model, and scaler
@st.cache_data
def load_data():
    df = pd.read_csv('Data/flight_filtered.csv')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('models/flight_delay_logistic_model.pkl')
    scaler = joblib.load('models/scaler.joblib')
    return model, scaler

logistic_model, scaler = load_model()

def get_real_time_data(api_key):
    url = f"http://api.aviationstack.com/v1/flights?access_key={api_key}"
    response = requests.get(url)
    return response.json()['data']

def preprocess_real_time_data(flight):
    features = {}
    features['AIRLINE'] = flight['airline']['name']
    features['ORIGIN'] = flight['departure']['iata']
    features['DEST'] = flight['arrival']['iata']

    dep_time = datetime.fromisoformat(flight['departure']['scheduled'])
    features['CRS_DEP_TIME_MINUTES'] = dep_time.hour * 60 + dep_time.minute

    arr_time = datetime.fromisoformat(flight['arrival']['scheduled'])
    features['CRS_ARR_TIME_MINUTES'] = arr_time.hour * 60 + arr_time.minute

    flight_date = datetime.fromisoformat(flight['flight_date'])
    features['FL_DAY_OF_WEEK'] = flight_date.weekday()
    features['FL_MONTH'] = flight_date.month
    features['FL_YEAR'] = flight_date.year

    return pd.DataFrame([features])

def engineer_features(df):
    df_encoded = pd.get_dummies(df, columns=['AIRLINE', 'ORIGIN', 'DEST'], drop_first=True)

    bool_col = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_col] = df_encoded[bool_col].astype(int)

    numeric_features = ['FL_DAY_OF_WEEK', 'FL_MONTH', 'FL_YEAR', 'CRS_DEP_TIME_MINUTES', 'CRS_ARR_TIME_MINUTES']

    df_encoded[numeric_features] = scaler.transform(df_encoded[numeric_features])

    return df_encoded

def predict_delay(flight_data):
    X = preprocess_real_time_data(flight_data)
    X_engineered = engineer_features(X)

    expected_columns = logistic_model.feature_names_in_
    missing_columns = set(expected_columns) - set(X_engineered.columns)

    missing_df = pd.DataFrame(0, index=X_engineered.index, columns=list(missing_columns))

    X_engineered = pd.concat([X_engineered, missing_df], axis=1)

    X_engineered = X_engineered.reindex(columns=expected_columns, fill_value=0)

    prediction = logistic_model.predict(X_engineered)
    probability = logistic_model.predict_proba(X_engineered)
    return prediction[0], probability[0][1]

# New function for user input processing
def process_user_input(user_input):
    features = {}
    features['AIRLINE'] = user_input['airline']
    features['ORIGIN'] = user_input['origin']
    features['DEST'] = user_input['destination']
    
    dep_time = datetime.strptime(user_input['departure_time'], '%Y-%m-%d %H:%M')
    features['CRS_DEP_TIME_MINUTES'] = dep_time.hour * 60 + dep_time.minute
    
    arr_time = datetime.strptime(user_input['arrival_time'], '%Y-%m-%d %H:%M')
    features['CRS_ARR_TIME_MINUTES'] = arr_time.hour * 60 + arr_time.minute
    
    features['FL_DAY_OF_WEEK'] = dep_time.weekday()
    features['FL_MONTH'] = dep_time.month
    features['FL_YEAR'] = dep_time.year
    
    return pd.DataFrame([features])

# Streamlit app
st.markdown('<p class="big-font">✈️ Flight Delay Prediction</p>', unsafe_allow_html=True)

# Create tabs for real-time and user input
tab1, tab2 = st.tabs(["Real-time Predictions", "User Input Prediction"])

with tab1:
    st.markdown('<p class="medium-font">🔑 Enter your AviationStack API key:</p>', unsafe_allow_html=True)
    api_key = st.text_input('', type='password', key='api_key')

    if api_key:
        if st.button('🚀 Fetch Real-time Data and Predict Delays'):
            try:
                real_time_data = get_real_time_data(api_key)

                st.markdown(f'<p class="medium-font">Retrieved {len(real_time_data)} flights.</p>', unsafe_allow_html=True)

                for flight in real_time_data:
                    try:
                        delay_prediction, delay_probability = predict_delay(flight)

                        st.markdown("---")
                        st.markdown(f'<p class="medium-font">✈️ Flight: {flight["flight"]["iata"] or "N/A"}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="small-font">🛩️ Airline: {flight["airline"]["name"] or "N/A"}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="small-font">📍 From: {flight["departure"]["airport"] or "N/A"} ({flight["departure"]["iata"] or "N/A"}) to {flight["arrival"]["airport"] or "N/A"} ({flight["arrival"]["iata"] or "N/A"})</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="small-font">🕒 Scheduled Departure: {flight["departure"]["scheduled"] or "N/A"}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="small-font">🕒 Scheduled Arrival: {flight["arrival"]["scheduled"] or "N/A"}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="medium-font">📊 Delay Prediction: {"🚨 Delayed" if delay_prediction == 0.5 else "✅ On Time"}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="medium-font">🔢 Delay Probability: {delay_probability * 100:.2f}%</p>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error processing flight: {str(e)}")

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

with tab2:
    st.markdown('<p class="medium-font">📝 Enter Flight Details:</p>', unsafe_allow_html=True)
    
    user_input = {}
    user_input['airline'] = st.text_input('Airline')
    user_input['origin'] = st.text_input('Origin Airport Code')
    user_input['destination'] = st.text_input('Destination Airport Code')
    user_input['departure_time'] = st.text_input('Departure Time (YYYY-MM-DD HH:MM)')
    user_input['arrival_time'] = st.text_input('Arrival Time (YYYY-MM-DD HH:MM)')

    if st.button('🔮 Predict Delay'):
        try:
            X = process_user_input(user_input)
            X_engineered = engineer_features(X)
            
            expected_columns = logistic_model.feature_names_in_
            missing_columns = set(expected_columns) - set(X_engineered.columns)
            missing_df = pd.DataFrame(0, index=X_engineered.index, columns=list(missing_columns))
            X_engineered = pd.concat([X_engineered, missing_df], axis=1)
            X_engineered = X_engineered.reindex(columns=expected_columns, fill_value=0)

            prediction = logistic_model.predict(X_engineered)
            probability = logistic_model.predict_proba(X_engineered)

            st.markdown("---")
            st.markdown(f'<p class="medium-font">✈️ Flight: {user_input["airline"]} from {user_input["origin"]} to {user_input["destination"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="small-font">🕒 Scheduled Departure: {user_input["departure_time"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="small-font">🕒 Scheduled Arrival: {user_input["arrival_time"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="medium-font">📊 Delay Prediction: {"🚨 Delayed" if prediction[0] == 0.5 else "✅ On Time"}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="medium-font">🔢 Delay Probability: {probability[0][1]* 100:.2f}%</p>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing input: {str(e)}")

st.markdown('<p class="small-font">Note: This app uses both real-time flight data and user input for predictions. The model is based on historical data and may not reflect all current factors affecting flight delays.</p>', unsafe_allow_html=True)