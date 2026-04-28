import streamlit as st
import pickle

# ሞዴሉን እና ቬክተራይዘሩን መጫን
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found. Please run 'prepare_and_train.py' first!")

# --- UI SETTINGS ---
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

# --- SIDEBAR (Pro Look) ---
st.sidebar.title("About Project")
st.sidebar.info("""
**Project:** Sentiment Analysis
**Languages:** Amharic & English
**Models Used:** Logistic Regression
**Developer:** [MAU Group One Students]
""")
st.sidebar.image("confusion_matrix.png", caption="Model Performance")

# --- MAIN PAGE ---
st.title("Sentiment Analysis App 🤖")
st.write("Enter Amharic or English text to predict the sentiment (Positive/Negative).")

user_input = st.text_area("Enter your comment here:", placeholder="e.g. This is great or በጣም ጥሩ ነው")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Preprocess and Predict
        cleaned_input = user_input.lower()
        vec = vectorizer.transform([cleaned_input])
        prediction = model.predict(vec)[0]
        
        # Display Result
        if prediction == "positive":
            st.success(f"Result: {prediction.upper()} 😊")
        elif prediction == "negative":
            st.error(f"Result: {prediction.upper()} 😞")
        else:
            st.warning(f"Result: {prediction.upper()} 😐") # Neutral ሲሆን ቢጫ ቀለም ያሳያል

st.markdown("---")
st.caption("AI Model Trained with Scikit-learn | UI by Streamlit")
