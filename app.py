import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load Model
model = pickle.load(open("fraud_model.pkl", "rb"))

# Page Config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Sidebar
st.sidebar.title("📌 Project Information")
st.sidebar.success("Random Forest Classifier")
st.sidebar.write("Accuracy: 99.88%")
st.sidebar.write("Recall: 88%")
st.sidebar.write("ROC-AUC: 98.5%")

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload transaction data to detect fraudulent transactions."
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### 👨‍💻 Developer

    **S.dev Singh**

    *Credit Card Fraud Detection System*
    """
)

# Header
st.markdown(
    """
    <h1 style='text-align:center;color:#1E88E5;'>
    💳 Credit Card Fraud Detection Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='text-align:center;'>
    Machine Learning Powered Fraud Analysis
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align:center;'>
    👨‍💻 Developed by <b>S.dev Singh</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Upload Section
uploaded_file = st.file_uploader(
    "📂 Upload Transaction CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head())

    if st.button("🚀 Predict Fraud"):

        with st.spinner("Analyzing Transactions..."):

            predictions = model.predict(df)

        df["Prediction"] = predictions

        fraud_count = (predictions == 1).sum()
        genuine_count = (predictions == 0).sum()

        fraud_percentage = round(
            (fraud_count / len(df)) * 100,
            2
        )

        st.success("✅ Prediction Complete")

        st.subheader("📊 Transaction Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Transactions", len(df))
        col2.metric("Fraud Transactions", fraud_count)
        col3.metric("Genuine Transactions", genuine_count)
        col4.metric("Fraud %", f"{fraud_percentage}%")

        # Pie Chart
        st.subheader("📈 Fraud Distribution")

        left, right = st.columns([2, 1])

        with right:

            fig, ax = plt.subplots(figsize=(3, 3))

            ax.pie(
                [genuine_count, fraud_count],
                labels=["Genuine", "Fraud"],
                autopct="%1.1f%%"
            )

            st.pyplot(fig)
            plt.close(fig)

        # Fraud Transactions
        st.subheader("🚨 Detected Fraud Transactions")

        frauds = df[df["Prediction"] == 1]

        if len(frauds) > 0:
            st.dataframe(frauds)
        else:
            st.success("No Fraud Transactions Found")

        # Complete Results
        st.subheader("📋 Complete Prediction Results")
        st.dataframe(df)

        # Download Button
        csv = df.to_csv(index=False)

        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;'>
        <p>Credit Card Fraud Detection System</p>
        <p>Developed using Streamlit, Python, and Machine Learning</p>
        <p>👨‍💻 Developed by <b>S.dev Singh</b></p>
    </div>
    """,
    unsafe_allow_html=True
)