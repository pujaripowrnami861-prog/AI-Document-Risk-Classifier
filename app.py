import streamlit as st
from pypdf import PdfReader
import re
import joblib

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Document Risk Classifier",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("models/risk_classifier.pkl")

# =========================================================
# COLORS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #020617,
        #071a52,
        #24104f
    );
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #020617,
        #071a52,
        #24104f
    );
}

h1 {
    color: #f9a8d4 !important;
}

h2, h3 {
    color: #c4b5fd !important;
}

.stButton button {
    background: linear-gradient(
        90deg,
        #6366f1,
        #ec4899
    );
    color: white;
    border: none;
    border-radius: 12px;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #071a52,
        #26104f
    );
    border: 1px solid #6366f1;
    border-radius: 16px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 AI Assistant")

    st.info(
        "Upload a document and let the AI model "
        "analyze its risk category."
    )

    st.subheader("🧠 Machine Learning")

    st.write("🔹 TF-IDF")
    st.write("🔹 Logistic Regression")
    st.write("🔹 Confidence Analysis")

    st.divider()

    st.subheader("🎨 Risk Categories")

    st.success("🟢 Low Risk")
    st.warning("🟡 Medium Risk")
    st.error("🔴 High Risk")

    st.divider()

    st.caption(
        "AI Document Risk & Compliance Classifier"
    )

# =========================================================
# MAIN TITLE
# =========================================================

st.title("🤖 AI Document Risk Classifier")

st.subheader(
    "Smart document analysis powered by Machine Learning"
)

st.write(
    "Detect Risk • Analyze Content • Support Compliance Review"
)

st.info(
    "✨ Intelligent • Secure • AI Powered"
)

st.divider()

# =========================================================
# UPLOAD
# =========================================================

st.header("📤 Upload Your Document")

st.write(
    "Upload a PDF or TXT document for AI-powered risk analysis."
)

uploaded_file = st.file_uploader(
    "Choose your document",
    type=["pdf", "txt"]
)

# =========================================================
# DOCUMENT PROCESSING
# =========================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded successfully: {uploaded_file.name}"
    )

    document_text = ""

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    if uploaded_file.name.lower().endswith(".pdf"):

        try:

            reader = PdfReader(uploaded_file)

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    document_text += text + "\n"

        except Exception as e:

            st.error(
                f"Unable to read PDF: {e}"
            )

    # -----------------------------------------------------
    # TXT
    # -----------------------------------------------------

    else:

        try:

            document_text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        except Exception as e:

            st.error(
                f"Unable to read text file: {e}"
            )

    # =====================================================
    # CHECK DOCUMENT
    # =====================================================

    if not document_text.strip():

        st.warning(
            "⚠️ No readable text was found in this document."
        )

    else:

        # =================================================
        # DOCUMENT PREVIEW
        # =================================================

        st.header("📄 Document Preview")

        st.text_area(
            "Extracted Text",
            document_text,
            height=220
        )

        # =================================================
        # CLEAN TEXT
        # =================================================

        cleaned_text = document_text.lower()

        cleaned_text = re.sub(
            r"[^a-zA-Z0-9\s]",
            " ",
            cleaned_text
        )

        cleaned_text = re.sub(
            r"\s+",
            " ",
            cleaned_text
        ).strip()

        # =================================================
        # AI PREDICTION
        # =================================================

        try:

            prediction = model.predict(
                [cleaned_text]
            )[0]

            probabilities = model.predict_proba(
                [cleaned_text]
            )[0]

        except Exception as e:

            st.error(
                f"Model prediction failed: {e}"
            )

            st.stop()

        # =================================================
        # CONFIDENCE
        # =================================================

        confidence = max(probabilities) * 100

        prediction_text = str(prediction)

        # =================================================
        # REVIEW STATUS
        # =================================================

        if confidence < 70:

            review_status = "Manual Review"

        else:

            review_status = "Accepted"

        # =================================================
        # AI MESSAGE
        # =================================================

        if prediction_text.lower() == "high":

            risk_icon = "🔴"

            message = (
                "High-risk information may be present."
            )

        elif prediction_text.lower() == "medium":

            risk_icon = "🟡"

            message = (
                "Moderately sensitive information "
                "may be present."
            )

        else:

            risk_icon = "🟢"

            message = (
                "The document appears to contain "
                "mostly general information."
            )

        # =================================================
        # RISK ANALYSIS
        # =================================================

        st.header("🤖 AI Risk Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🛡️ Risk Level",
                prediction_text
            )

        with col2:

            st.metric(
                "🎯 Confidence",
                f"{confidence:.2f}%"
            )

        with col3:

            st.metric(
                "⚠️ Review Status",
                review_status
            )

        st.info(
            f"🤖 AI Assistant: {message}"
        )

        # =================================================
        # CONFIDENCE
        # =================================================

        st.header("🎯 Prediction Confidence")

        st.progress(
            min(int(confidence), 100)
        )

        st.write(
            f"Model confidence: {confidence:.2f}%"
        )

        if confidence < 70:

            st.warning(
                "⚠️ Manual Review Required"
            )

        else:

            st.success(
                "✅ AI Prediction Accepted"
            )

        # =================================================
        # ANALYTICS
        # =================================================

        st.header("📊 Document Analytics")

        word_count = len(
            cleaned_text.split()
        )

        character_count = len(
            cleaned_text
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "📝 Words",
                word_count
            )

        with col2:

            st.metric(
                "🔤 Characters",
                character_count
            )

        with col3:

            st.metric(
                "🛡️ Risk",
                prediction_text
            )

        # =================================================
        # PROBABILITY
        # =================================================

        st.header("📈 Risk Probability Distribution")

        risk_names = list(
            model.classes_
        )

        probability_data = {}

        for name, probability in zip(
            risk_names,
            probabilities
        ):

            probability_data[name] = [
                probability * 100
            ]

        st.bar_chart(
            probability_data
        )

        # =================================================
        # ANALYSIS SUMMARY
        # =================================================

        st.header("📝 Analysis Summary")

        st.write(
            f"📄 **Document:** {uploaded_file.name}"
        )

        st.write(
            f"🛡️ **Predicted Risk:** "
            f"{risk_icon} {prediction_text}"
        )

        st.write(
            f"🎯 **Confidence:** "
            f"{confidence:.2f}%"
        )

        st.write(
            f"📝 **Words Analyzed:** "
            f"{word_count}"
        )

        st.write(
            f"⚠️ **Review Status:** "
            f"{review_status}"
        )

        # =================================================
        # REPORT
        # =================================================

        st.header("📥 Download Analysis Report")

        probability_dict = dict(
            zip(
                risk_names,
                probabilities
            )
        )

        low_probability = (
            probability_dict.get("Low", 0) * 100
        )

        medium_probability = (
            probability_dict.get("Medium", 0) * 100
        )

        high_probability = (
            probability_dict.get("High", 0) * 100
        )

        report = f"""
AI DOCUMENT RISK & COMPLIANCE CLASSIFIER
=========================================

DOCUMENT
--------
File Name: {uploaded_file.name}
Words Analyzed: {word_count}
Characters: {character_count}

AI RISK ANALYSIS
----------------
Predicted Risk: {prediction_text}
Confidence: {confidence:.2f}%

RISK PROBABILITIES
------------------
Low Risk: {low_probability:.2f}%
Medium Risk: {medium_probability:.2f}%
High Risk: {high_probability:.2f}%

REVIEW STATUS
-------------
{review_status}

TECHNOLOGY
----------
Python
Streamlit
TF-IDF
Logistic Regression
Machine Learning
"""

        st.download_button(
            "📥 Download AI Analysis Report",
            report,
            file_name="AI_Document_Risk_Analysis.txt",
            mime="text/plain"
        )

# =========================================================
# NO FILE MESSAGE
# =========================================================

else:

    st.info(
        "🤖 Upload a PDF or TXT document above "
        "to start AI risk analysis."
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "💜 AI Document Risk & Compliance Classifier "
    "• Machine Learning Project • Python + Streamlit"
)