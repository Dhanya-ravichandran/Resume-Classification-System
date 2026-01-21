# import os
# import sys
# import pickle
# import numpy as np
# import pandas as pd
# import streamlit as st

# # ==================================================
# # PATH SETUP
# # ==================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
# sys.path.append(PROJECT_ROOT)

# from src.data_loader import extract_text
# from src.nlp.text_cleaning import clean_text

# # ==================================================
# # LOAD MODEL & VECTORIZER
# # ==================================================
# with open(os.path.join(PROJECT_ROOT, "models", "SVM.pkl"), "rb") as f:
#     model = pickle.load(f)

# with open(os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.pkl"), "rb") as f:
#     vectorizer = pickle.load(f)

# # ==================================================
# # STREAMLIT CONFIG
# # ==================================================
# st.set_page_config(
#     page_title="Resume Classification System",
#     page_icon="📄",
#     layout="wide"
# )

# # ==================================================
# # SESSION STATE
# # ==================================================
# if "show_result" not in st.session_state:
#     st.session_state.show_result = False

# # ==================================================
# # CUSTOM STYLES
# # ==================================================
# st.markdown("""
# <style>
# .main-title {
#     font-size: 42px;
#     font-weight: 700;
#     text-align: center;
#     color: #1f2937;
# }
# .sub-title {
#     font-size: 18px;
#     text-align: center;
#     color: #4b5563;
# }
# .card {
#     padding: 20px;
#     border-radius: 12px;
#     background-color: #f9fafb;
#     box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
# }
# </style>
# """, unsafe_allow_html=True)

# # ==================================================
# # SIDEBAR
# # ==================================================
# st.sidebar.title("Resume Classifier")

# st.sidebar.markdown("""
# ### Purpose
# Automatically identifies the **best-fit job role** for a resume using AI.

# ### 👥 Who is this for?
# - Recruiters  
# - HR Teams  
# - Hiring Managers  

# ### How it works
# - Resume text is analyzed using NLP  
# - Skills & keywords are extracted  
# - AI model predicts the closest role  

# ### 📌 Supported Roles
# - PeopleSoft  
# - SQL Developer  
# - Workday  
# - React Developer  
# - ReactJS Developer  
# - Internship  
# - Others  

# ### ⚠️ Disclaimer
# This system supports hiring decisions.  
# Final evaluation should include human review.
# """)

# # ==================================================
# # HEADER
# # ==================================================
# st.markdown('<div class="main-title">📄 Resume Classification Dashboard</div>', unsafe_allow_html=True)
# st.markdown('<div class="sub-title">AI-powered resume screening for smarter hiring</div>', unsafe_allow_html=True)
# st.markdown("<hr>", unsafe_allow_html=True)

# # ==================================================
# # UPLOAD PAGE
# # ==================================================
# if not st.session_state.show_result:

#     st.subheader("📤 Upload Resume")

#     uploaded_file = st.file_uploader(
#         "Upload a resume file (PDF or DOCX)",
#         type=["pdf", "docx"]
#     )

#     st.info("Your resume is processed securely and is not stored.")

#     if uploaded_file:

#         if uploaded_file.size > 5 * 1024 * 1024:
#             st.error("❌ File size exceeds 5MB limit.")
#             st.stop()

#         with st.spinner("Analyzing resume..."):

#             temp_path = os.path.join(BASE_DIR, uploaded_file.name)
#             with open(temp_path, "wb") as f:
#                 f.write(uploaded_file.read())

#             raw_text = extract_text(temp_path)
#             os.remove(temp_path)

#             if not raw_text.strip():
#                 st.error("❌ Unable to extract text from the resume.")
#             else:
#                 cleaned_text = clean_text(raw_text)

#                 # Vectorize input
#                 X_input = vectorizer.transform([cleaned_text])

#                 # ===============================
#                 # CORRECT CONFIDENCE CALCULATION
#                 # ===============================
#                 decision_scores = model.decision_function(X_input)

#                 # Absolute margin from hyperplane
#                 max_score = np.max(np.abs(decision_scores))

#                 # Normalize to 0–100
#                 confidence = min(100, (max_score / 10) * 100)
#                 confidence = round(confidence, 2)

#                 # Predict class
#                 prediction = model.predict(X_input)[0]

#                 # Threshold-based rejection
#                 CONFIDENCE_THRESHOLD = 60
#                 if confidence < CONFIDENCE_THRESHOLD:
#                     prediction = "Others"

#                 # Save to session
#                 st.session_state.prediction = prediction
#                 st.session_state.confidence = confidence
#                 st.session_state.raw_text = raw_text
#                 st.session_state.cleaned_text = cleaned_text
#                 st.session_state.show_result = True

#                 st.rerun()

# # ==================================================
# # RESULT PAGE
# # ==================================================
# else:
#     st.success("✅ Resume Analysis Complete")

#     col1, col2 = st.columns([2, 1])

#     with col1:
#         st.markdown(f"""
#         <div class="card">
#             <h3>Predicted Job Category</h3>
#             <h2 style="color:#2563eb;">{st.session_state.prediction}</h2>
#         </div>
#         """, unsafe_allow_html=True)

#         st.metric(
#             label="Prediction Confidence",
#             value=f"{st.session_state.confidence} %"
#         )

#         st.info("""
#         **Confidence Explanation**  
#         This score reflects how strongly the resume content matches the predicted role
#         based on skills, experience, and keywords.
#         """)

#         st.subheader(" Why this result?")
#         feature_names = vectorizer.get_feature_names_out()
#         X_vec = vectorizer.transform([st.session_state.cleaned_text]).toarray()[0]
#         top_idx = np.argsort(X_vec)[-10:][::-1]
#         keywords = [feature_names[i] for i in top_idx]

#         st.write("Key skills and terms identified:")
#         st.write(", ".join(keywords))

#     with col2:
#         st.subheader(" Resume Insights")
#         insights = pd.DataFrame({
#             "Metric": ["Total Characters", "Total Words"],
#             "Value": [
#                 len(st.session_state.raw_text),
#                 len(st.session_state.raw_text.split())
#             ]
#         })
#         st.table(insights)

#     result_text = f"""
# Resume Classification Result

# Predicted Category: {st.session_state.prediction}
# Confidence Score: {st.session_state.confidence} %

# Generated using an AI-based resume classification system.
# """

#     st.download_button(
#         "📥 Download Result Summary",
#         data=result_text,
#         file_name="resume_classification_result.txt"
#     )

#     if st.button("🔄 Analyze Another Resume"):
#         st.session_state.show_result = False
#         st.rerun()

# # ==================================================
# # FOOTER
# # ==================================================
# st.markdown("""
# <hr>
# <p style="text-align:center; font-size:13px;">
# AI-assisted resume classification | Built for real-world recruitment workflows
# </p>
# """, unsafe_allow_html=True)





import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ==================================================
# PATH SETUP
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.data_loader import extract_text
from src.nlp.text_cleaning import clean_text

# ==================================================
# LOAD MODEL & VECTORIZER
# ==================================================
with open(os.path.join(PROJECT_ROOT, "models", "SVM.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(
    page_title="Resume Classification System",
    page_icon="📄",
    layout="wide"
)

# ==================================================
# SESSION STATE
# ==================================================
if "show_result" not in st.session_state:
    st.session_state.show_result = False

# ==================================================
# CUSTOM STYLES
# ==================================================
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    color: #1f2937;
}
.sub-title {
    font-size: 18px;
    text-align: center;
    color: #4b5563;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f9fafb;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("Resume Classifier")

st.sidebar.markdown("""
### Purpose
Automatically identifies the **best-fit job role** for a resume using AI.

### 👥 Who is this for?
- Recruiters  
- HR Teams  
- Hiring Managers  

### How it works
- Resume text is analyzed using NLP  
- Skills & keywords are extracted  
- AI model predicts the closest role  

### 📌 Supported Roles
- PeopleSoft  
- SQL Developer  
- Workday  
- React Developer  
- ReactJS Developer  
- Internship  
- Others  

### ⚠️ Disclaimer
This system supports hiring decisions.  
Final evaluation should include human review.
""")

# ==================================================
# HEADER
# ==================================================
st.markdown('<div class="main-title">📄 Resume Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered resume screening for smarter hiring</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ==================================================
# UPLOAD PAGE
# ==================================================
if not st.session_state.show_result:

    st.subheader("📤 Upload Resume")

    uploaded_file = st.file_uploader(
        "Upload a resume file (PDF or DOCX)",
        type=["pdf", "docx"]
    )

    st.info("Your resume is processed securely and is not stored.")

    if uploaded_file:

        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("❌ File size exceeds 5MB limit.")
            st.stop()

        with st.spinner("Analyzing resume..."):

            temp_path = os.path.join(BASE_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            raw_text = extract_text(temp_path)
            os.remove(temp_path)

            if not raw_text.strip():
                st.error("❌ Unable to extract text from the resume.")
            else:
                cleaned_text = clean_text(raw_text)
                X_input = vectorizer.transform([cleaned_text])

                # ==================================================
                # ✅ CORRECT SVM CONFIDENCE (GAP-BASED)
                # ==================================================
                decision_scores = model.decision_function(X_input)[0]

                # Sort scores
                sorted_scores = np.sort(decision_scores)

                # Confidence = gap between top 2 classes
                confidence_gap = sorted_scores[-1] - sorted_scores[-2]

                # Convert to readable %
                confidence = min(100, confidence_gap * 100)
                confidence = round(confidence, 2)

                prediction = model.predict(X_input)[0]

                # Reject uncertain predictions
                CONFIDENCE_THRESHOLD = 15  # realistic for SVM

                if confidence < CONFIDENCE_THRESHOLD:
                    prediction = "Others"

                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                st.session_state.raw_text = raw_text
                st.session_state.cleaned_text = cleaned_text
                st.session_state.show_result = True

                st.rerun()

# ==================================================
# RESULT PAGE
# ==================================================
else:
    st.success("✅ Resume Analysis Complete")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        <div class="card">
            <h3>Predicted Job Category</h3>
            <h2 style="color:#2563eb;">{st.session_state.prediction}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.metric(
            label="Prediction Confidence",
            value=f"{st.session_state.confidence} %"
        )

        st.info("""
        **Confidence Explanation**  
        Confidence is computed using the difference between the top two
        predicted class scores. Larger gaps indicate stronger certainty.
        """)

        st.subheader("Why this result?")
        feature_names = vectorizer.get_feature_names_out()
        X_vec = vectorizer.transform([st.session_state.cleaned_text]).toarray()[0]
        top_idx = np.argsort(X_vec)[-10:][::-1]
        keywords = [feature_names[i] for i in top_idx]

        st.write("Key skills and terms identified:")
        st.write(", ".join(keywords))

    with col2:
        st.subheader("Resume Insights")
        insights = pd.DataFrame({
            "Metric": ["Total Characters", "Total Words"],
            "Value": [
                len(st.session_state.raw_text),
                len(st.session_state.raw_text.split())
            ]
        })
        st.table(insights)

    result_text = f"""
Resume Classification Result

Predicted Category: {st.session_state.prediction}
Confidence Score: {st.session_state.confidence} %

Generated using an AI-based resume classification system.
"""

    st.download_button(
        "📥 Download Result Summary",
        data=result_text,
        file_name="resume_classification_result.txt"
    )

    if st.button("🔄 Analyze Another Resume"):
        st.session_state.show_result = False
        st.rerun()

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<hr>
<p style="text-align:center; font-size:13px;">
AI-assisted resume classification | Built for real-world recruitment workflows
</p>
""", unsafe_allow_html=True)
