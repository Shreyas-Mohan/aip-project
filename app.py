import streamlit as st
import numpy as np
import PIL.Image
import os
from dotenv import load_dotenv

# Load local .env variables
load_dotenv()

# Modular imports
from utils.model_handler import load_cnn_model, preprocess_image, generate_gradcam, apply_heatmap, get_reference_healthy_image, CLASS_LABELS
from utils.llm_handler import configure_gemini, generate_initial_report, chat_with_image_context
from utils.pdf_generator import create_pdf_report

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & INITIALIZATION
# ---------------------------------------------------------
st.set_page_config(page_title="AI Knee Diagnostician", page_icon="🦴", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🦴 Complete AI Knee Osteoarthritis Assistant")

# Cache model
@st.cache_resource
def get_model():
    return load_cnn_model()

cnn_model = get_model()

# ---------------------------------------------------------
# 2. SIDEBAR FOR SETTINGS
# ---------------------------------------------------------
# Securely fetch API Key for both local (.env) and deployed (secrets)
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        gemini_api_key = None

with st.sidebar:
    if not gemini_api_key:
        st.error("⚠️ Setup Required: Missing GEMINI_API_KEY in .env or Streamlit Secrets.")

    st.header("📋 Patient Information")
    patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=65)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    patient_history = st.text_area("Symptoms & Medical History", "e.g., Chronic left knee pain, worsens after walking.")
    
    st.header("🌍 Multi-Feature Options")
    audience_mode = st.radio("Explanation Style", ["Patient (Simplified)", "Doctor (Clinical)"])
    language_choice = st.selectbox("Report Language", ["English", "Hindi", "Spanish", "French", "Mandarin"])
    
    st.header("📚 Reference Guide")
    st.info("K-L Grading:\n\n0: Normal\n1: Doubtful\n2: Mild\n3: Moderate\n4: Severe")

# ---------------------------------------------------------
# 3. MAIN WORKFLOW
# ---------------------------------------------------------
if cnn_model is None:
    st.warning("⚠️ Model file not found. Please ensure 'final_model_export.keras' exists in the root folder.")

uploaded_file = st.file_uploader("Upload a Knee X-Ray Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and cnn_model is not None:
    # Preprocess
    img = PIL.Image.open(uploaded_file)
    img_array = preprocess_image(img)
    predictions = cnn_model.predict(img_array)[0]
    
    predicted_class_idx = int(np.argmax(predictions))
    predicted_label = CLASS_LABELS[predicted_class_idx]
    confidence_score = float(predictions[predicted_class_idx] * 100)
    
    st.markdown("---")
    
    # 3 Tab Interface
    tab1, tab2, tab3 = st.tabs(["📊 Diagnostics & Imagery", "📝 Multi-Modal AI Report", "💬 Interactive Follow-up Chat"])
    
    # --- TAB 1: DIAGNOSTICS & IMAGERY ---
    with tab1:
        st.success(f"**Diagnosis:** {predicted_label} | **Confidence:** {confidence_score:.2f}%")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("Your X-Ray")
            st.image(img, use_container_width=True)
            
        with col2:
            st.subheader("Healthy Knee (Reference)")
            ref_path = get_reference_healthy_image()
            if ref_path:
                # Need absolute path or use PIL to open it since Streamlit reads local relative paths depending on PWD
                st.image(PIL.Image.open(ref_path), use_container_width=True, caption="Grade 0 (Normal)")
            else:
                st.warning("No reference image found in workspace.")
                
        with col3:
            st.subheader("Grad-CAM Explainability")
            heatmap = generate_gradcam(img_array, cnn_model)
            if heatmap is not None:
                heatmap_img = apply_heatmap(img, heatmap)
                st.image(heatmap_img, use_container_width=True, caption="CNN Hotspots")
            else:
                st.info("Grad-CAM not available for this layer setup.")
                
        st.subheader("Class Probability Distribution")
        st.bar_chart({CLASS_LABELS[i]: float(predictions[i]) for i in range(5)})

    # --- TAB 2: AI REPORT (GEMINI) ---
    patient_info = {"age": patient_age, "gender": patient_gender, "history": patient_history}
    
    with tab2:
        if not gemini_api_key:
            st.error("Please provide Gemini API Key in sidebar.")
        else:
            if st.button("Generate Detailed Report", type="primary"):
                with st.spinner(f"Generating {language_choice} report in {audience_mode} tone..."):
                    vision_model = configure_gemini(gemini_api_key)
                    if vision_model:
                        report_text = generate_initial_report(
                            vision_model, img, predicted_label, confidence_score, 
                            patient_info, audience_mode, language_choice
                        )
                        st.write(report_text)
                        
                        pdf_path = create_pdf_report(patient_info, predicted_label, confidence_score, report_text, audience_mode, language_choice)
                        with open(pdf_path, "rb") as file:
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=file,
                                file_name=f"OA_Report_{audience_mode}_{language_choice}.pdf",
                                mime="application/pdf"
                            )
                        
                        # Save to context for chat
                        st.session_state.messages.append({"role": "assistant", "content": report_text})
                    else:
                        st.error("Failed to initialize Gemini. Check your API key.")

    # --- TAB 3: INTERACTIVE CHAT ---
    with tab3:
        st.subheader(f"Discuss X-Ray with AI ({audience_mode} mode)")
        
        # Display existing chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Chat input
        if prompt := st.chat_input("Ask a follow-up question regarding the X-ray or report..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            if gemini_api_key:
                with st.spinner("Analyzing..."):
                    vision_model = configure_gemini(gemini_api_key)
                    answer = chat_with_image_context(vision_model, img, st.session_state.messages, prompt)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)
            else:
                st.error("API Key needed to chat.")