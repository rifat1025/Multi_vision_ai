import streamlit as st
import requests
import time
import warnings

# Silence the PyTorch CPU warning to keep the terminal perfectly clean
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    message=".*'pin_memory' argument is set as true but no accelerator is found.*"
)

# FastAPI Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

# Professional Page Setup
st.set_page_config(
    page_title="Multi-Model Vision AI App", 
    layout="wide", 
    page_icon="👁️",
    initial_sidebar_state="expanded"
)

# Deep Tech Premium UI Style Injection
st.markdown("""
    <style>
    /* Global layout polishes */
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1200px; }
    h1 { font-weight: 700 !important; color: #1E293B; letter-spacing: -0.5px; }
    h2, h3 { font-weight: 600 !important; color: #334155; }
    
    /* Custom Stylized Cards for AI Data blocks */
    .ai-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .metric-label { font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase; margin-bottom: 0.25rem; }
    
    /* Clean borders for code blocks */
    .stCodeBlock { border: 1px solid #E2E8F0; border-radius: 8px !important; }
    
    /* Primary Action Buttons */
    div.stButton > button:first-child {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Application Header Accent
st.title("👁️ Multi-Model Vision AI Dashboard")
st.markdown("<p style='color: #64748B; font-size: 1.1rem; margin-top: -0.5rem;'>Orchestrate real-time Object Detection, Dense Captioning, OCR, and Contextual Analysis through a unified intelligence plane.</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# SIDEBAR CONFIGURATION: CONTROLS & INGESTION
# ==========================================
st.sidebar.header("Controls & History")
st.sidebar.markdown("---")

st.sidebar.subheader("📁 Browse File")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# --- FIX: Track if the file has changed to allow consecutive new uploads ---
if uploaded_file is not None:
    # If the user uploaded a completely different file name, clear the old session variables
    if "current_filename" not in st.session_state or st.session_state["current_filename"] != uploaded_file.name:
        st.session_state["current_filename"] = uploaded_file.name
        if "analysis_data" in st.session_state:
            del st.session_state["analysis_data"]
        if "image_id" in st.session_state:
            del st.session_state["image_id"]
        st.rerun()

if uploaded_file is not None:
    # Trigger processing interface ONLY if data isn't processed yet
    if "analysis_data" not in st.session_state:
        st.sidebar.markdown("### 🖼️ Selected Source")
        st.sidebar.image(uploaded_file, use_container_width=True)
        
        if st.sidebar.button("⚡ Run Vision Inference", type="primary", use_container_width=True):
            with st.sidebar.spinner("Running vision pipelines..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload/", files=files)
                    
                    if response.status_code == 200:
                        img_data = response.json()
                        image_id = img_data["id"]
                        st.toast(f"📦 Ingested Asset ID: {image_id}", icon="✅")
                        
                        analysis_ready = False
                        with st.sidebar.spinner("Executing deep inference on CPU..."):
                            for _ in range(45):  
                                time.sleep(1)
                                res = requests.get(f"{BACKEND_URL}/images/{image_id}/analysis")
                                if res.status_code == 200:
                                    analysis_data = res.json()
                                    analysis_ready = True
                                    break
                        
                        if analysis_ready:
                            st.balloons()
                            st.session_state["image_id"] = image_id
                            st.session_state["analysis_data"] = analysis_data
                            st.rerun()
                        else:
                            st.sidebar.error("Processing timed out. Check backend console logs.")
                    else:
                        st.sidebar.error(f"Upload failed. Status Code: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.sidebar.error("Could not connect to FastAPI server.")

# Dynamic LangChain Chat Placement Below File Browser
if "analysis_data" in st.session_state:
    img_id = st.session_state["image_id"]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 LangChain Image Chat")
    st.sidebar.markdown("<p style='color: #64748B; font-size: 0.85rem; margin-top: -0.5rem;'>Ask questions contextually linked to the active image.</p>", unsafe_allow_html=True)
    
    user_question = st.sidebar.text_input(
        "Ask a specific question about this image:", 
        placeholder="e.g., What is written on the board?",
        label_visibility="collapsed"
    )
    
    if st.sidebar.button("Ask AI Engine", type="primary", use_container_width=True):
        if user_question:
            st.sidebar.markdown(f"**Q:** `{user_question}`")
            with st.sidebar.spinner("Interrogating Context..."):
                try:
                    chat_res = requests.post(
                        f"{BACKEND_URL}/images/{img_id}/chat",
                        json={"question": user_question}
                    )
                    if chat_res.status_code == 200:
                        st.sidebar.markdown("**🤖 Response:**")
                        st.sidebar.info(chat_res.json()['answer'])
                    else:
                        st.sidebar.error("Failed to get response from the Chat node.")
                except Exception as e:
                    st.sidebar.error(f"Chat failed: {str(e)}")

# ==========================================
# MAIN WORKSPACE AREA: VIEWPORT & TELEMETRY
# ==========================================
if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]
    img_id = st.session_state["image_id"]
    
    # Side-by-Side Images
    st.markdown("### 📸 Image Process Mapping")
    col_source, col_detected = st.columns(2, gap="large")
    
    with col_source:
        st.markdown('<div class="ai-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">📥 Given Image</div>', unsafe_allow_html=True)
        st.image(uploaded_file, caption="Original Input Source", width=500)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_detected:
        st.markdown('<div class="ai-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">📦 Get Image (Detected Objects)</div>', unsafe_allow_html=True)
        detected_image_url = f"{BACKEND_URL}/images/{img_id}/detected-image"
        st.image(detected_image_url, caption="YOLO Model Analytical Plot Bounding Boxes", width=500)
        st.markdown('</div>', unsafe_allow_html=True)

    # Telemetry data running downward
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("📊 Deep Inference Insights & Telemetry Ledger")
    st.markdown("---")
    
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">📝 Image Caption Interpretation</div>', unsafe_allow_html=True)
    st.info(data.get("caption", "No caption generated."))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">🔍 Extracted Text (OCR Payload)</div>', unsafe_allow_html=True)
    if data.get("ocr_text"):
        st.code(data["ocr_text"], language="text")
    else:
        st.caption("No alphanumeric characters detected in asset matrix.")
    st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">🤖 Soft Classification Spectrum</div>', unsafe_allow_html=True)
    st.json(data.get("classification", {}))
    st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">📦 YOLO Target Instance Array</div>', unsafe_allow_html=True)
    metadata_list = data.get("detected_objects", [])
    if metadata_list:
        st.write(metadata_list)
    else:
        st.caption("No explicit object instances passed confidence thresholds.")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Neutral state when no image is uploaded/analyzed yet
    st.info("💡 Application ready. Ingest an image file using the sidebar panel control parameters to execute visual pipelines.")