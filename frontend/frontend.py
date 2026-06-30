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

# Sidebar for history or configurations
st.sidebar.header("Controls & History")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    # Outer wrapping card for data ingestion
    with st.container():
        st.markdown("### 🖼️ Source Asset Preview")
        # Forces display preview strictly within a professional 500px width boundary
        st.image(uploaded_file, caption="Uploaded Image File (Target)", width=500)
        
        # Action button spanning full width of the upload container
        if st.button("⚡ Run Vision Inference", type="primary", use_container_width=True):
            with st.spinner("Uploading and running vision pipelines..."):
                # Send file to FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload/", files=files)
                    
                    if response.status_code == 200:
                        img_data = response.json()
                        image_id = img_data["id"]
                        st.toast(f"📦 Ingested Asset ID: {image_id}", icon="✅")
                        
                        # Polling backend for processing results
                        analysis_ready = False
                        
                        # Increased loop iterations to 45 to give your CPU plenty of headroom to complete inference without timing out
                        with st.spinner("AI Models are executing deep inference on CPU..."):
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
                            st.error("Processing timed out. Verify your backend logs for model blockage or long CPU calculation times.")
                    else:
                        st.error(f"Failed to upload image to backend. Status Code: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI. Please ensure your backend server is running on port 8000.")

# 2. Display Results if Analysis Data is in State
if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]
    img_id = st.session_state["image_id"]
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.header("📊 Multi-Model Inference Insights")
    st.markdown("---")
    
    # Keeping your original two primary columns layout intact
    col_img, col_info = st.columns(2, gap="large")
    
    with col_img:
        st.subheader("📦 Target Visual Mapping")
        # Direct URL linking to the FileResponse endpoint - constrained to 500px width
        detected_image_url = f"{BACKEND_URL}/images/{img_id}/detected-image"
        
        # Render the image inside a clean structural border layout
        st.markdown('<div class="ai-card">', unsafe_allow_html=True)
        st.image(detected_image_url, caption="YOLO Model Analytical Plot Bounding Boxes", width=500)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_info:
        # Wrap textual analytical payloads into modular telemetry components
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
        st.metadata_list = data.get("detected_objects", [])
        if st.metadata_list:
            st.write(st.metadata_list)
        else:
            st.caption("No explicit object instances passed confidence thresholds.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. LangChain Chat Section (Identical flow, stylized to be clean)
    st.markdown("---")
    st.header("💬 Contextual Interrogation Studio")
    st.markdown("<p style='color: #64748B; margin-top: -0.5rem;'>Interact natively with the asset layer using conversational natural language models.</p>", unsafe_allow_html=True)
    
    # Chat container layout
    with st.container():
        user_question = st.text_input(
            "Ask a specific question about this image:", 
            placeholder="e.g., 'What is written on the board?' or 'List the objects visible'",
            label_visibility="collapsed"
        )
        
        if st.button("Ask AI Engine", type="secondary", use_container_width=True):
            if user_question:
                st.markdown(f"**Question:** `{user_question}`")
                with st.spinner("Interrogating Image Context Engine..."):
                    try:
                        chat_res = requests.post(
                            f"{BACKEND_URL}/images/{img_id}/chat",
                            json={"question": user_question}
                        )
                        if chat_res.status_code == 200:
                            st.markdown("### 🤖 Response Payload:")
                            st.info(chat_res.json()['answer'])
                        else:
                            st.error("Failed to get a valid response from the Chat API node.")
                    except Exception as e:
                        st.error(f"Chat operation failed: {str(e)}")