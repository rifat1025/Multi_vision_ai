import streamlit as st
import requests
import time

# FastAPI Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Multi-Model Vision AI App", layout="wide")
st.title("👁️ Multi-Model Vision AI Dashboard")
st.subheader("Upload an image to trigger Object Detection, Captioning, OCR & Chat!")

# Sidebar for history or configurations
st.sidebar.header("Controls & History")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("Uploading and running vision pipelines..."):
            # Send file to FastAPI
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{BACKEND_URL}/upload/", files=files)
            
            if response.status_code == 200:
                img_data = response.json()
                image_id = img_data["id"]
                st.success(f"Image received successfully! Assigned ID: {image_id}")
                
                # Polling backend for processing results
                analysis_ready = False
                with st.spinner("AI Models are running inference in the background..."):
                    for _ in range(15):  # Try for 15 seconds
                        time.sleep(2)
                        res = requests.get(f"{BACKEND_URL}/images/{image_id}/analysis")
                        if res.status_code == 200:
                            analysis_data = res.json()
                            analysis_ready = True
                            break
                
                if analysis_ready:
                    st.balloons()
                    st.session_state["image_id"] = image_id
                    st.session_state["analysis_data"] = analysis_data
                else:
                    st.error("Processing timed out. Please check backend logs.")
            else:
                st.error("Failed to upload image to backend.")

# 2. Display Results if Analysis Data is in State
if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]
    img_id = st.session_state["image_id"]
    
    st.markdown("---")
    st.header("📊 AI Inference Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Image Captioning")
        st.info(data.get("caption", "No caption generated."))
        
        st.subheader("🔍 Extracted OCR Text")
        if data.get("ocr_text"):
            st.code(data["ocr_text"])
        else:
            st.text("No text detected in image.")
            
    with col2:
        st.subheader("🤖 Classification (Confidence)")
        st.json(data.get("classification", {}))
        
        st.subheader("📦 Detected Objects (YOLO)")
        st.write(data.get("detected_objects", []))

    # 3. LangChain Chat Section
    st.markdown("---")
    st.header("💬 Chat with this Image (Powered by LangChain)")
    
    user_question = st.text_input("Ask a question about this image (e.g., 'What is written on the board?' or 'List the objects'):")
    
    if st.button("Ask AI"):
        
        if user_question:
            with st.spinner("Thinking..."):
                chat_res = requests.post(
                    f"{BACKEND_URL}/images/{img_id}/chat",
                    json={"question": user_question}
                )
                if chat_res.status_code == 200:
                    st.markdown(f"**AI Answer:** {chat_res.json()['answer']}")
                else:
                    st.error("Failed to get response from Chat API.")