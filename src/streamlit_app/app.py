import streamlit as st
import pandas as pd
from PIL import Image
import io
from src.backend.service import predictor 
from src.backend.service import predictor   
from src.backend.generete_dpf import pdf_service
from src.backend.gemini import gemini_service 


def main():
    st.title("🌿 Plant Health Diagnosis")
    st.write("Upload a leaf image for instant AI diagnosis and expert recommendations.")
    st.divider()


    if 'prediction_result' not in st.session_state:
        st.session_state['prediction_result'] = None
    if 'prediction_done' not in st.session_state:
        st.session_state['prediction_done'] = False

    uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
        

        if st.button("🔍 Analyze Image"):
            with st.spinner("Model is calculating..."):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                result =predictor.predict(img_bytes)
                
    
                st.session_state['prediction_result'] = result
                st.session_state['prediction_done'] = True

        if st.session_state['prediction_done']:
            res = st.session_state['prediction_result']
            
            st.markdown("---")
            st.subheader("Diagnosis Result")
            st.info(f"**Detected:** {res['class_name']}")
            st.write(f"**Confidence Level:** {res['confidence']}")

            if st.button("📝 Generate Detailed PDF Report"):
                with st.spinner("Consulting Gemini & Formatting PDF..."):
               
                    full_text = gemini_service.get_plant_report(res['class_name'], res['confidence'])
                    
                    # 2. نبعت كل الداتا لخدمة الـ PDF
                    pdf_data = pdf_service.generate(
                        res['class_name'], 
                        res['confidence'], 
                        full_text, 
                        image
                    )
                    
                    st.markdown("### 📋 Expert AI Advice Preview")
                    st.write(full_text)
                    
                    st.download_button(
                        label="📥 Download Diagnosis PDF",
                        data=pdf_data,
                        file_name=f"Plant_Diagnosis_{res['class_name']}.pdf",
                        mime="application/pdf"
        )
                    

if __name__ == "__main__":
    main()