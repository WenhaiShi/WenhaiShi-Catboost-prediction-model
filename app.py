
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Risk Prediction Model for Invasive Mechanical Ventilation in ICU Patients with Left Heart Failure",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    try:
        # 尝试直接加载模型
        model = joblib.load('final_CatBoost_model_selected_features.pkl')
        
        # 检查加载的对象类型
        if hasattr(model, 'predict_proba'):
            # 如果直接是模型
            return model, None
        elif isinstance(model, dict) and 'model' in model:
            # 如果是包含模型的字典
            return model['model'], model.get('selected_features')
        else:
            st.error(f"Unexpected model format: {type(model)}")
            return None, None
            
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

st.title("Risk Prediction Model for Invasive Mechanical Ventilation in ICU Patients with Left Heart Failure")
st.markdown("---")

model, selected_features = load_model()
if model is None:
    st.stop()

# 如果selected_features为None，使用默认的特征列表
if selected_features is None:
    selected_features = ['Age', 'LOS', 'Weight', 'SOFA', 'RR', 'PH', 'PCO2', 'OI', 'WBC', 'INR', 'Chloride', 'Glucose', 'PT', 'ALT', 'Average output urine']

feature_ranges = {
    'Age': (18, 120, 'years'),
    'LOS': (0, 100, 'days'),
    'Weight': (0, 400, 'kg'),
    'SOFA': (0, 24, 'points'),
    'RR': (0, 100, 'BPM'),
    'PH': (6.8, 7.8, ''),
    'PCO2': (0, 200, 'mmHg'),
    'OI': (0, 3000, 'mmHg'),
    'WBC': (0, 50, '×10⁹/L'),
    'INR': (0, 10, ''),
    'Chloride': (80, 130, 'mmol/L'),
    'Glucose': (0, 1000, 'mg/dL'),
    'PT': (0, 100, 'second'),
    'ALT': (0, 2000, 'U/L'),
    'Average output urine': (0, 5000, 'mL')
}

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Patient Information Input")
    
    with st.form("patient_form"):
        input_col1, input_col2 = st.columns(2)
        
        input_values = {}
        
        with input_col1:
            for i, feature in enumerate(selected_features[:len(selected_features)//2]):
                min_val, max_val, unit = feature_ranges.get(feature, (0, 100, ''))
                input_values[feature] = st.number_input(
                    f"{feature} ({unit})" if unit else feature,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=0.1,
                    key=f"{feature}_1"
                )
        
        with input_col2:
            for i, feature in enumerate(selected_features[len(selected_features)//2:]):
                min_val, max_val, unit = feature_ranges.get(feature, (0, 100, ''))
                input_values[feature] = st.number_input(
                    f"{feature} ({unit})" if unit else feature,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=0.1,
                    key=f"{feature}_2"
                )
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # 确保所有选中的特征都有值
            missing_features = set(selected_features) - set(input_values.keys())
            if missing_features:
                st.error(f"Missing features: {missing_features}")
            else:
                # 按选中的特征顺序创建输入数据
                input_data = [input_values[feature] for feature in selected_features]
                input_df = pd.DataFrame([input_data], columns=selected_features)
                
                try:
                    probability = model.predict_proba(input_df)[0][1] * 100
                    st.success("Prediction completed!")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

with col2:
    st.header("Prediction Results")
    
    if submitted and 'probability' in locals():
        st.metric(
            label="Invasive Mechanical Ventilation Probability",
            value=f"{probability:.1f}%"
        )
    else:
        st.info("Please fill in the patient information and click 'Predict' to see results.")

st.markdown("---")
st.header("About This Calculator")

about_text = f"""
This Risk Prediction Model for Invasive Mechanical Ventilation in ICU Patients with Left Heart Failure uses a machine learning model trained on clinical data to assess the risk of requiring invasive mechanical ventilation based on {len(selected_features)} key patient parameters.

**Selected Clinical Features**:
- **Demographics & Clinical Scores**: {', '.join([f for f in selected_features if f in ['Age', 'LOS', 'Weight', 'SOFA']])}
- **Vital Signs & Blood Gas**: {', '.join([f for f in selected_features if f in ['RR', 'PH', 'PCO2', 'OI']])}
- **Laboratory Values**: {', '.join([f for f in selected_features if f in ['WBC','INR', 'Chloride', 'Glucose', 'PT', 'ALT']])}
- **Renal Function**: {', '.join([f for f in selected_features if f in ['Average output urine']])}

**Note**: This tool is intended for clinical decision support and should be used in conjunction with professional medical judgment.
"""

st.markdown(about_text)
