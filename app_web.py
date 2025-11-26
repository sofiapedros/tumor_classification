import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import os

st.title('Predicción de Lesiones Mamarias')
st.write('Ingresa las características clínicas y de ultrasonido para obtener una predicción.')

# Configuración exacta extraída del dataset de entrenamiento
ALL_EXPECTED_COLUMNS = [
    'Pixel_size',
    'Halo',
    'Skin_thickening',
    'Age_18', 'Age_20', 'Age_21', 'Age_23', 'Age_25', 'Age_26', 'Age_27', 'Age_29', 'Age_30',
    'Age_31', 'Age_32', 'Age_33', 'Age_34', 'Age_35', 'Age_36', 'Age_37', 'Age_38', 'Age_39',
    'Age_40', 'Age_41', 'Age_42', 'Age_43', 'Age_44', 'Age_45', 'Age_46', 'Age_47', 'Age_48',
    'Age_49', 'Age_50', 'Age_51', 'Age_52', 'Age_53', 'Age_54', 'Age_55', 'Age_56', 'Age_57',
    'Age_58', 'Age_59', 'Age_60', 'Age_61', 'Age_62', 'Age_63', 'Age_64', 'Age_65', 'Age_66',
    'Age_67', 'Age_68', 'Age_69', 'Age_70', 'Age_71', 'Age_72', 'Age_73', 'Age_74', 'Age_75',
    'Age_76', 'Age_77', 'Age_78', 'Age_80', 'Age_85', 'Age_86', 'Age_87', 'Age_not available',
    'Tissue_composition_heterogeneous: predominantly fat',
    'Tissue_composition_heterogeneous: predominantly fibroglandular',
    'Tissue_composition_homogeneous: fat',
    'Tissue_composition_homogeneous: fibroglandular',
    'Tissue_composition_lactating',
    'Tissue_composition_lactating&heterogeneous: predominantly fat',
    'Tissue_composition_lactating&heterogeneous: predominantly fibroglandular',
    'Tissue_composition_lactating&homogeneous: fibroglandular',
    'Tissue_composition_not available',
    'Signs_breast scar', 'Signs_breast scar&skin retraction', 'Signs_nipple retraction',
    'Signs_nipple retraction&palpable', 'Signs_no', 'Signs_not available', 'Signs_palpable',
    'Signs_palpable&breast scar', 'Signs_peau d`orange&palpable', 'Signs_redness&warmth',
    'Signs_redness&warmth&palpable', 'Signs_skin retraction&palpable', 'Signs_warmth&palpable',
    'Symptoms_HRT/hormonal contraception', 'Symptoms_breast injury',
    'Symptoms_family history of breast/ovarian cancer',
    'Symptoms_family history of breast/ovarian cancer&HRT/hormonal contraception',
    'Symptoms_nipple discharge', 'Symptoms_nipple discharge&family history of breast/ovarian cancer',
    'Symptoms_no', 'Symptoms_not available', 'Symptoms_personal history of breast cancer',
    'Symptoms_personal history of breast cancer&family history of breast/ovarian cancer',
    'Shape_irregular', 'Shape_not applicable', 'Shape_oval', 'Shape_round',
    'Margin_circumscribed', 'Margin_not applicable', 'Margin_not circumscribed - angular',
    'Margin_not circumscribed - angular&indistinct', 'Margin_not circumscribed - angular&microlobulated',
    'Margin_not circumscribed - angular&microlobulated&indistinct',
    'Margin_not circumscribed - indistinct', 'Margin_not circumscribed - microlobulated',
    'Margin_not circumscribed - microlobulated&indistinct', 'Margin_not circumscribed - spiculated',
    'Margin_not circumscribed - spiculated&angular',
    'Margin_not circumscribed - spiculated&angular&indistinct',
    'Margin_not circumscribed - spiculated&angular&microlobulated&indistinct',
    'Margin_not circumscribed - spiculated&indistinct',
    'Margin_not circumscribed - spiculated&microlobulated&indistinct',
    'Echogenicity_anechoic', 'Echogenicity_complex cystic/solid', 'Echogenicity_heterogeneous',
    'Echogenicity_hyperechoic', 'Echogenicity_hypoechoic', 'Echogenicity_isoechoic',
    'Echogenicity_not applicable',
    'Posterior_features_combined', 'Posterior_features_enhancement', 'Posterior_features_no',
    'Posterior_features_not applicable', 'Posterior_features_shadowing',
    'Calcifications_in a mass', 'Calcifications_indefinable', 'Calcifications_intraductal',
    'Calcifications_no', 'Calcifications_not applicable',
]

NORMALIZATION_STATS = {
    'Pixel_size': {'mean': 0.007615044713020325, 'std': 0.0016304438468068838},
}

# Crear dos columnas para organizar mejor los inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Datos del Paciente")
    pixel_size = st.number_input('Tamaño de píxel', min_value=0.0, value=0.0076, step=0.001, format="%.4f")
    age = st.number_input('Edad', min_value=18, max_value=90, value=50, step=1)
    
    st.subheader("Composición Tisular")
    tissue_composition = st.selectbox(
        'Composición del tejido',
        ['homogeneous: fat', 'homogeneous: fibroglandular', 
         'heterogeneous: predominantly fat', 'heterogeneous: predominantly fibroglandular',
         'lactating', 'lactating&heterogeneous: predominantly fat',
         'lactating&heterogeneous: predominantly fibroglandular',
         'lactating&homogeneous: fibroglandular', 'not available']
    )

with col2:
    st.subheader("Signos")
    signs = st.selectbox('Signos', [
        'no', 'palpable', 'nipple retraction', 'breast scar', 'skin retraction&palpable',
        'redness&warmth', 'warmth&palpable', 'redness&warmth&palpable', 
        'nipple retraction&palpable', 'palpable&breast scar', 'breast scar&skin retraction',
        'peau d`orange&palpable', 'not available'
    ])
    
    st.subheader("Síntomas")
    symptoms = st.selectbox('Síntomas', [
        'no', 'family history of breast/ovarian cancer', 'HRT/hormonal contraception',
        'personal history of breast cancer', 'nipple discharge', 'breast injury',
        'family history of breast/ovarian cancer&HRT/hormonal contraception',
        'nipple discharge&family history of breast/ovarian cancer',
        'personal history of breast cancer&family history of breast/ovarian cancer',
        'not available'
    ])

st.subheader("Características de la Lesión")

col3, col4 = st.columns(2)

with col3:
    shape = st.selectbox('Forma', ['oval', 'round', 'irregular', 'not applicable'])
    
    margin = st.selectbox('Margen', [
        'circumscribed', 'not circumscribed - angular', 'not circumscribed - indistinct',
        'not circumscribed - microlobulated', 'not circumscribed - spiculated',
        'not circumscribed - angular&indistinct', 'not circumscribed - angular&microlobulated',
        'not circumscribed - angular&microlobulated&indistinct',
        'not circumscribed - microlobulated&indistinct',
        'not circumscribed - spiculated&angular', 'not circumscribed - spiculated&indistinct',
        'not circumscribed - spiculated&angular&indistinct',
        'not circumscribed - spiculated&microlobulated&indistinct',
        'not circumscribed - spiculated&angular&microlobulated&indistinct',
        'not applicable'
    ])
    
    echogenicity = st.selectbox('Ecogenicidad', [
        'hypoechoic', 'isoechoic', 'hyperechoic', 'anechoic', 
        'complex cystic/solid', 'heterogeneous', 'not applicable'
    ])

with col4:
    posterior_features = st.selectbox('Características posteriores', [
        'no', 'enhancement', 'shadowing', 'combined', 'not applicable'
    ])
    
    halo = st.selectbox('Halo', ['no (0)', 'yes (1)', 'not applicable (2)'])
    calcifications = st.selectbox('Calcificaciones', [
        'no', 'in a mass', 'intraductal', 'indefinable', 'not applicable'
    ])
    skin_thickening = st.selectbox('Engrosamiento de piel', ['no (0)', 'yes (1)', 'not applicable (2)'])

def process_input_data(data_dict):
    """Procesa los datos de entrada EXACTAMENTE como el dataset de entrenamiento"""
    
    # Crear DataFrame inicial vacío con todas las columnas esperadas
    df = pd.DataFrame(0.0, index=[0], columns=ALL_EXPECTED_COLUMNS, dtype=np.float32)
    
    # 1. Pixel_size (normalizado)
    df['Pixel_size'] = (data_dict['Pixel_size'] - NORMALIZATION_STATS['Pixel_size']['mean']) / \
                       (NORMALIZATION_STATS['Pixel_size']['std'] + 1e-6)
    
    # 2. Halo y Skin_thickening (valores 0, 1, 2)
    df['Halo'] = data_dict['Halo']
    df['Skin_thickening'] = data_dict['Skin_thickening']
    
    # 3. Age (one-hot encoding)
    age_col = f"Age_{data_dict['Age']}"
    if age_col in df.columns:
        df[age_col] = 1.0
    
    # 4. Tissue_composition (one-hot encoding)
    tissue_col = f"Tissue_composition_{data_dict['Tissue_composition']}"
    if tissue_col in df.columns:
        df[tissue_col] = 1.0
    
    # 5. Signs (one-hot encoding)
    signs_col = f"Signs_{data_dict['Signs']}"
    if signs_col in df.columns:
        df[signs_col] = 1.0
    
    # 6. Symptoms (one-hot encoding)
    symptoms_col = f"Symptoms_{data_dict['Symptoms']}"
    if symptoms_col in df.columns:
        df[symptoms_col] = 1.0
    
    # 7. Shape (one-hot encoding)
    shape_col = f"Shape_{data_dict['Shape']}"
    if shape_col in df.columns:
        df[shape_col] = 1.0
    
    # 8. Margin (one-hot encoding)
    margin_col = f"Margin_{data_dict['Margin']}"
    if margin_col in df.columns:
        df[margin_col] = 1.0
    
    # 9. Echogenicity (one-hot encoding)
    echo_col = f"Echogenicity_{data_dict['Echogenicity']}"
    if echo_col in df.columns:
        df[echo_col] = 1.0
    
    # 10. Posterior_features (one-hot encoding)
    post_col = f"Posterior_features_{data_dict['Posterior_features']}"
    if post_col in df.columns:
        df[post_col] = 1.0
    
    # 11. Calcifications (one-hot encoding)
    calc_col = f"Calcifications_{data_dict['Calcifications']}"
    if calc_col in df.columns:
        df[calc_col] = 1.0
    
    return df

# Botón para hacer la predicción
if st.button(' Obtener Predicción', type='primary'):
    # Extraer valores de halo y skin_thickening
    halo_value = 0 if 'no' in halo else (1 if 'yes' in halo else 2)
    skin_value = 0 if 'no' in skin_thickening else (1 if 'yes' in skin_thickening else 2)
    
    # Crear diccionario con los datos de entrada
    input_data = {
        'Pixel_size': pixel_size,
        'Age': age,
        'Tissue_composition': tissue_composition,
        'Signs': signs,
        'Symptoms': symptoms,
        'Shape': shape,
        'Margin': margin,
        'Echogenicity': echogenicity,
        'Posterior_features': posterior_features,
        'Halo': halo_value,
        'Calcifications': calcifications,
        'Skin_thickening': skin_value
    }
    
    try:
        processed_df = process_input_data(input_data)
        features = processed_df.values.flatten().tolist()
        
        st.info(f"Número de features enviadas: {len(features)}")
        
        payload = {'features': features}
        api_url = os.environ.get('API_URL', 'http://127.0.0.1:5000/predict')
        
        with st.spinner('Realizando predicción...'):
            try:
                response = requests.post(api_url, data=json.dumps(payload),
                                       headers={'Content-Type': 'application/json'}, timeout=10)
                
                if response.status_code == 200:
                    prediction_result = response.json().get('prediction')
                    classification_map = {0: 'Benigno', 1: 'Maligno', 2: 'Normal'}
                    predicted_class = classification_map.get(prediction_result, 'Desconocido')
                    
                    if prediction_result == 0:
                        st.success(f"La predicción es: **{predicted_class}**")
                    elif prediction_result == 1:
                        st.error(f"La predicción es: **{predicted_class}**")
                    else:
                        st.info(f"La predicción es: **{predicted_class}**")
                    
                    st.info("**Nota:** Esta predicción es solo una herramienta de apoyo. Siempre consulte con un profesional médico.")
                else:
                    st.error(f"Error en la petición: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"No se pudo conectar con la API.")
                st.error(f"Error: {e}")
                st.info(f"Intentando conectar a: {api_url}")
                
    except Exception as e:
        st.error(f"Error al procesar los datos: {str(e)}")
        st.exception(e)

# Sidebar
with st.sidebar:
    st.header("ℹ️ Información")
    st.write("""
    Esta aplicación utiliza un modelo de aprendizaje automático 
    para predecir la clasificación de lesiones mamarias basándose 
    en características de ultrasonido y datos clínicos.
    
    **Clasificaciones:**
    - Benigno
    - Maligno
    - Normal
    """)
    
    st.header("Configuración")
    st.write(f"API URL: `{os.environ.get('API_URL', 'http://127.0.0.1:5000/predict')}`")
    st.write(f"Features esperadas: **{len(ALL_EXPECTED_COLUMNS)}**")
    
    if st.button("Probar Conexión API"):
        api_url = os.environ.get('API_URL', 'http://127.0.0.1:5000/predict')
        try:
            test_payload = {'features': [0.0] * len(ALL_EXPECTED_COLUMNS)}
            response = requests.post(api_url, json=test_payload, timeout=5)
            if response.status_code == 200:
                st.success("Conexión exitosa")
                st.write(f"Predicción de prueba: {response.json()}")
            else:
                st.error(f"Error: {response.status_code}")
        except Exception as e:
            st.error(f"Error: {str(e)}")