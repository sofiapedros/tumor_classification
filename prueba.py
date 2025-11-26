"""
Script para extraer la configuración exacta del dataset de entrenamiento
Ejecuta este script en el mismo entorno donde entrenaste el modelo
"""

import pandas as pd
import numpy as np

# CAMBIA ESTA RUTA por la de tu archivo Excel
EXCEL_FILE = "data/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
SHEET_NAME = "BrEaST-Lesions-USG clinical dat"

def extract_dataset_config(excel_file, sheet_name):
    """Extrae la configuración exacta del procesamiento del dataset"""
    
    print("=" * 80)
    print("EXTRAYENDO CONFIGURACIÓN DEL DATASET")
    print("=" * 80)
    
    # Cargar el Excel
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    print(f"✓ Dataset cargado: {len(df)} filas")
    
    # Eliminar columnas que no son features
    drop_cols = [
        "CaseID",
        "Image_filename",
        "Mask_tumor_filename",
        "Mask_other_filename",
        "Diagnosis",
        "Verification",
        "Interpretation",
        "BIRADS",
        "Classification"
    ]
    
    tabular_df = df.drop(columns=drop_cols, errors="ignore")
    print(f"✓ Columnas después de eliminar metadata: {len(tabular_df.columns)}")
    
    # Mapear yes/no/not applicable
    def map_yes_no_na(col):
        if set(col.dropna().unique()).issubset({"yes", "no", "not applicable"}):
            return col.map({"yes": 1, "no": 0, "not applicable": 2}).astype(np.float32)
        return col
    
    tabular_df = tabular_df.apply(map_yes_no_na)
    print(f"✓ Mapeo yes/no/not applicable aplicado")
    
    # One-hot encoding
    categorical_cols = tabular_df.select_dtypes(include="object").columns
    print(f"✓ Columnas categóricas encontradas: {list(categorical_cols)}")
    
    tabular_df = pd.get_dummies(tabular_df, columns=categorical_cols)
    print(f"✓ One-hot encoding aplicado: {len(tabular_df.columns)} columnas totales")
    
    # Convertir a float32
    for col in tabular_df.columns:
        tabular_df[col] = pd.to_numeric(tabular_df[col], errors="coerce").fillna(0).astype(np.float32)
    
    # Identificar columnas binarias y numéricas
    binary_cols = [col for col in tabular_df.columns 
                   if set(tabular_df[col].unique()).issubset({0.0, 1.0, 2.0})]
    numeric_cols = [col for col in tabular_df.columns if col not in binary_cols]
    
    print(f"✓ Columnas numéricas (a normalizar): {numeric_cols}")
    print(f"✓ Columnas binarias (sin normalizar): {len(binary_cols)}")
    
    # Calcular estadísticas de normalización
    normalization_stats = {}
    for col in numeric_cols:
        normalization_stats[col] = {
            'mean': float(tabular_df[col].mean()),
            'std': float(tabular_df[col].std())
        }
    
    print("\n" + "=" * 80)
    print("CONFIGURACIÓN PARA STREAMLIT")
    print("=" * 80)
    
    print("\n# 1. COPIA ESTA LISTA COMPLETA en ALL_EXPECTED_COLUMNS:")
    print("ALL_EXPECTED_COLUMNS = [")
    for col in tabular_df.columns:
        print(f"    '{col}',")
    print("]")
    
    print(f"\n# Total de columnas: {len(tabular_df.columns)}")
    
    print("\n# 2. COPIA ESTAS ESTADÍSTICAS en NORMALIZATION_STATS:")
    print("NORMALIZATION_STATS = {")
    for col, stats in normalization_stats.items():
        print(f"    '{col}': {{'mean': {stats['mean']}, 'std': {stats['std']}}},")
    print("}")
    
    print("\n" + "=" * 80)
    print("VERIFICACIÓN")
    print("=" * 80)
    
    # Crear un ejemplo de procesamiento
    example_row = {
        'Pixel_size': 0.05,
        'Age': 50,
        'Tissue_composition': 'homogeneous background echotexture - fat',
        'Signs': 'no',
        'Symptoms': 'yes',
        'Shape': 'oval',
        'Margin': 'circumscribed',
        'Echogenicity': 'hypoechoic',
        'Posterior_features': 'no posterior features',
        'Halo': 'no',
        'Calcifications': 'no',
        'Skin_thickening': 'no'
    }
    
    # Procesar el ejemplo
    test_df = pd.DataFrame([example_row])
    
    # Mapear yes/no
    yes_no_cols = ['Signs', 'Symptoms', 'Halo', 'Calcifications', 'Skin_thickening']
    for col in yes_no_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].map({'yes': 1, 'no': 0, 'not applicable': 2}).astype(np.float32)
    
    # One-hot encoding
    cat_cols = ['Tissue_composition', 'Shape', 'Margin', 'Echogenicity', 'Posterior_features']
    test_df = pd.get_dummies(test_df, columns=cat_cols)
    
    # Añadir columnas faltantes
    for col in tabular_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0.0
    
    # Reordenar
    test_df = test_df[tabular_df.columns]
    
    # Normalizar
    for col in numeric_cols:
        if col in test_df.columns:
            test_df[col] = (test_df[col] - normalization_stats[col]['mean']) / (normalization_stats[col]['std'] + 1e-6)
    
    print(f"✓ Ejemplo procesado correctamente: {len(test_df.columns)} features")
    print(f"✓ Shape del vector de features: {test_df.values.shape}")
    
    return tabular_df.columns.tolist(), normalization_stats


if __name__ == "__main__":
    try:
        columns, stats = extract_dataset_config(EXCEL_FILE, SHEET_NAME)
        print("\n✅ CONFIGURACIÓN EXTRAÍDA EXITOSAMENTE")
        print(f"📊 Total de features: {len(columns)}")
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró el archivo '{EXCEL_FILE}'")
        print("Por favor, actualiza la variable EXCEL_FILE con la ruta correcta")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()