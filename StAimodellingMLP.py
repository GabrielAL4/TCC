import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib  # Para salvar e carregar o modelo treinado

# Carregar os dados
csv_name = 'alzheimers_prediction_dataset.csv'
df = pd.read_csv(csv_name)

# Remover a coluna 'Country' pois não é relevante
if 'Country' in df.columns:
    df = df.drop(columns=['Country'])

# Identificar colunas categóricas
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Alzheimer’s Diagnosis')

# Aplicar Label Encoding para a coluna alvo
encoder = LabelEncoder()
df['Alzheimer’s Diagnosis'] = encoder.fit_transform(df['Alzheimer’s Diagnosis'])

# Aplicar OneHot Encoding para colunas categóricas nominais
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Dividir os dados em features (X) e alvo (y)
X = df.drop(columns=['Alzheimer’s Diagnosis'])
y = df['Alzheimer’s Diagnosis']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

# Salvar o modelo treinado
joblib.dump(mlp, 'mlp_model.pkl')

# Streamlit UI
st.title("Alzheimer's Risk Prediction")
st.write("Preencha os campos abaixo para obter uma previsão do risco de Alzheimer.")

# Obter as features do usuário
user_input = {}

for col in X.columns:
    if 'Gender' in col or 'Smoking Status' in col or 'Alcohol Consumption' in col or 'Employment Status' in col or 'Marital Status' in col or 'Urban vs Rural Living' in col:
        user_input[col] = st.selectbox(f"{col}:", [0, 1])
    else:
        user_input[col] = st.number_input(f"{col}:", min_value=0.0, max_value=100.0, value=0.0)

# Converter entrada do usuário para DataFrame
user_data = pd.DataFrame([user_input])

# Carregar o modelo treinado
loaded_model = joblib.load('mlp_model.pkl')

# Fazer a previsão
prediction = loaded_model.predict(user_data)[0]
prediction_proba = loaded_model.predict_proba(user_data)[0][1]  # Probabilidade de ser positivo

# Exibir o resultado
if prediction == 1:
    st.warning(f"⚠️ O modelo indica um risco elevado de Alzheimer com uma probabilidade de {prediction_proba * 100:.2f}%.")
else:
    st.success(f"✅ O modelo indica um baixo risco de Alzheimer com uma probabilidade de {(1 - prediction_proba) * 100:.2f}%.")

