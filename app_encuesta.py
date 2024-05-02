import pandas as pd
import streamlit as st
#import numpy as np
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Modelo de Predicción del voto para Gobernatura para la Coalición PAN-PRI-NA")

@st.cache(allow_output_mutation=True)
def get_model():
    return load_model("final_modelo_lr")

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['prediction_label'][0]

model = get_model()

st.title("Modelo de Predicción del voto para Gobernatura para la coalición PAN-PRI-NA")
st.markdown("Elija los parámetros para realizar la predicción")

form = st.form("Perfil Ciudadano")
edad= form.number_input('edad', min_value = 18 , max_value = 100, value=50 , format = '%.2f', step = 1)
califica_presidente= form.number_input('califica_presidente', min_value = 0 , max_value = 10, value=5 , format = '%.2f', step = 1)
califica_gobernador= form.number_input('califica_gobernador', min_value = 0 , max_value = 10, value=5 , format = '%.2f', step = 1)
distance_km= form.number_input('distance_km', min_value = 0.0 , max_value = 500.0, value=50.0 , format = '%.2f', step = 0.1)
Nse_list = ['A/B', 'C+', 'C', 'C-', 'D+','D', 'E', 'N/D']
NSE = form.selectbox('NSE', Nse_list)
df_list = ['1', '2', '3', '4', '5', '6']
DF = form.selectbox('DF', df_list)
predict_button = form.form_submit_button('Predict')
input_dict = {'edad': edad, 'califica_presidente': califica_presidente, 'califica_gobernador': califica_gobernador,'distance_km':distance_km, 'NSE': NSE, 'DF': DF}
input_df = pd.DataFrame([input_dict])
if predict_button:
    out = predict(model, input_df)
    if out == 1:
        st.success('La predicción del voto es a favor de la alianza PAN-PRI-NA.')
    else:
        st.error('La predicción del voto es en contra de la alianza PAN-PRI-NA.')
