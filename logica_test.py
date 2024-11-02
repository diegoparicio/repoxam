import streamlit as st

# logica.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from ayudantes import *

#_________________________________________________________________
from sqlalchemy import engine
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
#_________________________________________________________________

# 1. CARGA DE DATOS

def cargar_datos(ruta_csv):
    """
    Carga el archivo CSV con los datos de los jugadores y devuelve un DataFrame.
    """
    df = pd.read_csv(ruta_csv)
    
    # Renombrar columnas para facilitar el acceso
    df.columns = [
        'jugador', 'equipo', 'numero', 'nacionalidad', 'posicion', 
        'edad', 'minutos', 'goles', 'asistencias', 'penaltis_marcados', 'penaltis_intentados',
        'total_tiros', 'tiros_a_puerta', 'amarillas', 'rojas', 'contactos', 'accion_defensiva', 'cortes',
        'bloqueos', 'goles_esp', 'goles_esp_np', 'goles_esp_asistidos', 'acciones_creacion_tiro', 'acciones_creacion_gol', 
        'pases_completados', 'pases_intentados', 'pases_completados_p', 'pases_progresivos', 
        'conducciones', 'conducciones_progresivas', 'intentos_regate', 'regates_exitosos', 'fecha'
    ]
    
    # Retornar el DataFrame cargado
    return df

# Cargar los datos (este es un ejemplo, el CSV debe estar en la ruta indicada)
df = cargar_datos('laliga/database.csv')

st.write(df)