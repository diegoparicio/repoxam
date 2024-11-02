import streamlit as st
import pandas as pd
import numpy as np
from ayudantes import *
from logica_test import *
from poison_test import *

#_________________________________________________________________

# Título de la aplicación
st.title("LaLiga: Similitud entre Futbolistas")

#_________________________________________________________________
# Agrupar los datos de jugadores

st.header("Estadísticas Globales")

df_agrupado = generar_df_agrupado(df)

# Mostrar el DataFrame en Streamlit
st.dataframe(df_agrupado)

#_________________________________________________________________

st.header("Estadísticas por Partido (90 Minutos)")

df_por_partido = generar_df_por_partido(df_agrupado)

st.dataframe(df_por_partido)

#_________________________________________________________________

# st.header("Estadísticas Barcelona")

df_consulta = generar_df_consulta(df_agrupado)

# st.dataframe(df_consulta)

#_________________________________________________________________

# DataFrame Filtrado
df_agrupado_f = generar_df_agrupado_f(df_agrupado)

#_________________________________________________________________

st.header("Estadísticas Métricas Depuradas")

df_metricas = generar_df_metricas(df_agrupado_f)

st.dataframe(df_metricas)

#_________________________________________________________________

st.header("Jugadores Similares")

jugador_base = "Raphinha"

# Menú para seleccionar un jugador
jugador = st.selectbox("Selecciona un jugador:", df_agrupado_f['jugador'], index=df_agrupado_f['jugador'].tolist().index(jugador_base))
# jugador = st.selectbox("Selecciona un jugador:", df_agrupado_f['jugador'])

# Slider para el número de jugadores similares a mostrar
topn = st.slider("Número de jugadores similares a mostrar:", 1, 10, 5)

# Calcular los jugadores más similares
resultados_similares = jugadores_similares(jugador, topn, df_agrupado_f)

# Mostrar resultados
st.write(f"Jugadores similares a {jugador}:")
st.write(resultados_similares)

# Mostrar gráfico de similitud
fig = generar_grafico_similitud(resultados_similares, jugador)
st.pyplot(fig)

#_________________________________________________________________

st.header("Gráficos Radiales")

J1_base = "Raphinha"
J2_base = "Lamine Yamal"

opcion_grafico = st.selectbox("Seleccione el tipo de gráfico radial. (Simple: métricas depuradas)", ["Simple", "Detallado"])

J1 = st.selectbox("Selecciona el primer jugador:", df_agrupado_f['jugador'], index=df_agrupado_f['jugador'].tolist().index(J1_base), key="jugador1")
J2 = st.selectbox("Selecciona el segundo jugador:", df_agrupado_f['jugador'], index=df_agrupado_f['jugador'].tolist().index(J2_base), key="jugador2")

# jugador1 = st.selectbox("Selecciona un jugador 1:", df_agrupado_f['jugador'])
# jugador2 = st.selectbox("Selecciona un jugador 2:", df_agrupado_f['jugador'])

if opcion_grafico == "Simple":

    df_percen_J1, pos_J1, valor_global_J1 = generar_percentiles_jugador_df_metricas(J1, df_metricas)
    df_percen_J2, pos_J2, valor_global_J2 = generar_percentiles_jugador_df_metricas(J2, df_metricas)

    df_final = pd.concat([df_percen_J1, df_percen_J2])
    valores_globales = [valor_global_J1, valor_global_J2]
    df_final['GLOBAL'] = valores_globales

    st.dataframe(df_final)

    # NO SE HACE ASÍ
    #globales = pd.concat([valor_global_J1, valor_global_J2])
    #st.dataframe(globales)

    
else:
    df_percen_J1, pos_J1 = generar_percentiles_jugador_df(J1, df_agrupado_f)
    df_percen_J2, pos_J2 = generar_percentiles_jugador_df(J2, df_agrupado_f)

    df_final = pd.concat([df_percen_J1, df_percen_J2])
    st.dataframe(df_final)


# Crear el gráfico de radar
fig = generar_grafico_radar_1(df_percen_J1, J1, pos_J1)

# Mostrar el gráfico en Streamlit
# st.pyplot(fig)
# st.text("Basado en los percentiles respecto a los jugadores que ocupan su misma posición")

# Crear el gráfico de radar
fig = generar_grafico_radar_2(df_percen_J1, df_percen_J2, J1, J2, pos_J1, pos_J2)

st.pyplot(fig)
st.write("Basado en los percentiles respecto a los jugadores que ocupan su misma posición")

#_________________________________________________________________

st.header("Predicción Goles y Asistencias")

jornadas_disputadas = calcular_jornadas_disputadas(df)

df_jugador_stat, jugador, stat_a_predecir = generar_df_jugador_stat(df, jornadas_disputadas)

pred_df, jornadas_a_predecir = generar_pred_df(df_jugador_stat, jornadas_disputadas)

fig = generar_grafico_predicion(df_jugador_stat, pred_df, stat_a_predecir, jornadas_disputadas, jornadas_a_predecir, jugador)

st.pyplot(fig)