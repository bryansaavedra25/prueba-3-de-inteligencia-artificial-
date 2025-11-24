import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

LOG = Path("logs/ep3_logs.jsonl")

st.title("EP3 - Dashboard Banco Andino")

if not LOG.exists():
    st.warning("No se encontraron logs en logs/ep3_logs.jsonl. Ejecuta consultas para generar datos.")
else:
    df = pd.read_json(LOG, lines=True)
    st.subheader("Resumen")
    st.metric("Consultas registradas", int(len(df)))
    st.metric("Latencia promedio (s)", round(df["latencia_total"].mean(), 3))
    st.metric("Tokens promedio", int(df["tokens"].mean()))
    st.subheader("Últimas consultas")
    st.dataframe(df[["request_id","pregunta","decision","latencia_total","tokens"]].sort_values(by="request_id", ascending=False).head(10))
    st.subheader("Distribución latencia")
    plt.hist(df["latencia_total"].dropna(), bins=20)
    st.pyplot()
    st.subheader("Top decisiones")
    st.bar_chart(df["decision"].value_counts())
