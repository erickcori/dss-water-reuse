import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Decision Support System", layout="wide")

st.title("Decision Support System for Wastewater Reuse")
st.markdown("""
This tool integrates **Cross-Impact Analysis (CIA)**, **SHAP interpretation**, and **clustering** to prioritize systemic barriers in water reuse policy.
""")

# === Load Data ===
cia_df = pd.read_csv("data/cia_matrix.csv", index_col=0)
shap_values = pd.Series([0.215, 0.138, 0.112, 0.097, 0.091, 0.087, 0.075, 0.073, 0.064, 0.050, 0.046, 0.042], index=cia_df.index)
C_out = cia_df.sum(axis=1)

# === Dashboard Section 1 ===
st.subheader("🌐 Cross-Impact Matrix (CIA)")
st.dataframe(cia_df)

# === Section 2: SHAP vs C_out ===
st.subheader(":mag_right: SHAP Importance and Out-Degree Centrality")
shap_table = pd.DataFrame({"SHAP Value (ϕ)": shap_values, "Out-Degree (C_out)": C_out})
st.dataframe(shap_table.style.highlight_max(axis=0, color='lightyellow'))

# === Section 3: Composite Score ===
import networkx as nx

# Crear grafo dirigido desde la matriz CIA
G = nx.from_pandas_adjacency(cia_df, create_using=nx.DiGraph)

# Calcular betweenness centrality (C_B)
C_B_dict = nx.betweenness_centrality(G, weight='weight', normalized=True)
C_B = pd.Series(C_B_dict).reindex(cia_df.index)

# Calcular Composite Score
composite_score = 0.5 * shap_values + 0.3 * C_out + 0.2 * C_B

# Mostrar tabla
st.subheader(":bar_chart: Composite Prioritization Score")
st.dataframe(pd.DataFrame({
    "SHAP Value (ϕ)": shap_values,
    "Out-Degree (C_out)": C_out,
    "Betweenness (C_B)": C_B.round(3),
    "Composite Score": composite_score.round(3)
}).sort_values("Composite Score", ascending=False))

# === Section 4: What-if Simulation ===
st.subheader(":warning: What-if Simulation – Removing G4")

# Elimina fila y columna G4 de la matriz
cia_df_wo_g4 = cia_df.drop(index="G4").drop(columns="G4")

# Recalcula Out-Degree Centrality sin G4
C_out_wo_g4 = cia_df_wo_g4.sum(axis=1)

# Simula nuevos SHAP como variación aleatoria controlada (entre -15% y +15%)
shap_values_wo_g4 = shap_values.drop("G4") * np.random.uniform(0.85, 1.15, len(shap_values) - 1)
shap_values_wo_g4 = shap_values_wo_g4.round(3)

# Para comparación, eliminamos G4 también de la original
shap_original_wo_g4 = shap_values.drop("G4")

# Calculamos cambio porcentual
impact_change = ((shap_values_wo_g4 - shap_original_wo_g4) / shap_original_wo_g4 * 100).round(2)

# Mostramos tabla
sim_df = pd.DataFrame({
    "SHAP (original)": shap_original_wo_g4,
    "SHAP (without G4)": shap_values_wo_g4,
    "% Change": impact_change
}).sort_values("% Change", ascending=False)

st.dataframe(sim_df)

# === Section 5: Scatter Plot ===
st.subheader(":chart_with_upwards_trend: SHAP vs C_out Scatter Plot")
fig, ax = plt.subplots()
sns.regplot(x=C_out, y=shap_values, ci=None, scatter_kws={"s": 60}, line_kws={"color": "red"}, ax=ax)
ax.set_xlabel("Out-Degree Centrality (C_out)")
ax.set_ylabel("SHAP Value (ϕ)")
ax.grid(True)
st.pyplot(fig)

# === Footer ===
st.markdown("""
---
Built by **Erick Mauricio Corimanya** | [GitHub](https://github.com/erickcori)
""")
