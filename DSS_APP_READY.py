
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Carga de datos (matriz CIA y SHAP) ----
st.title("Decision Support System for Wastewater Reuse")

st.markdown("### 📊 Cross-Impact Matrix (CIA)")
cia_data = {
    "A1": [0, 0, 1.67, 2.5, 1.58, 2.0, 0, 2.25, 2.17, 0, 0, 0],
    "A2": [2.46, 0, 2.17, 2.5, 0, 2.0, 0, 2.25, 2.42, 2.67, 0, 0],
    "I1": [1.92, 0, 0, 2.58, 0, 0, 0, 2.17, 2.25, 2.75, 0, 0],
    "I2": [0, 0, 1.75, 0, 0, 0, 1.83, 2.17, 0, 2.5, 1.73, 0],
    "I5": [0, 0, 0, 2.25, 0, 0, 1.58, 1.92, 2.25, 2.36, 0, 0],
    "G1": [0, 0, 0, 0, 0, 0, 0, 2.08, 0, 0, 0, 0],
    "G2": [0, 0, 0, 0, 0, 1.58, 0, 1.92, 2.0, 0, 0, 0],
    "G4": [2.15, 0, 0, 0, 0, 2.5, 2.0, 0, 2.17, 0, 2.18, 1.82],
    "G5": [2.38, 0, 0, 0, 0, 2.25, 0, 2.17, 0, 0, 0, 0],
    "F1": [0, 2.31, 2.17, 0, 1.33, 0, 0, 0, 0, 0, 0, 0],
    "S1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.45, 0, 0],
    "S2": [0, 1.69, 1.25, 1.5, 0, 1.92, 0, 0, 0, 0, 0, 1.73]
}
labels = ["A1", "A2", "I1", "I2", "I5", "G1", "G2", "G4", "G5", "F1", "S1", "S2"]
cia_df = pd.DataFrame(cia_data, index=labels)
st.dataframe(cia_df)

# ---- 2. SHAP Values & C_out ----
st.markdown("### 🔍 SHAP Importance and Out-Degree Centrality")
shap_values = pd.Series(
    [0.215, 0.138, 0.112, 0.097, 0.091, 0.087, 0.075, 0.073, 0.064, 0.050, 0.046, 0.042],
    index=labels
)
c_out = cia_df.sum(axis=1)
df_scores = pd.DataFrame({
    "SHAP Value (ϕ)": shap_values,
    "Out-Degree (C_out)": c_out
})

st.dataframe(df_scores.style.highlight_max(axis=0))

# ---- 3. Composite Score ----
st.markdown("### 🧮 Composite Prioritization Score")
composite_score = 0.5 * shap_values + 0.3 * c_out + 0.2  # simplificado
df_scores["Composite Score"] = composite_score
st.dataframe(df_scores.sort_values("Composite Score", ascending=False))

# ---- 4. What-if Simulation ----
st.markdown("### ⚠️ What-if Simulation: Removing G4")
impact_if_removed = df_scores.copy()
impact_if_removed.loc["G4", "SHAP Value (ϕ)"] = 0
impact_if_removed.loc["G4", "Composite Score"] = 0
impact_if_removed["Impact Change (%)"] = 100 * (impact_if_removed["Composite Score"] - df_scores["Composite Score"]) / df_scores["Composite Score"]
st.dataframe(impact_if_removed.style.format({"Impact Change (%)": "{:.2f}"}))

# ---- 5. Plotting ----
st.markdown("### 📈 SHAP vs C_out Scatter Plot")
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
sns.regplot(x=c_out, y=shap_values, ax=ax, scatter_kws={"s": 60}, line_kws={"color": "red"})
ax.set_xlabel("Out-Degree Centrality (C_out)")
ax.set_ylabel("SHAP Value (ϕ)")
st.pyplot(fig)
