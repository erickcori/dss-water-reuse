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
st.subheader(":bar_chart: Composite Prioritization Score")
composite_score = 0.5*shap_values + 0.3*C_out + 0.2*np.random.rand(len(shap_values))  # Placeholder
st.dataframe(pd.DataFrame({"SHAP Value (ϕ)": shap_values, "Out-Degree (C_out)": C_out, "Composite Score": composite_score}).sort_values("Composite Score", ascending=False))

# === Section 4: What-if Simulation ===
st.subheader(":warning: What-if Simulation: Removing G4")
shap_without_G4 = shap_values.copy()
shap_without_G4["G4"] = 0
impact_change = 100 * (shap_without_G4 - shap_values) / shap_values

sim_df = pd.DataFrame({"SHAP (original)": shap_values, "SHAP (without G4)": shap_without_G4, "% Change": impact_change.round(1)})
st.dataframe(sim_df.sort_values("% Change"))

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