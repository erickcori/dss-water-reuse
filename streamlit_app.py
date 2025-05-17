import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
import shap

st.set_page_config(page_title="Decision Support System", layout="wide")

st.title("Decision Support System for Wastewater Reuse")
st.markdown("""
This tool integrates **Cross-Impact Analysis (CIA)**, **SHAP interpretation**, and **graph centrality metrics** to prioritize systemic barriers in water reuse policy.
""")

# === Load Data ===
cia_df = pd.read_csv("data/cia_matrix.csv", index_col=0)

# === Normalize CIA Matrix for SHAP ===
scaler = MinMaxScaler()
cia_normalized = pd.DataFrame(scaler.fit_transform(cia_df), columns=cia_df.columns, index=cia_df.index)

# === Create SHAP Target ===
y = cia_normalized.idxmax(axis=1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Train XGBoost Classifier ===
model = XGBClassifier(learning_rate=0.01, max_depth=3, use_label_encoder=False, eval_metric='mlogloss')
model.fit(cia_normalized, y_encoded)

# === Compute SHAP Values ===
explainer = shap.Explainer(model)
shap_values = explainer(cia_normalized)
shap_means = np.abs(shap_values.values).mean(axis=0)
shap_series = pd.Series(shap_means, index=cia_df.columns)

# === C_out ===
C_out = cia_df.sum(axis=1)

# === C_B ===
G = nx.from_pandas_adjacency(cia_df, create_using=nx.DiGraph)
C_B = pd.Series(nx.betweenness_centrality(G, weight='weight'), index=cia_df.index)

# === Composite Score ===
composite_score = 0.5 * shap_series + 0.3 * C_out + 0.2 * C_B

# === Section: CIA Matrix ===
st.subheader("🌐 Cross-Impact Matrix (CIA)")
st.dataframe(cia_df)

# === Section: SHAP and Centrality ===
st.subheader(":mag_right: SHAP Importance and Centrality Metrics")
shap_table = pd.DataFrame({
    "SHAP Value (ϕ)": shap_series,
    "Out-Degree (C_out)": C_out,
    "Betweenness (C_B)": C_B
})
st.dataframe(shap_table.style.highlight_max(axis=0, color='lightyellow'))

# === Section: Composite Score ===
st.subheader(":bar_chart: Composite Prioritization Score")
st.dataframe(pd.DataFrame({
    "SHAP Value (ϕ)": shap_series,
    "Out-Degree (C_out)": C_out,
    "Betweenness (C_B)": C_B,
    "Composite Score": composite_score
}).sort_values("Composite Score", ascending=False))

# === Section: SHAP vs C_out ===
st.subheader(":chart_with_upwards_trend: SHAP vs C_out Scatter Plot")
fig1, ax1 = plt.subplots()
sns.regplot(x=C_out, y=shap_series, ci=None, scatter_kws={"s": 60}, line_kws={"color": "red"}, ax=ax1)
ax1.set_xlabel("Out-Degree Centrality (C_out)")
ax1.set_ylabel("SHAP Value (ϕ)")
ax1.grid(True)
st.pyplot(fig1)

# === Section: What-if Simulation ===
st.subheader(":warning: What-if Simulation – Removing G4")
if "G4" in cia_df.index:
    cia_df_wo_g4 = cia_df.drop(index="G4", columns="G4")
    cia_norm_wo_g4 = pd.DataFrame(scaler.fit_transform(cia_df_wo_g4), index=cia_df_wo_g4.index, columns=cia_df_wo_g4.columns)
    y_wo = cia_norm_wo_g4.idxmax(axis=1)
    y_wo_enc = le.fit_transform(y_wo)
    model_wo = XGBoostClassifier(learning_rate=0.01, max_depth=3, use_label_encoder=False, eval_metric='mlogloss')
    model_wo.fit(cia_norm_wo_g4, y_wo_enc)
    shap_wo = shap.Explainer(model_wo)(cia_norm_wo_g4)
    shap_wo_mean = pd.Series(np.abs(shap_wo.values).mean(axis=0), index=cia_df_wo_g4.columns)

    shap_delta = ((shap_wo_mean - shap_series.drop("G4")) / shap_series.drop("G4") * 100).round(2)
    sim_df = pd.DataFrame({
        "SHAP (original)": shap_series.drop("G4"),
        "SHAP (without G4)": shap_wo_mean,
        "% Change": shap_delta
    }).sort_values("% Change", ascending=False)

    st.dataframe(sim_df)

# === Footer ===
st.markdown("""
---
Built by **Erick Mauricio Corimanya** | [GitHub](https://github.com/erickcori)
""")
