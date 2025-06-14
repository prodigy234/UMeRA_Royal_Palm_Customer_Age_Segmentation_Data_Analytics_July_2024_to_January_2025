import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Investment Analytics Dashboard", layout="wide")
st.title("📊 Investment Analytics Dashboard")

# Load data
df = pd.read_excel("processed_investment_data.xlsx")

# Data preprocessing
columns_to_keep = ['DOB', 'INVESTMENT YEAR', 'INVESTMENT MONTH', 'LAND', 'UNIT', 'AMOUNT']
df = df[columns_to_keep].copy()
df = df.dropna(subset=['DOB', 'AMOUNT'])
df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
df = df.dropna(subset=['DOB'])
df['AGE'] = df['DOB'].apply(lambda x: 2025 - x.year if pd.notnull(x) else None)

month_order = ['JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER', 'JANUARY']
df['INVESTMENT MONTH'] = df['INVESTMENT MONTH'].astype(str).str.strip().str.upper()
df = df[df['INVESTMENT MONTH'].isin(month_order)]
df['INVESTMENT MONTH'] = pd.Categorical(df['INVESTMENT MONTH'], categories=month_order, ordered=True)

df['UNIT'] = pd.to_numeric(df['UNIT'], errors='coerce')
df['AMOUNT'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
df = df.dropna()

bins = [18, 30, 40, 50, 60, 70, 100]
labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
df["AGE GROUP"] = pd.cut(df["AGE"], bins=bins, labels=labels, right=False)
df = df[df["INVESTMENT MONTH"].notna()]

# Sidebar filters
st.sidebar.header("🔎 Filter Data")
selected_year = st.sidebar.selectbox("Select Investment Year", sorted(df["INVESTMENT YEAR"].unique()), index=0)
filtered_df = df[df["INVESTMENT YEAR"] == selected_year]

# Key Metrics
total_investment = filtered_df['AMOUNT'].sum()
total_units = filtered_df['UNIT'].sum()
total_investors = filtered_df['DOB'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Investment", f"₦{total_investment:,.0f}")
col2.metric("📦 Total Units", f"{total_units:,.0f}")
col3.metric("👥 Unique Investors", f"{total_investors}")

# Chart 1: Total Investment by Age Group
age_amount_df = filtered_df.groupby("AGE GROUP")["AMOUNT"].sum().reindex(labels).reset_index()
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x="AGE GROUP", y="AMOUNT", data=age_amount_df, palette="viridis", ax=ax1)
ax1.set_title("Total Investment by Age Group")
ax1.set_ylabel("Total Amount Paid (₦)")
ax1.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig1)

# Chart 2: Distribution of Land Type Purchased by Age Group
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.countplot(data=filtered_df, x="AGE GROUP", hue="LAND", order=labels, palette="coolwarm", ax=ax2)
ax2.set_title("Distribution of Land Type Purchased by Age Group")
ax2.set_ylabel("Count of Investors")
ax2.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig2)

# Chart 3: Total Units Purchased by Age Group
age_units_df = filtered_df.groupby("AGE GROUP")["UNIT"].sum().reindex(labels).reset_index()
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(x="AGE GROUP", y="UNIT", data=age_units_df, palette="magma", ax=ax3)
ax3.set_title("Total Units Purchased by Age Group")
ax3.set_ylabel("Total Units Purchased")
ax3.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig3)

# Chart 4: Heatmap of Investment Amount by Age Group and Month
heatmap_data = filtered_df.pivot_table(index="INVESTMENT MONTH", columns="AGE GROUP", values="AMOUNT", aggfunc="sum", fill_value=0)
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm", linewidths=0.5, ax=ax4)
ax4.set_title("Heatmap of Investment Amount by Age Group and Month")
st.pyplot(fig4)

# Chart 5: Stacked Bar Chart: Investment by Age Group and Month
df_grouped = filtered_df.groupby(["INVESTMENT MONTH", "AGE GROUP"])["AMOUNT"].sum().unstack().reindex(columns=labels)
fig5, ax5 = plt.subplots(figsize=(12, 6))
df_grouped.plot(kind="bar", stacked=True, colormap="viridis", ax=ax5)
ax5.set_title("Stacked Bar Chart: Investment by Age Group and Month")
ax5.set_xlabel("Investment Month")
ax5.set_ylabel("Total Investment Amount")
ax5.legend(title="Age Group")
ax5.set_xticklabels(df_grouped.index, rotation=45)
st.pyplot(fig5)

# Chart 6: Grouped Bar Chart: Investment by Age Group and Month
fig6, ax6 = plt.subplots(figsize=(12, 6))
df_grouped.plot(kind="bar", stacked=False, colormap="coolwarm", ax=ax6)
ax6.set_title("Grouped Bar Chart: Investment by Age Group and Month")
ax6.set_xlabel("Investment Month")
ax6.set_ylabel("Total Investment Amount")
ax6.legend(title="Age Group")
ax6.set_xticklabels(df_grouped.index, rotation=45)
st.pyplot(fig6)

# Chart 7: Bubble Chart: Investment Amount by Age Group and Month
fig7, ax7 = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=filtered_df, x="INVESTMENT MONTH", y="AGE GROUP", size="AMOUNT", hue="AGE GROUP", sizes=(20, 1000), palette="coolwarm", alpha=0.6, ax=ax7)
ax7.set_title("Bubble Chart: Investment Amount by Age Group and Month")
ax7.set_xlabel("Investment Month")
ax7.set_ylabel("Age Group")
ax7.legend(title="Investment Amount", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig7)

# Chart 8: Trendline: Investment Trend by Age Group Over Time
fig8, ax8 = plt.subplots(figsize=(12, 6))
sns.lineplot(data=filtered_df, x="INVESTMENT MONTH", y="AMOUNT", hue="AGE GROUP", marker="o", ax=ax8)
ax8.set_title("Investment Trend by Age Group Over Time")
ax8.set_xlabel("Investment Month")
ax8.set_ylabel("Total Investment Amount")
ax8.legend(title="Age Group")
ax8.grid(True)
st.pyplot(fig8)

# Chart 9: Cross-tabulation Heatmap: Age Group vs Investment Month
age_month_ct = pd.crosstab(filtered_df["INVESTMENT MONTH"], filtered_df["AGE GROUP"])
fig9, ax9 = plt.subplots(figsize=(10, 6))
sns.heatmap(age_month_ct, annot=True, cmap="coolwarm", linewidths=0.5, fmt="d", ax=ax9)
ax9.set_title("Investment Distribution Across Months and Age Groups")
ax9.set_xlabel("Age Group")
ax9.set_ylabel("Investment Month")
st.pyplot(fig9)

# Display unique months
st.sidebar.markdown("---")
st.sidebar.markdown("**Unique Investment Months:**")
st.sidebar.write(df["INVESTMENT MONTH"].unique())
