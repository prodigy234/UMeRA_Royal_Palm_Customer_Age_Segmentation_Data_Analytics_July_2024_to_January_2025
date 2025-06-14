import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Investment Analytics Dashboard", layout="wide")
st.title("📊 Investment Insights Dashboard")

# Load data
df = pd.read_excel("processed_investment_data.xlsx")

# Clean and process data
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
st.sidebar.header("Filter Options")
selected_age_groups = st.sidebar.multiselect("Select Age Groups", options=df["AGE GROUP"].unique(), default=df["AGE GROUP"].unique())
selected_land_types = st.sidebar.multiselect("Select Land Types", options=df["LAND"].unique(), default=df["LAND"].unique())

filtered_df = df[df["AGE GROUP"].isin(selected_age_groups) & df["LAND"].isin(selected_land_types)]

# Metrics
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Investment (₦)", f"{filtered_df['AMOUNT'].sum():,.0f}")
col2.metric("Total Units Purchased", f"{filtered_df['UNIT'].sum():,.0f}")
col3.metric("Unique Investors", f"{filtered_df['DOB'].nunique()}")

# Bar chart: Total Amount by Age Group
st.subheader("💸 Total Investment by Age Group")
age_amount_df = filtered_df.groupby("AGE GROUP")["AMOUNT"].sum().reset_index()
fig1 = px.bar(age_amount_df, x="AGE GROUP", y="AMOUNT", color="AGE GROUP", title="Total Investment by Age Group", text_auto=True)
st.plotly_chart(fig1, use_container_width=True)

# Land type distribution
st.subheader("🌍 Land Type Distribution by Age Group")
fig2 = px.histogram(filtered_df, x="AGE GROUP", color="LAND", barmode="group", title="Land Type Distribution by Age Group")
st.plotly_chart(fig2, use_container_width=True)

# Units purchased by age group
st.subheader("📦 Units Purchased by Age Group")
age_units_df = filtered_df.groupby("AGE GROUP")["UNIT"].sum().reset_index()
fig3 = px.bar(age_units_df, x="AGE GROUP", y="UNIT", color="AGE GROUP", text_auto=True, title="Total Units Purchased by Age Group")
st.plotly_chart(fig3, use_container_width=True)

# Pie chart
st.subheader("🥧 Investment Distribution by Age Group")
fig4 = px.pie(age_amount_df, values="AMOUNT", names="AGE GROUP", title="Proportion of Investment by Age Group")
st.plotly_chart(fig4, use_container_width=True)

# Trend over time
st.subheader("📈 Investment Trend Over Time")
monthly_investment = filtered_df.groupby("INVESTMENT MONTH")["AMOUNT"].sum().reset_index()
fig5 = px.line(monthly_investment, x="INVESTMENT MONTH", y="AMOUNT", markers=True, title="Monthly Investment Trend")
st.plotly_chart(fig5, use_container_width=True)

# Trend by age group
st.subheader("📉 Investment Trend by Age Group")
fig6 = px.line(filtered_df, x="INVESTMENT MONTH", y="AMOUNT", color="AGE GROUP", markers=True, title="Investment Trend by Age Group")
st.plotly_chart(fig6, use_container_width=True)

# Additional charts
st.subheader("📊 Advanced Investment Visualizations")

# Heatmap: Investment Amount by Age Group
st.markdown("**Heatmap: Investment Amount by Age Group**")
age_investment_pivot = df.pivot_table(index="AGE GROUP", values="AMOUNT", aggfunc="sum")
fig, ax = plt.subplots()
sns.heatmap(age_investment_pivot, annot=True, fmt=".0f", cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig)

# Age group vs. month heatmap
st.markdown("**Investment Distribution Across Months and Age Groups**")
age_month_ct = pd.crosstab(df["INVESTMENT MONTH"], df["AGE GROUP"])
fig, ax = plt.subplots()
sns.heatmap(age_month_ct, annot=True, cmap="coolwarm", linewidths=0.5, fmt="d", ax=ax)
st.pyplot(fig)

# Heatmap of investment amount by age group and month
st.markdown("**Heatmap of Investment Amount by Age Group and Month**")
heatmap_data = df.pivot_table(index="INVESTMENT MONTH", columns="AGE GROUP", values="AMOUNT", aggfunc="sum", fill_value=0)
fig, ax = plt.subplots()
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig)

# Stacked bar chart
st.markdown("**Stacked Bar Chart: Investment by Age Group and Month**")
df_grouped = df.groupby(["INVESTMENT MONTH", "AGE GROUP"])["AMOUNT"].sum().unstack()
fig, ax = plt.subplots()
df_grouped.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
plt.title("Stacked Bar Chart: Investment by Age Group and Month")
plt.xlabel("Investment Month")
plt.ylabel("Total Investment Amount")
plt.xticks(rotation=45)
st.pyplot(fig)

# Grouped bar chart
st.markdown("**Grouped Bar Chart: Investment by Age Group and Month**")
fig, ax = plt.subplots()
df_grouped.plot(kind="bar", stacked=False, colormap="coolwarm", ax=ax)
plt.title("Grouped Bar Chart: Investment by Age Group and Month")
plt.xlabel("Investment Month")
plt.ylabel("Total Investment Amount")
plt.xticks(rotation=45)
st.pyplot(fig)

# Bubble Chart: Investment Amount by Age Group and Month
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=filtered_df, x="INVESTMENT MONTH", y="AGE GROUP", size="AMOUNT", hue="AGE GROUP", sizes=(20, 1000), palette="coolwarm", alpha=0.6, ax=ax)
ax.set_title("Bubble Chart: Investment Amount by Age Group and Month")
ax.set_xlabel("Investment Month")
ax.set_ylabel("Age Group")
ax.legend(title="Investment Amount", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit, Seaborn, Matplotlib and Plotly")
