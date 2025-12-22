import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

df = pd.read_csv("./data/data.csv")

def change(row):
    if row == "negative":
        row = 1
        return row 
    elif row == "positive":
        row = 2
        return row 
    else :
        row = 3
        return row

df["num"] = df["airline_sentiment"].apply(lambda x : change(x))

# data = df.groupby("num")["num"].count()

fig1, ax1 = plt.subplots()

sns.histplot(df["num"], ax=ax1)

st.pyplot(fig1)

st.dataframe(df)


fig2, ax2 = plt.subplots()

sns.countplot(x="num", data=df, ax=ax2)

st.pyplot(fig2)
