import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import pickle

@st.cache_data
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="diabetes.csv">Download CSV File</a>'
    return href

# Sidebar image path correction
st.sidebar.image("images_folder/photo1.jpg", width=200)

def main():
    st.markdown('<h1 style="text-align: center;color:brown">Streamlit Diabetes Prediction App</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center;color:blue">Predicting Diabetes using Machine Learning</h2>', unsafe_allow_html=True)
    
    # Menu options
    menu = ["Home", "Data Exploration", "Data Visualization", "Machine Learning"]
    choice = st.sidebar.selectbox("Menu", menu)  
    
    # Load dataset
    data = load_data("Dataset/diabetes.csv")
    
    if choice == "Home":
        left, middle, right = st.columns(3)
        with middle:
            st.image("images_folder/photo2.jpg", width=200)
        st.write("This is an app that will analyze diabetes data with some Python tools that can optimize decisions.")
        st.subheader("Diabetes Information")
        st.write("In Cameroon, the prevalence of diabetes in adults in urban areas is currently estimated at 6 – 8%, with as much as 80% of people living with diabetes who are currently undiagnosed. The burden of diabetes in Cameroon is rising rapidly.")

    elif choice == "Data Exploration":
        st.subheader("Diabetes Dataset")
        st.write(data.head())
        if st.checkbox("Show Summary"):
            st.write(data.describe())
        if st.checkbox("Show Correlation"):
            fig = plt.figure(figsize=(15, 15))
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
            st.pyplot(fig)

    elif choice == "Data Visualization":
        if st.checkbox("Count Plot"):
            fig = plt.figure(figsize=(9, 5))
            sns.countplot(data=data, x="Age")
            st.pyplot(fig)
        if st.checkbox("Scatter Plot"):
            fig = plt.figure(figsize=(8, 8))
            sns.scatterplot(x="Glucose", y="Age", data=data, hue="Outcome")
            st.pyplot(fig)
    
    elif choice == "Machine Learning":
        tab1, tab2, tab3 = st.tabs([":clipboard: Data", ":bar_chart: Visualisation", ":mask: :smile: Prediction"])
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = load_data(uploaded_file)
            with tab1:
                st.subheader("Input Data")
                st.write(df.head())
            with tab2:
                st.subheader("Histogram")
                fig = plt.figure(figsize=(9,9))
                sns.histplot(x="Glucose", data=df)
                st.pyplot(fig)
            with tab3:
                st.subheader("Prediction")
                model = pickle.load(open("model_dump.pkl", "rb"))
                prediction = model.predict(df)
                st.subheader("prediction")
                pp = pd.DataFrame(prediction, columns=["Prediction"])
                ndf = pd.concat([df, pp], axis=1)
                ndf.Prediction.replace(0, "No diabetes", inplace=True)
                ndf.Prediction.replace(1, "Diabetes", inplace=True)
                st.write(ndf)
                button = st.button("Download CSV")
                if button:
                    st.markdown(file_download(ndf), unsafe_allow_html=True)

if __name__ == "__main__":
    main()