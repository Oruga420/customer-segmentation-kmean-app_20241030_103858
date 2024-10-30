import streamlit as st
import pandas as pd
import numpy as np
from utils import perform_kmeans, create_cluster_plot, create_radar_plot, generate_cluster_statistics
import plotly.express as px
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Customer Segmentation Analysis")
st.markdown("""
This application helps you segment your customers using K-means clustering.
Upload your customer data and discover meaningful customer segments!
""")

# File upload section
st.subheader("1. Data Upload")
uploaded_file = st.file_uploader(
    "Upload your CSV file (required columns: AnnualSpend, VisitsPerMonth, AveragePurchaseValue)",
    type=['csv']
)

# Sample data option
if st.checkbox("Use sample data instead"):
    uploaded_file = "assets/sample_data.csv"

if uploaded_file:
    try:
        # Read data
        if isinstance(uploaded_file, str):
            df = pd.read_csv(uploaded_file)
        else:
            string_data = StringIO(uploaded_file.getvalue().decode('utf-8'))
            df = pd.read_csv(string_data)

        # Validate required columns
        required_columns = ['AnnualSpend', 'VisitsPerMonth', 'AveragePurchaseValue']
        if not all(col in df.columns for col in required_columns):
            st.error("Missing required columns. Please ensure your data contains: AnnualSpend, VisitsPerMonth, AveragePurchaseValue")
            st.stop()

        # Display raw data
        st.subheader("2. Data Preview")
        st.dataframe(df.head())

        # Clustering parameters
        st.subheader("3. Clustering Parameters")
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
        
        # Perform clustering
        features = df[required_columns]
        labels, centers, scaled_features = perform_kmeans(features, n_clusters)
        df['Cluster'] = labels

        # Visualizations
        st.subheader("4. Cluster Visualization")
        
        # 2D scatter plot
        col1, col2 = st.columns(2)
        with col1:
            scatter_plot = create_cluster_plot(df, scaled_features, labels, centers)
            st.plotly_chart(scatter_plot, use_container_width=True)
        
        # Radar plot
        with col2:
            radar_plot = create_radar_plot(df, required_columns, labels)
            st.plotly_chart(radar_plot, use_container_width=True)

        # Cluster statistics
        st.subheader("5. Cluster Statistics")
        stats_df = generate_cluster_statistics(df, required_columns, labels)
        st.dataframe(stats_df)

        # Download segmented data
        st.subheader("6. Download Results")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download segmented data as CSV",
            data=csv,
            file_name="segmented_customers.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
