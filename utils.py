import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

def perform_kmeans(features, n_clusters):
    """
    Perform K-means clustering on the input features
    """
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    
    # Get cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    return labels, centers, scaled_features

def create_cluster_plot(df, scaled_features, labels, centers):
    """
    Create a 2D scatter plot of clusters using PCA
    """
    from sklearn.decomposition import PCA
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame(features_2d, columns=['Component 1', 'Component 2'])
    plot_df['Cluster'] = labels
    
    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x='Component 1',
        y='Component 2',
        color='Cluster',
        title='Customer Segments Visualization',
        template='plotly_white'
    )
    
    return fig

def create_radar_plot(df, features, labels):
    """
    Create a radar plot showing cluster characteristics
    """
    # Calculate mean values for each cluster
    cluster_means = df.groupby('Cluster')[features].mean()
    
    # Create radar plot
    fig = go.Figure()
    
    for cluster in cluster_means.index:
        fig.add_trace(go.Scatterpolar(
            r=cluster_means.loc[cluster],
            theta=features,
            name=f'Cluster {cluster}'
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Cluster Characteristics'
    )
    
    return fig

def generate_cluster_statistics(df, features, labels):
    """
    Generate summary statistics for each cluster
    """
    stats = []
    
    for cluster in range(len(set(labels))):
        cluster_data = df[df['Cluster'] == cluster]
        
        cluster_stats = {
            'Cluster': cluster,
            'Size': len(cluster_data),
            'Percentage': f"{(len(cluster_data) / len(df) * 100):.1f}%"
        }
        
        # Calculate statistics for each feature
        for feature in features:
            cluster_stats[f'{feature} (Mean)'] = f"{cluster_data[feature].mean():.2f}"
            cluster_stats[f'{feature} (Std)'] = f"{cluster_data[feature].std():.2f}"
        
        stats.append(cluster_stats)
    
    return pd.DataFrame(stats)

