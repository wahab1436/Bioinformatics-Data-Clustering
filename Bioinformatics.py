import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Bioinformatics K-Means Clustering", layout="wide")

# Title
st.title("Bioinformatics K-Means Clustering ")
st.markdown("Cluster bioinformatics data .")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Step 1: Dataset Preview")
    st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(10))
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    st.write(f"Numeric columns found: {len(numeric_cols)}")
    st.write(f"Non-numeric columns (metadata): {len(non_numeric_cols)}")
    
    if len(numeric_cols) == 0:
        st.error("No numeric columns found for clustering. Please check your dataset.")
    else:
        # Step 2: Clean data
        st.subheader("Step 2: Data Cleaning")
        df_clean = df.dropna()
        st.write(f"Rows after removing missing values: {df_clean.shape[0]}")
        
        # Prepare data for clustering
        X = df_clean[numeric_cols].values
        
        # Step 3: K-Means Clustering
        st.subheader("Step 3: K-Means Clustering")
        k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)
        
        if st.button("Run Clustering"):
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            df_clean['Cluster'] = clusters
            
            # Store in session state
            st.session_state['df_clean'] = df_clean
            st.session_state['kmeans'] = kmeans
            st.session_state['X_scaled'] = X_scaled
            st.session_state['numeric_cols'] = numeric_cols
            st.session_state['k'] = k
            
        # Check if clustering has been run
        if 'df_clean' in st.session_state:
            df_clean = st.session_state['df_clean']
            kmeans = st.session_state['kmeans']
            X_scaled = st.session_state['X_scaled']
            numeric_cols = st.session_state['numeric_cols']
            k = st.session_state['k']
            
            st.success(f"Clustering complete! Created {k} clusters.")
            
            # Display cluster distribution
            cluster_counts = df_clean['Cluster'].value_counts().sort_index()
            st.write("Cluster distribution:")
            st.bar_chart(cluster_counts)
            
            # Step 4: PCA Visualization
            st.subheader("Step 4: PCA Visualization")
            
            # 2D PCA
            pca_2d = PCA(n_components=2)
            X_pca_2d = pca_2d.fit_transform(X_scaled)
            df_clean['PCA1'] = X_pca_2d[:, 0]
            df_clean['PCA2'] = X_pca_2d[:, 1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("2D PCA Scatter Plot")
                fig, ax = plt.subplots(figsize=(8, 6))
                for cluster in range(k):
                    cluster_data = df_clean[df_clean['Cluster'] == cluster]
                    ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
                             label=f'Cluster {cluster}', alpha=0.6, s=50)
                ax.set_xlabel('First Principal Component')
                ax.set_ylabel('Second Principal Component')
                ax.set_title('2D PCA Clustering Visualization')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # 3D PCA
            with col2:
                if X_scaled.shape[1] >= 3:
                    st.write("3D PCA Scatter Plot")
                    pca_3d = PCA(n_components=3)
                    X_pca_3d = pca_3d.fit_transform(X_scaled)
                    df_clean['PCA3'] = X_pca_3d[:, 2]
                    
                    fig_3d = px.scatter_3d(df_clean, x='PCA1', y='PCA2', z='PCA3',
                                          color='Cluster', 
                                          labels={'PCA1': 'PC1', 'PCA2': 'PC2', 'PCA3': 'PC3'},
                                          title='3D PCA Clustering Visualization')
                    fig_3d.update_traces(marker=dict(size=5))
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.info("Not enough features for 3D visualization. Need at least 3 numeric columns.")
            
            # Step 5: Automatic Insights
            st.subheader("Step 5: Valuable Insights")
            
            insights = []
            
            # Overall summary
            insights.append(f"Your data has been organized into {k} distinct groups based on patterns in the gene expression data.")
            insights.append("")
            
            # Calculate cluster statistics
            cluster_stats = {}
            for cluster in range(k):
                cluster_data = df_clean[df_clean['Cluster'] == cluster][numeric_cols]
                cluster_stats[cluster] = {
                    'count': len(cluster_data),
                    'means': cluster_data.mean(),
                    'percent': (len(cluster_data) / len(df_clean)) * 100
                }
            
            # Generate insights for each cluster
            for cluster in range(k):
                stats = cluster_stats[cluster]
                insights.append(f"Cluster {cluster}:")
                insights.append(f"- Contains {stats['count']} samples, which is {stats['percent']:.1f}% of your data.")
                
                # Find distinctive features
                mean_differences = {}
                for col in numeric_cols[:5]:  # Focus on first 5 genes
                    cluster_mean = stats['means'][col]
                    other_clusters_mean = np.mean([cluster_stats[c]['means'][col] 
                                                   for c in range(k) if c != cluster])
                    mean_differences[col] = cluster_mean - other_clusters_mean
                
                # Top 2 most distinctive features
                sorted_diffs = sorted(mean_differences.items(), key=lambda x: abs(x[1]), reverse=True)
                
                if len(sorted_diffs) > 0:
                    top_feature = sorted_diffs[0]
                    if top_feature[1] > 0:
                        insights.append(f"- This group shows higher levels of {top_feature[0]} compared to other clusters.")
                    else:
                        insights.append(f"- This group shows lower levels of {top_feature[0]} compared to other clusters.")
                
                if len(sorted_diffs) > 1:
                    second_feature = sorted_diffs[1]
                    if second_feature[1] > 0:
                        insights.append(f"- Also has elevated {second_feature[0]} expression.")
                    else:
                        insights.append(f"- Also has reduced {second_feature[0]} expression.")
                
                insights.append("")
            
            # Cluster similarity
            insights.append("Cluster Relationships:")
            cluster_centers = kmeans.cluster_centers_
            min_distance = float('inf')
            most_similar = None
            
            for i in range(k):
                for j in range(i+1, k):
                    distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    if distance < min_distance:
                        min_distance = distance
                        most_similar = (i, j)
            
            if most_similar:
                insights.append(f"- Clusters {most_similar[0]} and {most_similar[1]} are the most similar to each other in terms of gene expression patterns.")
            
            # Display insights
            for insight in insights:
                st.write(insight)
            
            # Step 6: Optional Gene Visualization (FIXED)
            st.subheader("Step 6: Individual Gene Analysis")
            
            if len(numeric_cols) > 0:
                selected_gene = st.selectbox("Select a gene to visualize", numeric_cols)
                
                if selected_gene:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Histogram")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot histogram for each cluster
                        for cluster in range(k):
                            cluster_data = df_clean[df_clean['Cluster'] == cluster][selected_gene]
                            ax.hist(cluster_data, alpha=0.5, label=f'Cluster {cluster}', bins=20)
                        
                        ax.set_xlabel(f'{selected_gene} Expression')
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {selected_gene} across Clusters')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.write("Box Plot")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Prepare data for boxplot
                        data_for_boxplot = [df_clean[df_clean['Cluster'] == c][selected_gene].values 
                                           for c in range(k)]
                        
                        bp = ax.boxplot(data_for_boxplot, labels=[f'Cluster {c}' for c in range(k)],
                                       patch_artist=True)
                        
                        # Color the boxplots
                        colors = plt.cm.viridis(np.linspace(0, 1, k))
                        for patch, color in zip(bp['boxes'], colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.6)
                        
                        ax.set_xlabel('Cluster')
                        ax.set_ylabel(f'{selected_gene} Expression')
                        ax.set_title(f'{selected_gene} Expression by Cluster')
                        ax.grid(True, alpha=0.3, axis='y')
                        st.pyplot(fig)
                        plt.close()
                    
                    # Gene insight
                    st.write(f"\n**{selected_gene} Analysis:**")
                    cluster_means = [df_clean[df_clean['Cluster'] == c][selected_gene].mean() 
                                   for c in range(k)]
                    max_cluster = np.argmax(cluster_means)
                    min_cluster = np.argmin(cluster_means)
                    
                    st.write(f"- {selected_gene} is highest in Cluster {max_cluster} (mean: {cluster_means[max_cluster]:.3f}) and lowest in Cluster {min_cluster} (mean: {cluster_means[min_cluster]:.3f}).")
                    
                    # Additional statistics
                    max_diff = cluster_means[max_cluster] - cluster_means[min_cluster]
                    st.write(f"- The difference between highest and lowest cluster is {max_diff:.3f}.")
            
            # Download results
            st.subheader("Download Results")
            csv = df_clean.to_csv(index=False)
            st.download_button(
                label="Download clustered data as CSV",
                data=csv,
                file_name="clustered_data.csv",
                mime="text/csv"
            )

else:
    st.info("Please upload a CSV file to begin analysis.")
    st.markdown("""
    ### Expected File Format:
    - CSV file with numeric columns representing genes or features
    - Rows represent samples
    - Optional non-numeric columns for metadata
    - Missing values will be removed automatically
    """)
