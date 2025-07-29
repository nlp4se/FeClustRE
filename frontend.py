import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
from typing import Dict, Any, Optional

# Configuration
BASE_URL = 'http://localhost:3000'  # Backend host and port

st.set_page_config(page_title="Feature Extraction & App Taxonomy", layout="wide")
st.title("FeClustRE - Feature Extraction & Mobile App Feature Taxonomy generation")

# Initialize session state
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = None
if 'selected_app' not in st.session_state:
    st.session_state.selected_app = None
if 'clustering_saved' not in st.session_state:
    st.session_state.clustering_saved = {}
if 'taxonomy_results' not in st.session_state:
    st.session_state.taxonomy_results = {}


def clean_data_for_json(obj: Any) -> Any:
    """Clean data to make it JSON serializable"""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    elif isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [clean_data_for_json(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: clean_data_for_json(v) for k, v in obj.items()}
    return obj


def make_request(method: str, url: str, **kwargs) -> requests.Response:
    """Make HTTP request with error handling"""
    try:
        response = requests.request(method, url, timeout=120, **kwargs)
        return response
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server might be busy processing your request.")
        raise
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend at {BASE_URL}. Please ensure the server is running.")
        raise
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        raise


def display_candidate_metrics(summary: Dict[str, Any]) -> str:
    """Format candidate metrics for display"""
    metrics = summary.get('metrics', {})
    n_clusters = summary.get('n_clusters', 0)
    avg_size = summary.get('avg_cluster_size', 0)
    threshold = summary.get('threshold', 0)

    silhouette = metrics.get('silhouette_score', 0)
    davies_bouldin = metrics.get('davies_bouldin_score', 0)

    return f"""
    **Clusters:** {n_clusters} | **Avg Size:** {avg_size} | **Threshold:** {threshold:.3f}

    **Silhouette Score:** {silhouette:.3f} (higher is better)
    **Davies-Bouldin Score:** {davies_bouldin:.3f} (lower is better)
    """


def display_cluster_preview(clustering: Dict[str, Any], max_clusters: int = 3) -> None:
    """Display a preview of clusters"""
    clusters = clustering.get('clusters', {})
    if not clusters:
        st.warning("No clusters found in this candidate.")
        return

    st.write("**Cluster Preview:**")
    for i, (cluster_id, features) in enumerate(list(clusters.items())[:max_clusters]):
        with st.expander(f"Cluster {cluster_id} ({len(features)} features)"):
            if len(features) <= 10:
                st.write(", ".join(features))
            else:
                st.write(", ".join(features[:10]) + f"... (+{len(features) - 10} more)")


# Sidebar: Configuration
st.sidebar.header("Configuration")

# Model selection
model_option = st.sidebar.selectbox(
    "Feature extraction model",
    ["tfrex", "transfeatex"],
    index=0,
    help="Choose 'tfrex' for local extraction or 'transfeatex' for remote API."
)

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (with required columns)",
    type=["csv"],
    help="CSV must contain: app_name, review, app_package, app_categoryId, reviewId, score"
)

# Backend health check
with st.sidebar.expander("Backend Status"):
    if st.button("Check Backend Health"):
        try:
            health_resp = make_request("GET", f"{BASE_URL}/health")
            if health_resp.ok:
                health_data = health_resp.json()
                if health_data.get('status') == 'healthy':
                    st.success("✅ Backend is healthy")
                else:
                    st.warning("⚠️ Backend has issues")
                    st.json(health_data)
            else:
                st.error(f"❌ Backend unhealthy: {health_resp.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot reach backend: {str(e)}")

# Main content
if uploaded_file:
    # Display uploaded data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.subheader("📊 Uploaded Data Preview")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(), use_container_width=True)

    with col2:
        st.metric("Total Rows", len(df))
        st.metric("Apps", df['app_name'].nunique() if 'app_name' in df.columns else 0)
        st.write("**Columns:**", df.columns.tolist())

    # Validate required columns
    required_cols = {'app_name', 'review', 'app_package', 'app_categoryId', 'reviewId', 'score'}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.info(f"Required columns: {required_cols}")
        st.stop()

    # Process data
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🚀 Run Feature Extraction & Clustering", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Uploading data...")
            progress_bar.progress(25)

            # Reset session state for new processing
            st.session_state.clustering_saved = {}
            st.session_state.taxonomy_results = {}

            uploaded_file.seek(0)
            files = {'file': (uploaded_file.name, uploaded_file, 'text/csv')}
            params = {'model_type': model_option}

            try:
                status_text.text("Processing features and generating clusters...")
                progress_bar.progress(50)

                resp = make_request("POST", f"{BASE_URL}/process_reviews/upload", params=params, files=files)

                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")

                if resp.ok:
                    data = resp.json()
                    if data.get('status') == 'success':
                        st.session_state.processing_results = data.get('results', {})
                        st.success("🎉 Feature extraction and clustering completed successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Backend error: {data.get('error', 'Unknown error')}")
                else:
                    st.error(f"❌ HTTP {resp.status_code}: {resp.text}")

            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()

    with col2:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.processing_results = None
            st.session_state.selected_app = None
            st.session_state.clustering_saved = {}
            st.session_state.taxonomy_results = {}
            st.rerun()

# Display results if available
if st.session_state.processing_results:
    st.divider()
    st.subheader("📈 Processing Results")

    # App selection
    app_names = list(st.session_state.processing_results.keys())

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_app = st.selectbox(
            "Select app to analyze:",
            app_names,
            key="app_selector"
        )
        st.session_state.selected_app = selected_app

    with col2:
        # Show app stats
        if selected_app:
            result = st.session_state.processing_results[selected_app]
            st.metric("Reviews Processed", result.get('processed_reviews', 0))
            st.metric("Unique Features", result.get('unique_features', 0))

    if selected_app:
        result = st.session_state.processing_results[selected_app]

        # Top features
        with st.expander("🔍 Top Features Found", expanded=False):
            top_features = result.get('top_features', {})
            if top_features:
                top_df = pd.DataFrame([
                    {"Feature": feature, "Frequency": count}
                    for feature, count in top_features.items()
                ])
                st.dataframe(top_df, use_container_width=True)
            else:
                st.info("No top features available.")

        # Clustering candidates
        candidates = result.get("clustering_results", {}).get("candidates", [])

        if not candidates:
            st.warning("⚠️ No clustering candidates available for this app.")
        else:
            st.subheader("🎯 Clustering Candidates")
            st.write(f"Found {len(candidates)} clustering options. Select the best one:")

            # Create tabs for candidates
            tab_names = [f"Option {i + 1}" for i in range(min(len(candidates), 3))]
            if len(candidates) > 3:
                tab_names.append("More...")

            tabs = st.tabs(tab_names)

            for tab_idx, tab in enumerate(tabs):
                with tab:
                    if tab_idx < len(candidates):
                        candidate = candidates[tab_idx]
                        summary = candidate.get("summary", {})
                        clustering = candidate.get("clustering", {})

                        # Display metrics
                        st.markdown(display_candidate_metrics(summary))

                        # Display cluster preview
                        display_cluster_preview(clustering)

                        # Check if this candidate was already saved
                        candidate_key = f"{selected_app}_{tab_idx}"
                        is_saved = st.session_state.clustering_saved.get(candidate_key, False)

                        if is_saved:
                            st.success("✅ This clustering has been saved and taxonomy generated!")

                            # Show taxonomy results if available
                            if candidate_key in st.session_state.taxonomy_results:
                                taxonomy_result = st.session_state.taxonomy_results[candidate_key]
                                with st.expander("📋 Taxonomy Generation Results"):
                                    st.json(taxonomy_result)
                        else:
                            # Save button
                            button_key = f"save_candidate_{selected_app}_{tab_idx}"
                            if st.button(
                                    f"💾 Save This Clustering & Generate Taxonomy",
                                    key=button_key,
                                    type="primary",
                                    use_container_width=True
                            ):
                                with st.spinner("Saving clustering and generating semantic taxonomy..."):
                                    try:
                                        # Clean the clustering data
                                        cleaned_clustering = clean_data_for_json(clustering)

                                        # Save the clustering
                                        save_resp = make_request(
                                            "POST",
                                            f"{BASE_URL}/save_selected_clustering/{selected_app}",
                                            json={"clustering": cleaned_clustering}
                                        )

                                        if save_resp.ok:
                                            taxonomy_result = save_resp.json()

                                            # Update session state
                                            st.session_state.clustering_saved[candidate_key] = True
                                            st.session_state.taxonomy_results[candidate_key] = taxonomy_result

                                            st.success("🎉 Clustering saved and taxonomy generated successfully!")

                                            merge_results = taxonomy_result.get('merge_results', {})
                                            merged_count = merge_results.get('merged_count', 0)

                                            if merged_count > 0:
                                                merges = merge_results.get('merges', [])
                                                with st.expander(
                                                        f"🔄 {merged_count} Mini Taxonomies Were Automatically Merged",
                                                        expanded=True):
                                                    merge_df = pd.DataFrame([
                                                        {
                                                            "Merged Taxonomy": merge['merged_taxonomies'][0],
                                                            "Into Taxonomy": merge['into_taxonomy'],
                                                            "Similarity": f"{merge['similarity']:.4f}"
                                                        }
                                                        for merge in merges
                                                    ])
                                                    st.dataframe(merge_df, use_container_width=True)
                                            else:
                                                st.info("✅ No similar taxonomies needed merging.")

                                            st.json(taxonomy_result)

                                            # Rerun to update the UI
                                            time.sleep(1)
                                            st.rerun()

                                        else:
                                            error_msg = save_resp.text
                                            try:
                                                error_data = save_resp.json()
                                                error_msg = error_data.get('error', error_msg)
                                            except:
                                                pass
                                            st.error(f"❌ Failed to save clustering: {error_msg}")

                                    except Exception as e:
                                        st.error(f"❌ Error saving clustering: {str(e)}")

                    elif tab_idx == len(tabs) - 1 and len(candidates) > 3:
                        # "More..." tab
                        st.write("Additional clustering candidates:")
                        for i in range(3, len(candidates)):
                            candidate = candidates[i]
                            summary = candidate.get("summary", {})
                            with st.expander(f"Candidate {i + 1}"):
                                st.markdown(display_candidate_metrics(summary))

# LLM Taxonomy Metrics
if st.session_state.processing_results:
    st.divider()
    if st.button("📊 View LLM Taxonomy Metrics"):
        with st.spinner("Fetching taxonomy metrics..."):
            try:
                metrics_resp = make_request("GET", f"{BASE_URL}/llm_taxonomy_metrics")

                if metrics_resp.ok:
                    metrics_data = metrics_resp.json()

                    st.subheader("🏷️ LLM Taxonomy Analysis")

                    # Overview metrics
                    overview = metrics_data.get('overview', {})
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Taxonomies", overview.get('total_taxonomies', 0))
                    with col2:
                        st.metric("Distinct Tags", overview.get('distinct_tags', 0))
                    with col3:
                        st.metric("Duplicate Tags", overview.get('duplicate_tags', 0))
                    with col4:
                        st.metric("Low Quality", overview.get('low_quality_count', 0))

                    # Tag statistics
                    with st.expander("📈 Tag Statistics"):
                        tag_stats = metrics_data.get('tag_statistics', {})

                        # Most common tags
                        most_common = tag_stats.get('most_common_tags', [])
                        if most_common:
                            st.write("**Most Common Tags:**")
                            common_df = pd.DataFrame(most_common, columns=['Tag', 'Count'])
                            st.dataframe(common_df, use_container_width=True)

                        # Duplicate tags
                        duplicates = tag_stats.get('duplicate_tags', {})
                        if duplicates:
                            st.write("**Duplicate Tags:**")
                            dup_df = pd.DataFrame([
                                {"Tag": tag, "Count": count}
                                for tag, count in duplicates.items()
                            ])
                            st.dataframe(dup_df, use_container_width=True)

                    # Structure analysis
                    with st.expander("🏗️ Structure Analysis"):
                        structure = metrics_data.get('structure_analysis', {})

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Average Depth", structure.get('avg_depth', 0))
                            st.metric("Average Leaves", structure.get('avg_leaves', 0))

                        with col2:
                            singletons = len(structure.get('singleton_taxonomies', []))
                            large = len(structure.get('large_taxonomies', []))
                            st.metric("Singleton Taxonomies", singletons)
                            st.metric("Large Taxonomies", large)

                    # Similarity analysis
                    similarity = metrics_data.get('similarity_analysis', {})
                    if similarity:
                        with st.expander("🔗 Similarity Analysis"):
                            for threshold, analysis in similarity.items():
                                st.write(f"**{threshold.replace('_', ' ').title()}:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Groups Found", analysis.get('total_groups', 0))
                                    st.metric("Tags in Groups", analysis.get('tags_in_groups', 0))
                                with col2:
                                    st.metric("Singleton Tags", analysis.get('singleton_tags', 0))

                    # Merge candidates
                    merge_candidates = metrics_data.get('merge_candidates', [])
                    if merge_candidates:
                        with st.expander("🔄 Merge Candidates"):
                            st.write("Tags with high similarity that could be merged:")
                            merge_df = pd.DataFrame(merge_candidates[:10])
                            st.dataframe(merge_df, use_container_width=True)

                else:
                    st.error(f"Failed to fetch taxonomy metrics: {metrics_resp.status_code}")

            except Exception as e:
                st.error(f"Error fetching taxonomy metrics: {str(e)}")

else:
    # Welcome message
    st.info("👋 Welcome! Upload a CSV file to get started with feature extraction and taxonomy generation.")

    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        1. **Upload CSV**: Use the sidebar to upload a CSV file with review data
        2. **Select Model**: Choose between 'tfrex' (local) or 'transfeatex' (API) for feature extraction
        3. **Process Data**: Click 'Run Feature Extraction & Clustering' to analyze your data
        4. **Review Candidates**: Examine different clustering options and their metrics
        5. **Save Best Option**: Select and save the clustering that best fits your needs
        6. **View Taxonomy**: See the generated semantic taxonomy for your selected clustering

        **Required CSV Columns:**
        - `app_name`: Name of the application
        - `review`: Review text content
        - `app_package`: Application package identifier
        - `app_categoryId`: Application category
        - `reviewId`: Unique review identifier  
        - `score`: Review rating/score
        """)