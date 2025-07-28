import streamlit as st
import pandas as pd
import numpy as np
import requests

# Configuration
BASE_URL = 'http://localhost:3000'  # Backend host and port

st.set_page_config(page_title="Feature Extraction & App Taxonomy", layout="wide")
st.title("Feature Extraction and App Taxonomy Generator")

# Sidebar: model selection and file upload
st.sidebar.header("Configuration")
model_option = st.sidebar.selectbox(
    "Feature extraction model",
    ["tfrex", "transfeatex"],
    index=0,
    help="Choose 'tfrex' for local extraction or 'transfeatex' for remote API."
)
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (with 'app_name' and 'review' columns)", type=["csv"]
)

# Main panel
if uploaded_file:
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.subheader("Uploaded Data Preview")
    st.dataframe(df)
    st.markdown(f"**Columns found:** {df.columns.tolist()}")

    # Validate required columns
    required_cols = {'app_name', 'review'}
    if not required_cols.issubset(set(df.columns)):
        st.error(
            f"CSV must contain the columns: {required_cols}.\n"
            f"Found: {set(df.columns)}"
        )
        st.stop()

    # Run pipeline
    if st.sidebar.button("Run Feature Extraction & Taxonomy"):
        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            files = {'file': (uploaded_file.name, uploaded_file, 'text/csv')}
            params = {'model_type': model_option}
            try:
                resp = requests.post(
                    f"{BASE_URL}/process_reviews/upload", params=params, files=files
                )
            except requests.exceptions.RequestException as err:
                st.error(f"Request failed: {err}")
                st.stop()

        if resp.ok:
            data = resp.json()
            if data.get('status') == 'success':
                results = data.get('results', {})
                if not results:
                    st.error("No results returned from backend.")
                else:
                    app_names = list(results.keys())
                    chosen_app = st.selectbox("Select app to view results", app_names)
                    st.success(data.get('message', 'Success'))
                    result = results[chosen_app]

                    # Show clustering candidates
                    candidates = result.get("clustering_results", {}).get("candidates", [])
                    if not candidates:
                        st.warning("No clustering candidates found.")
                    else:
                        st.markdown("### Select a Clustering Candidate")
                        for i, cand in enumerate(candidates[:3]):
                            summary = cand.get("summary", {})
                            if summary:
                                label = f"Candidate {summary.get('index', i)}: {summary.get('n_clusters')} clusters, Silhouette {summary.get('metrics', {}).get('silhouette_score'):.3f}"
                                with st.expander(label):
                                    st.json(summary)
                                    if st.button(f"Use Candidate {summary.get('index', i)}", key=f"use_candidate_{i}"):
                                        with st.spinner("Submitting selected candidate and awaiting taxonomy results..."):
                                            clustering_data = cand.get("clustering", {})
                                            def clean(obj):
                                                if isinstance(obj, float):
                                                    return obj if np.isfinite(obj) else None
                                                if isinstance(obj, list):
                                                    return [clean(v) for v in obj]
                                                if isinstance(obj, dict):
                                                    return {k: clean(v) for k, v in obj.items()}
                                                return obj
                                            cleaned = clean(clustering_data)
                                            try:
                                                save_resp = requests.post(
                                                    f"{BASE_URL}/save_selected_clustering/{chosen_app}",
                                                    json={"clustering": cleaned},
                                                    timeout=60  # in case processing takes time
                                                )
                                            except requests.exceptions.RequestException as e:
                                                st.error(f"Request failed while submitting clustering: {e}")
                                                st.stop()

                                            if save_resp.ok:
                                                st.success("Mini-taxonomy built successfully.")
                                                st.json(save_resp.json())
                                            else:
                                                st.error(f"Error saving clustering: {save_resp.text}")
            else:
                st.error(f"Error from backend: {data.get('error', resp.text)}")
        else:
            st.error(f"HTTP {resp.status_code}: {resp.text}")
else:
    st.info("Upload a CSV file via the sidebar to begin.")
