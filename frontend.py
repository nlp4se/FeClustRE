import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
from typing import Dict, Any, Optional
from pyvis.network import Network
import uuid
import tempfile
import os

import streamlit.components.v1 as components

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


def apply_sibling_threshold_filter(node, threshold, level=0):
    """Apply sibling threshold filtering to tree structure"""
    if not node:
        return node

    # Calculate grouping factor based on threshold
    # Lower threshold = more grouping, Higher threshold = less grouping
    max_siblings_to_show = max(2, int(10 * threshold))  # Range: 2-10 siblings
    max_depth_to_show = max(2, int(5 * threshold) + 1)  # Range: 2-6 levels

    filtered_node = {
        'feature': node.get('feature', 'Unknown'),
        'is_leaf': node.get('is_leaf', False),
        'children': []
    }

    children = node.get('children', [])

    # If we're at max depth or threshold is low, group remaining children
    if level >= max_depth_to_show or (threshold < 0.5 and len(children) > max_siblings_to_show):
        if children:
            # Count leaf features in all children
            leaf_count = count_leaf_features(children)
            if leaf_count > 0:
                grouped_node = {
                    'feature': f"Grouped Features ({leaf_count} items)",
                    'is_leaf': True,
                    'children': [],
                    'grouped': True
                }
                filtered_node['children'] = [grouped_node]
        return filtered_node

    # Show limited number of siblings if threshold is low
    if threshold < 0.7 and len(children) > max_siblings_to_show:
        visible_children = children[:max_siblings_to_show]
        remaining_count = len(children) - max_siblings_to_show

        # Process visible children
        for child in visible_children:
            filtered_child = apply_sibling_threshold_filter(child, threshold, level + 1)
            filtered_node['children'].append(filtered_child)

        # Add grouped node for remaining children
        if remaining_count > 0:
            remaining_leaves = count_leaf_features(children[max_siblings_to_show:])
            grouped_node = {
                'feature': f"+ {remaining_count} more ({remaining_leaves} features)",
                'is_leaf': True,
                'children': [],
                'grouped': True
            }
            filtered_node['children'].append(grouped_node)
    else:
        # Process all children normally
        for child in children:
            filtered_child = apply_sibling_threshold_filter(child, threshold, level + 1)
            filtered_node['children'].append(filtered_child)

    return filtered_node


def count_leaf_features(nodes):
    """Count total leaf features in a list of nodes"""
    count = 0
    for node in nodes:
        if node.get('is_leaf', False):
            count += 1
        else:
            count += count_leaf_features(node.get('children', []))
    return count


def tree_to_graph(graph, node, parent=None, level=0, sibling_threshold=0.5):
    """Convert tree structure to pyvis graph with sibling threshold applied"""
    node_id = str(uuid.uuid4())
    label = node.get("feature", "Unknown")
    is_leaf = node.get("is_leaf", False)
    is_grouped = node.get("grouped", False)

    # Determine node styling based on type and level
    if is_grouped:
        node_opts = {
            "label": label,
            "shape": "box",
            "color": {"background": "#FF9800", "border": "#F57C00"},  # Orange for grouped
            "font": {"size": 12, "color": "white"},
            "size": 25
        }
    elif is_leaf:
        node_opts = {
            "label": label,
            "shape": "ellipse",
            "color": {"background": "#4CAF50", "border": "#388E3C"},  # Green for features
            "font": {"size": 10, "color": "white"},
            "size": 15
        }
    else:
        # Color categories by level
        colors = [
            {"background": "#2196F3", "border": "#1976D2"},  # Blue
            {"background": "#9C27B0", "border": "#7B1FA2"},  # Purple
            {"background": "#FF5722", "border": "#D84315"},  # Deep Orange
            {"background": "#607D8B", "border": "#455A64"},  # Blue Grey
        ]
        color = colors[level % len(colors)]

        node_opts = {
            "label": label,
            "shape": "box",
            "color": color,
            "font": {"size": 14, "color": "white"},
            "size": max(20, 30 - level * 3)  # Decrease size with depth
        }

    graph.add_node(node_id, **node_opts)

    if parent:
        edge_opts = {"color": {"color": "#666666"}, "width": 2}
        if is_grouped:
            edge_opts["dashes"] = True  # Dashed lines for grouped items
        graph.add_edge(parent, node_id, **edge_opts)

    # Process children
    for child in node.get("children", []):
        tree_to_graph(graph, child, node_id, level + 1, sibling_threshold)

    return graph


def display_taxonomy_graph(tree_data, label="taxonomy_graph", sibling_threshold=0.5):
    """Display interactive taxonomy graph with pyvis"""
    # Apply threshold filtering
    filtered_tree = apply_sibling_threshold_filter(tree_data, sibling_threshold)

    # Create network
    net = Network(height="650px", width="100%", directed=True, notebook=False,
                  bgcolor="#1a1a1a", font_color="white")

    # Build graph
    tree_to_graph(net, filtered_tree, sibling_threshold=sibling_threshold)

    # Configure physics and layout
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "directed",
          "levelSeparation": 150,
          "nodeSpacing": 200,
          "treeSpacing": 250
        }
      },
      "physics": {
        "enabled": false,
        "hierarchicalRepulsion": {
          "centralGravity": 0.0,
          "springLength": 100,
          "springConstant": 0.01,
          "nodeDistance": 120,
          "damping": 0.09
        }
      },
      "nodes": {
        "borderWidth": 2,
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.5)",
          "size": 10,
          "x": 2,
          "y": 2
        }
      },
      "edges": {
        "arrows": { 
          "to": { 
            "enabled": true, 
            "scaleFactor": 0.8,
            "type": "arrow"
          } 
        },
        "smooth": {
          "enabled": true,
          "type": "dynamic"
        }
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hover": true
      }
    }
    """)

    # Generate and display the graph
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            temp_filename = tmp_file.name  # Save path to delete later

        # Now read after closing the `with` block
        with open(temp_filename, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Modify HTML
        enhanced_html = html_content.replace(
            '<body>',
            '''<body style="margin:0; background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);">
            <div style="position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.8); 
                       color: white; padding: 8px 12px; border-radius: 6px; font-size: 12px; z-index: 1000;">
                Threshold: {:.1f} | Drag nodes to explore | Scroll to zoom
            </div>'''.format(sibling_threshold)
        )

        # Display graph
        components.html(enhanced_html, height=700)

        try:
            os.unlink(temp_filename)
        except Exception as e:
            st.warning(f"Couldn't delete temp file: {str(e)}")

    except Exception as e:
        st.error(f"Error generating graph: {str(e)}")
        # Fallback to simple tree display
        st.write("**Fallback Tree View:**")
        _display_tree_node_simple(filtered_tree, 0)


def _display_tree_node_simple(node, level):
    """Simple fallback tree display"""
    indent = "  " * level

    if node.get('grouped', False):
        st.write(f"{indent}📦 *{node.get('feature', 'Group')}*")
    elif node.get('is_leaf', False):
        st.write(f"{indent}🍃 **{node.get('feature', 'Unknown')}**")
    else:
        label = node.get('feature', 'Internal')
        st.write(f"{indent}📁 *{label}*")

    # Display children
    for child in node.get('children', []):
        _display_tree_node_simple(child, level + 1)


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


def _display_tree_node(node, level):
    """Recursively display tree structure with indentation"""
    indent = "  " * level

    if node.get('is_leaf', False):
        st.write(f"{indent}🍃 **{node.get('feature', 'Unknown')}**")
    else:
        label = node.get('feature', 'Internal')
        if label == 'internal':
            label = f"Group {level + 1}"
        st.write(f"{indent}📁 *{label}*")

    # Display children
    for child in node.get('children', []):
        _display_tree_node(child, level + 1)


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

# Mini Taxonomy Visualization
if st.session_state.processing_results and st.session_state.selected_app:
    st.divider()
    st.subheader("🌳 LLM Tagged Mini Taxonomies")

    # Fetch mini taxonomies for the selected app
    try:
        taxonomies_resp = make_request("GET", f"{BASE_URL}/mini_taxonomies/{st.session_state.selected_app}")

        if taxonomies_resp.ok:
            taxonomies_data = taxonomies_resp.json()

            if taxonomies_data.get('taxonomies'):
                taxonomies = taxonomies_data['taxonomies']

                # Create tabs for each taxonomy
                if len(taxonomies) > 0:
                    tab_names = [f"{tax['label'][:25]}..." if len(tax['label']) > 25 else tax['label']
                                 for tax in taxonomies]

                    tabs = st.tabs(tab_names)

                    for tab_idx, tab in enumerate(tabs):
                        with tab:
                            taxonomy = taxonomies[tab_idx]

                            # Controls for this specific taxonomy
                            col1, col2 = st.columns([3, 1])

                            with col2:
                                st.write("**🎛️ View Controls**")

                                # Sibling threshold slider (frontend only)
                                sibling_threshold = st.slider(
                                    "Visual Grouping",
                                    min_value=0.1,
                                    max_value=1.0,
                                    value=0.5,
                                    step=0.1,
                                    key=f"threshold_{tab_idx}",
                                    help="Lower = more grouping, Higher = more detail"
                                )

                                # View type selector
                                view_type = st.selectbox(
                                    "View Type",
                                    ["Interactive Graph", "Simple Tree"],
                                    key=f"view_{tab_idx}"
                                )

                                # Reset button
                                if st.button("🔄 Reset View", key=f"reset_{tab_idx}"):
                                    st.rerun()

                            with col1:
                                # Taxonomy info
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Depth", taxonomy.get('depth', 0))
                                with col_b:
                                    st.metric("Leaves", taxonomy.get('leaf_count', 0))
                                with col_c:
                                    st.metric("Features", len(taxonomy.get('features', [])))

                                # Display based on view type
                                tree_data = taxonomy.get('tree_structure', {})
                                if tree_data:
                                    if view_type == "Interactive Graph":
                                        st.write(f"**🌐 Interactive Graph** (Threshold: {sibling_threshold})")
                                        display_taxonomy_graph(
                                            tree_data,
                                            label=f"graph_{taxonomy.get('label', 'tax')}",
                                            sibling_threshold=sibling_threshold
                                        )
                                    else:
                                        st.write("**🌳 Simple Tree View**")
                                        filtered_tree = apply_sibling_threshold_filter(tree_data, sibling_threshold)
                                        _display_tree_node_simple(filtered_tree, 0)
                                else:
                                    st.info("No tree structure available")

                            # Feature list
                            with st.expander("📋 All Features"):
                                features = taxonomy.get('features', [])
                                if features:
                                    # Apply threshold to feature display
                                    max_features = max(5, int(len(features) * sibling_threshold))
                                    displayed_features = features[:max_features]

                                    feature_cols = st.columns(3)
                                    for i, feature in enumerate(displayed_features):
                                        with feature_cols[i % 3]:
                                            st.code(feature, language=None)

                                    if len(displayed_features) < len(features):
                                        st.info(
                                            f"Showing {len(displayed_features)} of {len(features)} features (threshold: {sibling_threshold:.1f})")
                                else:
                                    st.info("No features found")
                else:
                    st.info("No mini taxonomies found for this app.")
            else:
                st.info("No mini taxonomies found. Save a clustering first to generate them.")
        else:
            st.error("Failed to fetch mini taxonomies")

    except Exception as e:
        st.error(f"Error fetching mini taxonomies: {str(e)}")

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
        6. **View Interactive Taxonomies**: Explore your taxonomies with the interactive graph visualization
        7. **Adjust Sibling Threshold**: Use the slider to control grouping and detail level

        **Required CSV Columns:**
        - `app_name`: Name of the application
        - `review`: Review text content
        - `app_package`: Application package identifier
        - `app_categoryId`: Application category
        - `reviewId`: Unique review identifier  
        - `score`: Review rating/score

        **Sibling Threshold Guide:**
        - **Low (0.1-0.4)**: More grouping, simplified view, fewer details
        - **Medium (0.5-0.7)**: Balanced view with moderate grouping
        - **High (0.8-1.0)**: Full detail, minimal grouping, complete tree
        """)