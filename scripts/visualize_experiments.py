#!/usr/bin/env python3
"""
Streamlit visualizer for Experiment 1 and Experiment 2 data.

Run with:
  .venv/bin/streamlit run scripts/visualize_experiments.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP1_CSV     = PROJECT_ROOT / "data/experiment1.csv"
EXP2_JSON    = PROJECT_ROOT / "data/experiment2.json"

st.set_page_config(page_title="FeClustRE Experiments", layout="wide")
st.title("FeClustRE — Experiment Data Explorer")

tab1, tab2 = st.tabs(["Experiment 1 — Parent/Child Validation", "Experiment 2 — Tree vs Flat List"])


def _exp2_tree_root(tree_json: dict) -> dict:
    if "tree" in tree_json:
        return tree_json["tree"]
    label = tree_json.get("label", "cluster")
    return {
        "label": label,
        "is_leaf": False,
        "children": [
            {"label": f["name"], "is_leaf": True, "name": f["name"], "reviews": f.get("reviews", [])}
            for f in tree_json.get("features", [])
        ],
    }


def _leaf_count(node: dict) -> int:
    children = node.get("children") or []
    if node.get("is_leaf") or not children:
        return 1
    return sum(_leaf_count(child) for child in children)


def _sunburst_from_node(node: dict, parent_id: str, ids, labels, parents, values, counter: list):
    node_id = f"n_{counter[0]}"
    counter[0] += 1
    ids.append(node_id)
    labels.append(node.get("label", node.get("name", "?")))
    parents.append(parent_id)
    children = node.get("children") or []
    values.append(_leaf_count(node))
    for child in children:
        _sunburst_from_node(child, node_id, ids, labels, parents, values, counter)


def _render_tree_node(node: dict, depth: int = 0):
    name = node.get("name") or node.get("label", "?")
    children = node.get("children") or []
    if node.get("is_leaf") or not children:
        reviews = node.get("reviews", [])
        with st.expander(f"{'  ' * depth}**{name}**  ({len(reviews)} reviews)"):
            if reviews:
                for rv in reviews:
                    st.markdown(f"> {rv[:200]}")
            else:
                st.caption("no review hits")
        return
    with st.expander(f"{'  ' * depth}**{name}**  ({len(children)} branches)", expanded=depth < 1):
        for child in children:
            _render_tree_node(child, depth + 1)

# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------
with tab1:
    if not EXP1_CSV.exists():
        st.warning(f"No data at {EXP1_CSV}. Run generate_experiment1.py first.")
    else:
        df = pd.read_csv(EXP1_CSV)
        df["siblings_list"] = df["sibling_features"].apply(json.loads)
        df["reviews_list"]  = df["example_reviews"].apply(json.loads)

        col_l, col_r = st.columns([2, 1])
        with col_l:
            app_filter = st.multiselect("Filter by app", sorted(df["app_name"].unique()), key="e1_app")
        with col_r:
            size_range = st.slider("Cluster size", int(df["cluster_size"].min()),
                                   int(df["cluster_size"].max()),
                                   (int(df["cluster_size"].min()), int(df["cluster_size"].max())),
                                   key="e1_size")

        view = df.copy()
        if app_filter:
            view = view[view["app_name"].isin(app_filter)]
        view = view[(view["cluster_size"] >= size_range[0]) & (view["cluster_size"] <= size_range[1])]

        st.caption(f"{len(view)} rows shown / {len(df)} total")

        # Summary stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", len(view))
        c2.metric("Apps", view["app_name"].nunique())
        c3.metric("Avg cluster size", f"{view['cluster_size'].mean():.1f}")
        c4.metric("Avg siblings", f"{view['n_siblings'].mean():.1f}")

        st.divider()

        # Cluster size distribution
        with st.expander("Cluster size distribution", expanded=False):
            dist = view["cluster_size"].value_counts().sort_index()
            fig = go.Figure(go.Bar(
                x=[str(k) for k in dist.index],
                y=dist.values,
                marker_color="steelblue",
            ))
            fig.update_layout(
                xaxis=dict(title="Cluster size", tickangle=0),
                yaxis=dict(title="Count"),
                height=300,
                margin=dict(t=10, l=0, r=0, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        def tree_svg(parent: str, child: str, siblings: list[str]) -> str:
            leaves = [child] + siblings
            n = len(leaves)
            BOX_W, BOX_H = 160, 44
            H_GAP, V_GAP = 16, 56
            total_w = n * BOX_W + (n - 1) * H_GAP
            svg_w = max(total_w, BOX_W) + 60
            svg_h = BOX_H + V_GAP + BOX_H + 40

            px = svg_w / 2
            start_x = (svg_w - total_w) / 2 + BOX_W / 2
            child_y_top = 20 + BOX_H + V_GAP

            def clip(t: str, n: int = 20) -> str:
                return (t[:n] + "…") if len(t) > n else t

            parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}">']

            # Lines parent → each leaf
            for i in range(n):
                cx = start_x + i * (BOX_W + H_GAP)
                parts.append(
                    f'<line x1="{px:.1f}" y1="{20+BOX_H}" '
                    f'x2="{cx:.1f}" y2="{child_y_top}" '
                    f'stroke="#94a3b8" stroke-width="1.5"/>'
                )

            # Parent box
            parts.append(
                f'<rect x="{px-BOX_W/2:.1f}" y="20" width="{BOX_W}" height="{BOX_H}" '
                f'rx="8" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{px:.1f}" y="{20+BOX_H/2+5:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#1e3a5f">'
                f'{clip(parent, 22)}</text>'
            )

            # Leaf boxes
            for i, leaf in enumerate(leaves):
                cx = start_x + i * (BOX_W + H_GAP)
                is_child = (leaf == child)
                fill   = "#d1fae5" if is_child else "#f1f5f9"
                stroke = "#059669" if is_child else "#94a3b8"
                color  = "#064e3b" if is_child else "#334155"
                label  = clip(leaf, 20) + (" ✓" if is_child else "")
                weight = "bold" if is_child else "normal"
                parts.append(
                    f'<rect x="{cx-BOX_W/2:.1f}" y="{child_y_top}" '
                    f'width="{BOX_W}" height="{BOX_H}" '
                    f'rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
                )
                parts.append(
                    f'<text x="{cx:.1f}" y="{child_y_top+BOX_H/2+5:.1f}" '
                    f'text-anchor="middle" font-family="sans-serif" '
                    f'font-size="12" font-weight="{weight}" fill="{color}">'
                    f'{label}</text>'
                )

            parts.append("</svg>")
            return "".join(parts)

        st.subheader("Rows")
        for _, row in view.iterrows():
            siblings = row["siblings_list"]
            reviews  = row["reviews_list"]
            header   = f"**{row['parent_feature']}** › {row['child_feature']}  `{row['app_name']}`  (size {row['cluster_size']})"
            with st.expander(header):
                svg = tree_svg(row["parent_feature"], row["child_feature"], siblings)
                st.markdown(f'<div style="overflow-x:auto;padding:8px 0">{svg}</div>',
                            unsafe_allow_html=True)
                if reviews:
                    st.markdown("**Example reviews**")
                    for i, rv in enumerate(reviews, 1):
                        st.markdown(f"_{i}. {rv[:220]}_")

# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------
with tab2:
    if not EXP2_JSON.exists():
        st.warning(f"No data at {EXP2_JSON}. Run generate_experiment2.py first.")
    else:
        data = json.load(open(EXP2_JSON))
        options = [
            f"{d['app_name']} · {d.get('label', d['tree_id'])}  "
            f"({d['n_features']} features, depth {d.get('tree_depth', '?')})"
            for d in data
        ]
        by_key = {opt: d for opt, d in zip(options, data)}

        col_l, col_r = st.columns([3, 1])
        with col_l:
            selected = st.selectbox("Select cluster", options, key="e2_cluster")
        with col_r:
            view_mode = st.radio("View mode", ["Tree", "Flat list"], key="e2_mode", horizontal=True)

        entry = by_key[selected]
        tree  = entry["tree_json"]
        flat  = entry["list_json"]
        label = entry.get("label") or tree.get("label", "cluster")

        c1, c2, c3 = st.columns(3)
        c1.metric("Features", entry["n_features"])
        c2.metric("Tree depth", entry.get("tree_depth", tree.get("depth", "?")))
        c3.metric("App", entry["app_name"])

        st.divider()

        if view_mode == "Tree":
            root = _exp2_tree_root(tree)
            ids, labels, parents, values = ["root"], [root.get("label", label)], [""], [_leaf_count(root)]
            _sunburst_from_node(root, "root", ids, labels, parents, values, [0])

            fig = go.Figure(go.Sunburst(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                insidetextorientation="radial",
            ))
            fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=550)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"Cluster: {label}")
            for child in root.get("children", []):
                _render_tree_node(child)

        else:  # Flat list
            st.subheader(f"Features ({len(flat['features'])})")
            for feat in flat["features"]:
                with st.expander(f"{feat['name']}  ({len(feat['reviews'])} reviews)"):
                    if feat["reviews"]:
                        for rv in feat["reviews"]:
                            st.markdown(f"> {rv[:200]}")
                    else:
                        st.caption("no review hits")
