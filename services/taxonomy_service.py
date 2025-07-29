import uuid
from datetime import datetime
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree, fcluster
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import logging
from datetime import datetime
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist
import uuid
from config import config
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import json
import re

logger = logging.getLogger(__name__)


class TaxonomyBuilder:
    def __init__(self, neo4j_conn, feature_extractor, height_threshold=0.5):
        self.neo4j_conn = neo4j_conn
        self.feature_extractor = feature_extractor
        self.height_threshold = height_threshold

    def merge_mini_taxonomies(self, app_name):
        try:
            with self.neo4j_conn.driver.session(database=self.neo4j_conn.database) as session:
                result = session.run("""
                    MATCH (app:App {name: $app_name})-[:HAS_MINI_TAXONOMY]->(root:MiniTaxonomyNode)
                    OPTIONAL MATCH (root)-[:HAS_CHILD*]->(leaf)
                    WHERE NOT (leaf)-[:HAS_CHILD]->()
                    RETURN root.id as root_id, 
                           root.llm_tag as label,
                           count(DISTINCT leaf) as leaf_count,
                           collect(DISTINCT leaf.feature) as leaf_features
                """, app_name=app_name)

                taxonomies = [
                    {
                        'root_id': record['root_id'],
                        'label': record['label'].strip() if record['label'] else '',
                        'leaf_count': record['leaf_count'] or 0,
                        'leaf_features': record['leaf_features'] or []
                    }
                    for record in result
                ]

            if len(taxonomies) < 2:
                return {"merged_count": 0, "merges": []}

            labels = [tax['label'] for tax in taxonomies]
            embeddings = self.feature_extractor.get_embeddings(labels)
            similarity_matrix = cosine_similarity(embeddings)

            merges = []
            merged_ids = set()

            for i in range(len(taxonomies)):
                for j in range(i + 1, len(taxonomies)):
                    if taxonomies[i]['root_id'] in merged_ids or taxonomies[j]['root_id'] in merged_ids:
                        continue

                    similarity = float(similarity_matrix[i][j])

                    if similarity >= 0.85:
                        tax_a, tax_b = taxonomies[i], taxonomies[j]

                        if tax_a['leaf_count'] >= tax_b['leaf_count']:
                            primary, secondary = tax_a, tax_b
                        else:
                            primary, secondary = tax_b, tax_a

                        with self.neo4j_conn.driver.session(database=self.neo4j_conn.database) as session:
                            session.write_transaction(lambda tx: tx.run("""
                                MATCH (r:MiniTaxonomyNode {id: $secondary_id})
                                OPTIONAL MATCH (r)-[:HAS_CHILD]->(c)
                                OPTIONAL MATCH (a:App)-[rel:HAS_MINI_TAXONOMY]->(r)
                                DELETE rel
                                WITH c, r
                                MATCH (p:MiniTaxonomyNode {id: $primary_id})
                                FOREACH (_ IN CASE WHEN c IS NOT NULL THEN [1] ELSE [] END |
                                    MERGE (p)-[:HAS_CHILD]->(c)
                                )
                                DETACH DELETE r
                            """, secondary_id=secondary['root_id'], primary_id=primary['root_id']))

                        merges.append({
                            'merged_taxonomies': [secondary['label']],
                            'into_taxonomy': primary['label'],
                            'similarity': round(similarity, 4)
                        })
                        merged_ids.add(secondary['root_id'])

            return {
                "merged_count": len(merges),
                "merges": merges
            }

        except Exception as e:
            logger.error(f"Mini taxonomy merge failed for '{app_name}': {e}", exc_info=True)
            return {"merged_count": 0, "merges": [], "error": str(e)}

    def build_and_store_taxonomy(self, app_name, features, embeddings, method="bert"):
        if len(features) < 2:
            logger.warning("Not enough features to build taxonomy.")
            return {"error": "Not enough features for taxonomy"}

        embeddings = np.array(embeddings)
        distance_matrix = cosine_distances(embeddings)
        linkage_matrix = linkage(distance_matrix, method='average')

        tree_root, _ = to_tree(linkage_matrix, rd=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{app_name}_taxonomy_{method}_{timestamp}"

        with self.neo4j_conn.driver.session(database=self.neo4j_conn.database) as session:
            node_counter = [0]
            root_id = f"taxonomy_node_{node_counter[0]}"
            session.write_transaction(
                self._create_taxonomy_tree_with_app_link,
                tree_root, features, session_id, app_name, method, node_counter, root_id=root_id
            )

        best_cut, top_results = self._tune_subcluster_cut(linkage_matrix, features, embeddings)

        for result in top_results:
            result["subclusters"] = {
                str(k): v for k, v in result["subclusters"].items()
            }

        return {
            "taxonomy_session_id": session_id,
            "top_subcluster_options": top_results[:3],
            "message": "Top subcluster cut options generated. Use /save_subclusters/<app_name> to persist one."
        }

    def save_selected_subclusters(self, app_name, session_id, selected_subclusters):
        with self.neo4j_conn.driver.session(database=self.neo4j_conn.database) as session:
            session.write_transaction(self._store_subclusters, app_name, session_id, selected_subclusters)

    def _create_taxonomy_tree_with_app_link(self, tx, node, features, session_id, app_name, method, node_counter,
                                            parent_id=None, root_id=None):
        is_leaf = node.is_leaf()
        feature = features[node.id] if is_leaf else None

        node_id = f"taxonomy_node_{node_counter[0]}"
        node_counter[0] += 1

        tx.run("""
            CREATE (n:TaxonomyNode {
                id: $node_id,
                is_leaf: $is_leaf,
                feature: $feature,
                session_id: $session_id
            })
        """, node_id=node_id, is_leaf=is_leaf, feature=feature, session_id=session_id)

        if parent_id:
            tx.run("""
                MATCH (p:TaxonomyNode {id: $parent_id}), (c:TaxonomyNode {id: $child_id})
                CREATE (p)-[:HAS_CHILD]->(c)
            """, parent_id=parent_id, child_id=node_id)

        if node_id == root_id:
            tx.run("""
                MATCH (a:App {name: $app_name})
                MATCH (root:TaxonomyNode {id: $root_id})
                MERGE (a)-[:HAS_TAXONOMY {method: $method}]->(root)
            """, app_name=app_name, root_id=root_id, method=method)

        if node.left:
            self._create_taxonomy_tree_with_app_link(tx, node.left, features, session_id, app_name, method,
                                                     node_counter, node_id, root_id)
        if node.right:
            self._create_taxonomy_tree_with_app_link(tx, node.right, features, session_id, app_name, method,
                                                     node_counter, node_id, root_id)

    def _cut_tree(self, linkage_matrix, features, height_threshold):
        cluster_assignments = fcluster(linkage_matrix, t=height_threshold, criterion='distance')
        subclusters = {}
        for feature, cluster_id in zip(features, cluster_assignments):
            if cluster_id not in subclusters:
                subclusters[cluster_id] = []
            subclusters[cluster_id].append(feature)
        return subclusters

    def _store_subclusters(self, tx, app_name, session_id, subclusters):
        for cluster_id, feature_list in subclusters.items():
            cluster_uuid = str(uuid.uuid4())
            tx.run("""
                MATCH (a:App {name: $app_name})
                CREATE (sc:Subcluster {
                    id: $cluster_uuid,
                    cluster_id: $cluster_id,
                    session_id: $session_id
                })
            """, app_name=app_name, cluster_uuid=cluster_uuid, cluster_id=cluster_id, session_id=session_id)

            for feature in feature_list:
                tx.run("""
                    MATCH (sc:Subcluster {id: $cluster_uuid})
                    MATCH (f:Feature {name: $feature})
                    MERGE (sc)-[:HAS_FEATURE]->(f)
                """, cluster_uuid=cluster_uuid, feature=feature)

    def _tune_subcluster_cut(self, linkage_matrix, features, embeddings):
        thresholds = np.linspace(0.2, 1.0, 10)
        embeddings_dict = {f: e for f, e in zip(features, embeddings)}
        results = []

        for t in thresholds:
            subclusters = self._cut_tree(linkage_matrix, features, t)
            if len(subclusters) == 0:
                continue

            avg_size = np.mean([len(v) for v in subclusters.values()])
            coverage = len(set([f for v in subclusters.values() for f in v])) / len(features)
            coherence = self._compute_avg_coherence(subclusters, embeddings_dict)
            n_singletons = sum(1 for v in subclusters.values() if len(v) == 1)
            singleton_ratio = n_singletons / len(subclusters)
            score = (len(subclusters) * avg_size * coherence * coverage) * (1 - singleton_ratio)

            results.append({
                "height_threshold": float(t),
                "n_subclusters": len(subclusters),
                "avg_size": round(avg_size, 2),
                "avg_coherence": round(coherence, 3),
                "coverage": round(coverage, 3),
                "singleton_ratio": round(singleton_ratio, 3),
                "score": round(score, 3),
                "subclusters": subclusters
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[0]["height_threshold"], results[:3]

    def _compute_avg_coherence(self, subclusters, embeddings_dict):
        scores = []
        for feats in subclusters.values():
            emb = np.array([embeddings_dict[f] for f in feats if f in embeddings_dict])
            if len(emb) > 1:
                sim = cosine_similarity(emb)
                sim_vals = sim[np.triu_indices_from(sim, k=1)]
                scores.append(np.mean(sim_vals))
        return np.mean(scores) if scores else 0.0

    def _extract_subtree_structures(self, features, taxonomy_tree):
        parent_features = set()
        child_features = set()

        for parent, children in taxonomy_tree.items():
            if parent in features:
                parent_features.add(parent)
                for child in children:
                    if child in features:
                        child_features.add(child)

        if not parent_features and not child_features:
            sorted_feats = sorted(features, key=len)
            return {
                "parent_features": [sorted_feats[0]] if sorted_feats else [],
                "child_features": sorted_feats[1:] if len(sorted_feats) > 1 else []
            }

        return {
            "parent_features": list(parent_features),
            "child_features": list(child_features)
        }

    def store_llm_taxonomy(self, app_name, clusters, method="llm-clustering"):
        session_id = f"{app_name}_llm_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        node_counter = [0]
        labels = {}

        def _create_node(tx, node_id, feature, is_leaf, session_id, llm_tag=None):
            tx.run("""
                MERGE (n:MiniTaxonomyNode {id: $id})
                SET n.is_leaf = $is_leaf,
                    n.feature = $feature,
                    n.session_id = $session_id,
                    n.llm_tag = $llm_tag
            """, id=node_id, is_leaf=is_leaf, feature=feature,
                   session_id=session_id, llm_tag=llm_tag)

        def _create_link(tx, parent_id, child_id):
            tx.run("""
                MATCH (p:MiniTaxonomyNode {id: $parent_id}),
                      (c:MiniTaxonomyNode {id: $child_id})
                MERGE (p)-[:HAS_CHILD]->(c)
            """, parent_id=parent_id, child_id=child_id)

        def _build_tree(tx, subtree, parent_id):
            node_id = f"mini_taxonomy_node_{node_counter[0]}"
            node_counter[0] += 1
            feature = subtree.get("feature", "internal")
            is_leaf = subtree.get("is_leaf", False)
            _create_node(tx, node_id, feature, is_leaf, session_id)
            _create_link(tx, parent_id, node_id)
            for child in subtree.get("children", []):
                _build_tree(tx, child, node_id)

        def _extract_tree(features):
            if len(features) == 1:
                return {"feature": features[0], "is_leaf": True, "children": []}
            embs = self.feature_extractor.get_embeddings(features)
            dists = pdist(embs, metric="cosine")
            root = to_tree(linkage(dists, method="average"))

            def recurse(node):
                if node.is_leaf():
                    return {"feature": features[node.id], "is_leaf": True, "children": []}
                return {
                    "feature": "internal",
                    "is_leaf": False,
                    "children": [recurse(node.left), recurse(node.right)]
                }

            return recurse(root)

        logger.info(f"Creating mini taxonomies for {len(clusters)} clusters.")
        for cluster_id, features in clusters.items():
            if len(features) == 1:
                logger.info(f"Skipping cluster {cluster_id} with only one feature: {features[0]}")
                continue

            raw_label = generate_cluster_label(features)
            label = raw_label.strip().lower()
            labels[cluster_id] = label
            try:
                subtree = _extract_tree(features)
                root_id = f"mini_taxonomy_root_{uuid.uuid4().hex[:8]}"
                with self.neo4j_conn.driver.session(database=self.neo4j_conn.database) as session:
                    session.write_transaction(_create_node, root_id, label, False, session_id, label)
                    session.write_transaction(
                        lambda tx: tx.run("""
                            MATCH (a:App {name: $app_name}), (r:MiniTaxonomyNode {id: $root_id})
                            MERGE (a)-[:HAS_MINI_TAXONOMY {method: $method}]->(r)
                        """, app_name=app_name, root_id=root_id, method=method)
                    )
                    session.write_transaction(_build_tree, subtree, root_id)
            except Exception as e:
                logger.error(f"Failed to store LLM taxonomy for cluster {cluster_id}: {e}", exc_info=True)

        logger.info(f"Attempting to merge mini taxonomies for app '{app_name}'...")
        merge_result = self.merge_mini_taxonomies(app_name)
        logger.info(f"Merging complete: {merge_result}")

        return labels


def generate_cluster_label(features, mode="few-shot", base_url=None, model=None, used_labels=None, retries=3):

    cfg = config["default"]()
    base_url = base_url or cfg.OLLAMA_BASE_URL
    model = model or cfg.OLLAMA_MODEL
    used_labels = used_labels or []
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    def is_similar(label, existing_labels, threshold=0.85):
        if not existing_labels:
            return False
        all_labels = existing_labels + [label]
        embs = embed_model.encode(all_labels)
        sim_matrix = cosine_similarity(embs)
        sims = sim_matrix[-1][:-1]  # similarity of new label with existing
        return any(s > threshold for s in sims)

    for attempt in range(retries):
        try:
            logger.info(f"Requesting label from Qwen for cluster with {len(features)} features.")

            if mode == "few-shot":
                prompt = (
                    "You are an assistant that assigns a single, concise category name to a list of app feature keywords.\n"
                    "Your response must be only the category name. No explanations. No punctuation. No labels like 'Usage' or 'Explanation'.\n"
                    "Return only the category name — a short phrase (1 to 4 words).\n\n"
                    "Examples:\n"
                    # ChatGPT
                    "Features: generate original images, transform existing images\nCategory: Image generation\n\n"
                    "Features: real-time convo, practice a new language, settle a debate\nCategory: Advanced voice mode\n\n"
                    "Features: snap a picture, upload a picture, transcribe handwritten recipe, get info about a landmark\nCategory: Photo upload\n\n"
                    "Features: find custom birthday gift ideas, create personalized greeting card\nCategory: Creative inspiration\n\n"
                    "Features: talk, ask for a detailed travel itinerary, get help crafting response\nCategory: Tailored advice\n\n"
                    "Features: brainstorm marketing copy, map out a business plan\nCategory: Professional input\n\n"
                    "Features: get recipe suggestions\nCategory: Instant answers\n\n"
                    # Copilot
                    "Features: summarized answers, translate, proofread across multiple languages, optimizing text, compose and draft emails, compose and draft cover letters, update your resume\nCategory: Smart work\n\n"
                    "Features: compose stories, compose scripts, image generation, create high quality visuals, render your concepts into stunning visuals, spark inspiration\nCategory: Personal support\n\n"
                    "Features: search by image, explore and develop new styles and ideas, create illustrations, curate social media content, visualize film and video storyboards, build and update a portfolio\nCategory: Image generation\n\n"
                    # Mistral
                    "Features: lightning fast search across the web, real-time news\nCategory: Instant answers\n\n"
                    "Features: document OCR with multilanguage support, multilanguage reasoning capabilities, voice recognition\nCategory: Multimodal understanding\n\n"
                    "Features: deep research, advanced reasoning for complex tasks\nCategory: Research and reasoning\n\n"
                    "Features: organization of data, documents, and notes into personalized Projects\nCategory: Productivity and organization\n\n"
                    "Features: image generation, contextual iteration\nCategory: Image generation\n\n"
                    # Perplexity
                    "Features: guided AI search for deeper exploration\nCategory: Perplexity Pro Search\n\n"
                    "Features: keep the conversation going for a deeper understanding\nCategory: Thread Follow-Up\n\n"
                    "Features: instant, up-to-date answers\nCategory: Voice\n\n"
                    "Features: cited sources for every answer\nCategory: Trust and credibility\n\n"
                    "Features: learn new things from the community\nCategory: Discover\n\n"
                    "Features: curation of your discoveries\nCategory: Your Library\n\n"
                    # Anthropic
                    "Features: draft a business proposal, translate menus, brainstorm gift ideas, compose a speech\nCategory: On-the-go assistance\n\n"
                    "Features: start a chat, attach a file, send a photo for real-time image analysis\nCategory: Instant answers\n\n"
                    "Features: step-by-step thinking, break down complex problems, consider multiple solutions\nCategory: Extended thinking\n\n"
                    "Features: collaborate on critical tasks, brainstorming, complex problems, continue conversations across devices\nCategory: Faster deep work\n\n"
                    "Features: draft emails, summarize meetings, assist with small tasks\nCategory: Productivity support\n\n"
                    "Features: advanced coding, advanced reasoning, AI agents\nCategory: Intelligence\n\n"
                    # Gemini
                    "Features: get help with writing, brainstorming, learning\nCategory: Creative and educational support\n\n"
                    "Features: summarise and find quick info from Gmail or Google Drive\nCategory: Productivity and information retrieval\n\n"
                    "Features: use text, voice, photos, and your camera to get help\nCategory: Multimodal assistance\n\n"
                    "Features: ask for help with what’s on your phone screen using 'Hey Google'\nCategory: Context-aware interaction\n\n"
                    "Features: make plans with Google Maps and Google Flights\nCategory: Planning and navigation\n\n"
                    f"Features: {', '.join(features)}\nCategory:"
                )
            else:
                logger.warning(f"Unsupported prompt mode: {mode}")
                return "Unknown"

            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                },
                stream=True,
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"Ollama returned status {response.status_code} for model '{model}'")
                return "Unknown"

            full_content = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                    chunk = msg.get("message", {}).get("content", "")
                    full_content += chunk
                except Exception:
                    continue

            label = full_content.strip().split("\n")[0]
            label = re.sub(r"[^a-zA-Z0-9\s\-]", "", label).strip()

            if not label or "explanation" in label.lower():
                logger.warning("LLM label invalid. Retrying...")
                continue

            if is_similar(label, used_labels):
                logger.info(f"Label '{label}' too similar to existing labels. Retrying...")
                continue

            logger.info(f"Generated label: '{label}'")
            return label

        except Exception as e:
            logger.error(f"Error generating label: {str(e)}")

    fallback = f"Category {len(used_labels)+1}"
    logger.warning(f"LLM failed to generate distinct label. Using fallback: {fallback}")
    return fallback

