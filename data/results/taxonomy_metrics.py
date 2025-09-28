import json
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkingTaxonomyAnalyzer:
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_taxonomy_metrics(self, file_path="taxonomy_metrics.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded taxonomy_metrics.json - Type: {type(data)}")
                if isinstance(data, dict):
                    logger.info(f"Root keys: {list(data.keys())}")
                return data
        except FileNotFoundError:
            logger.error(f"File {file_path} not found")
            return None

    def extract_taxonomies_from_metrics(self, metrics_data):
        """Extract taxonomy data from the complex metrics JSON structure"""
        taxonomies = []

        def extract_from_dict(obj, path=""):
            """Recursively search for taxonomy objects"""
            if isinstance(obj, dict):
                # Check if this looks like a taxonomy object
                if all(key in obj for key in ['root_id', 'tag', 'depth', 'leaves']):
                    taxonomies.append(obj)
                else:
                    # Recursively search nested objects
                    for key, value in obj.items():
                        extract_from_dict(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_from_dict(item, f"{path}[{i}]")

        extract_from_dict(metrics_data)

        if taxonomies:
            logger.info(f"Found {len(taxonomies)} taxonomies in the metrics file")
            return taxonomies

        return []

    def calculate_taxonomy_score(self, taxonomy):
        """Calculate quality score favoring smaller, well-structured taxonomies"""
        try:
            depth = taxonomy.get('depth', 0)
            leaves = taxonomy.get('leaves', 0)
            tag = taxonomy.get('tag', '')

            # Ensure numeric values
            depth = int(depth) if depth is not None else 0
            leaves = int(leaves) if leaves is not None else 0

            # Favor smaller taxonomies for screenshots
            if leaves <= 50:
                # Small taxonomies get bonus points
                size_bonus = 50
                structure_score = (depth * 2.0) + (leaves * 1.0) + size_bonus
            elif leaves <= 100:
                # Medium taxonomies get moderate scoring
                structure_score = (depth * 1.0) + (leaves * 0.5) + 20
            else:
                # Large taxonomies get penalized for screenshots
                structure_score = (depth * 0.5) + (leaves * 0.1)

            # Semantic quality based on tag
            semantic_score = self.evaluate_tag_quality(tag)

            # Depth quality bonus (good structure)
            depth_bonus = min(depth * 2, 20) if depth >= 3 else 0

            # Combined score with emphasis on smaller, well-structured taxonomies
            total_score = structure_score + (semantic_score * 2.0) + depth_bonus

            return total_score

        except Exception as e:
            logger.error(f"Error calculating score for taxonomy: {e}")
            logger.error(f"Taxonomy data: {taxonomy}")
            raise

    def evaluate_tag_quality(self, tag):
        """Evaluate semantic quality of taxonomy tag"""
        try:
            if not tag:
                return 0

            tag_lower = tag.lower()

            # Penalty patterns (low quality indicators)
            low_quality_patterns = [
                r"unknown", r"internal", r"cluster \d+", r"category \d+",
                r"misc", r"other", r"undefined", r"unlabeled"
            ]

            for pattern in low_quality_patterns:
                if re.search(pattern, tag_lower):
                    return 1  # Low quality

            # Quality indicators
            quality_indicators = [
                "support", "assistance", "management", "processing",
                "learning", "communication", "analysis", "development"
            ]

            score = 5  # Base score
            for indicator in quality_indicators:
                if indicator in tag_lower:
                    score += 2

            # Length penalty for overly verbose tags
            if len(tag.split()) > 6:
                score -= 1

            return min(score, 10)  # Cap at 10

        except Exception as e:
            logger.warning(f"Error evaluating tag quality for '{tag}': {e}")
            return 5  # Default score

    def perform_semantic_analysis(self, taxonomies):
        """Perform semantic analysis on taxonomy tags and features"""
        analysis = {
            'tag_clusters': [],
            'feature_patterns': {},
            'coherence_scores': {},
            'redundancy_analysis': {}
        }

        try:
            # Extract tags
            tags = [t.get('tag', '') for t in taxonomies if t.get('tag')]

            if not tags:
                logger.warning("No tags found for semantic analysis")
                return analysis

            # Tag clustering analysis
            logger.info(f"Analyzing {len(tags)} tags for semantic similarity")
            tag_embeddings = self.embed_model.encode(tags)
            tag_similarity = cosine_similarity(tag_embeddings)

            # Find similar tag clusters
            similar_pairs = []
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    similarity = tag_similarity[i][j]
                    if similarity > 0.8:  # High similarity threshold
                        similar_pairs.append({
                            'tag1': tags[i],
                            'tag2': tags[j],
                            'similarity': float(similarity)
                        })

            analysis['tag_clusters'] = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)[:20]  # Top 20

            # Basic feature patterns
            all_words = []
            for tag in tags:
                all_words.extend(tag.lower().split())

            word_counter = Counter(all_words)
            analysis['feature_patterns'] = {
                'most_common_words': word_counter.most_common(20),
                'unique_words': len(word_counter),
                'total_words': sum(word_counter.values())
            }

            logger.info("Semantic analysis completed successfully")

        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")

        return analysis

    def extract_best_worst_taxonomies(self):
        """Main function to extract and analyze top 20 best and worst taxonomies"""

        try:
            # Load taxonomy metrics
            metrics_data = self.load_taxonomy_metrics()
            if not metrics_data:
                return None

            # Extract taxonomy data
            structure_data = self.extract_taxonomies_from_metrics(metrics_data)

            if not structure_data:
                logger.error("Could not extract taxonomy data from the metrics file")
                return None

            logger.info(f"Successfully extracted {len(structure_data)} taxonomies")

            # Calculate scores for ranking
            scored_taxonomies = []
            logger.info(f"Calculating quality scores for {len(structure_data)} taxonomies...")

            for i, taxonomy in enumerate(structure_data):
                try:
                    score = self.calculate_taxonomy_score(taxonomy)
                    taxonomy['quality_score'] = score
                    scored_taxonomies.append(taxonomy)

                    if i % 100 == 0:
                        logger.info(f"Processed {i + 1}/{len(structure_data)} taxonomies")

                except Exception as e:
                    logger.warning(f"Failed to score taxonomy {i}: {e}")
                    continue

            logger.info(f"Successfully scored {len(scored_taxonomies)} taxonomies")

            if not scored_taxonomies:
                logger.error("No taxonomies could be scored")
                return None

            # Sort by score
            logger.info("Sorting taxonomies by quality score...")
            scored_taxonomies.sort(key=lambda x: x['quality_score'], reverse=True)
            logger.info("Sorting completed successfully")

            # Get top 20 and bottom 20
            best_20 = scored_taxonomies[:20]
            worst_20 = scored_taxonomies[-20:] if len(scored_taxonomies) >= 20 else []

            # Also get best small taxonomies (for screenshots)
            small_taxonomies = [t for t in scored_taxonomies if t.get('leaves', 0) <= 50]
            best_small = small_taxonomies[:10] if small_taxonomies else []

            logger.info(f"Selected top 20 best and {len(worst_20)} worst taxonomies")
            logger.info(f"Found {len(best_small)} small taxonomies (≤50 features) for screenshots")
            if best_20 and worst_20:
                logger.info(
                    f"Best score: {best_20[0]['quality_score']:.2f}, Worst score: {worst_20[-1]['quality_score']:.2f}")
            if best_small:
                logger.info(
                    f"Best small taxonomy: {best_small[0]['tag']} (L:{best_small[0]['leaves']}, S:{best_small[0]['quality_score']:.1f})")

            # Use the data we already have (skip Neo4j for now)
            best_detailed = best_20
            worst_detailed = worst_20

            # Add small taxonomies section
            small_detailed = best_small

            # Perform semantic analysis
            logger.info("Performing semantic analysis...")
            best_semantic_analysis = self.perform_semantic_analysis(best_detailed)
            worst_semantic_analysis = self.perform_semantic_analysis(worst_detailed)
            small_semantic_analysis = self.perform_semantic_analysis(small_detailed)
            logger.info("Semantic analysis completed successfully")

            # Compile results
            logger.info("Compiling final results...")
            results = {
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_taxonomies_analyzed': len(scored_taxonomies),
                    'best_taxonomies_count': len(best_detailed),
                    'worst_taxonomies_count': len(worst_detailed),
                    'small_taxonomies_count': len(small_detailed)
                },
                'best_taxonomies': {
                    'taxonomies': best_detailed,
                    'semantic_analysis': best_semantic_analysis
                },
                'worst_taxonomies': {
                    'taxonomies': worst_detailed,
                    'semantic_analysis': worst_semantic_analysis
                },
                'small_taxonomies': {
                    'taxonomies': small_detailed,
                    'semantic_analysis': small_semantic_analysis
                },
                'quality_distribution': {
                    'mean_score': float(np.mean([t['quality_score'] for t in scored_taxonomies])),
                    'median_score': float(np.median([t['quality_score'] for t in scored_taxonomies])),
                    'std_score': float(np.std([t['quality_score'] for t in scored_taxonomies])),
                    'min_score': float(min([t['quality_score'] for t in scored_taxonomies])),
                    'max_score': float(max([t['quality_score'] for t in scored_taxonomies]))
                }
            }

            logger.info("Results compilation completed successfully")
            return results

        except Exception as e:
            logger.error(f"Fatal error in extract_best_worst_taxonomies: {e}", exc_info=True)
            return None

    def generate_cypher_queries(self, taxonomies, query_type="analysis"):
        """Generate Cypher queries for analyzing taxonomies in Neo4j"""

        queries = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'query_type': query_type,
                'taxonomy_count': len(taxonomies)
            },
            'individual_queries': [],
            'batch_queries': {}
        }

        root_ids = [t['root_id'] for t in taxonomies if 'root_id' in t]

        # Individual queries for each taxonomy
        for i, taxonomy in enumerate(taxonomies):
            root_id = taxonomy.get('root_id', '')
            tag = taxonomy.get('tag', 'Unknown')

            individual_query = {
                'taxonomy_rank': i + 1,
                'root_id': root_id,
                'tag': tag,
                'quality_score': taxonomy.get('quality_score', 0),
                'queries': {
                    'screenshot_tree': f"""
// Screenshot-friendly tree for: {tag} (Top 10 features only)
MATCH (root:MiniTaxonomyNode {{id: '{root_id}'}})
WHERE NOT (()-[:HAS_CHILD]->(root))
OPTIONAL MATCH path = (root)-[:HAS_CHILD*1..3]->(node:MiniTaxonomyNode)
WITH root, node, path
ORDER BY length(path), node.feature
WITH root, collect(DISTINCT {{
    feature: node.feature,
    depth: length(path),
    is_leaf: NOT (node)-[:HAS_CHILD]->()
}})[0..10] as sample_nodes
RETURN root.llm_tag as taxonomy_name,
       root.feature as root_feature,
       sample_nodes""",

                    'sample_features': f"""
// Sample features for screenshots: {tag} (Max 8 features)
MATCH (root:MiniTaxonomyNode {{id: '{root_id}'}})
WHERE NOT (()-[:HAS_CHILD]->(root))
MATCH (root)-[:HAS_CHILD*]->(leaf:MiniTaxonomyNode)
WHERE NOT (leaf)-[:HAS_CHILD]->() AND leaf.feature IS NOT NULL
WITH root, collect(DISTINCT leaf.feature) as all_features
RETURN root.llm_tag as taxonomy_name,
       size(all_features) as total_features,
       all_features[0..8] as sample_features""",

                    'compact_hierarchy': f"""
// Compact hierarchy for: {tag} (Screenshot ready)
MATCH (root:MiniTaxonomyNode {{id: '{root_id}'}})
WHERE NOT (()-[:HAS_CHILD]->(root))
OPTIONAL MATCH (root)-[:HAS_CHILD]->(level1:MiniTaxonomyNode)
OPTIONAL MATCH (level1)-[:HAS_CHILD]->(level2:MiniTaxonomyNode)
WHERE NOT (level2)-[:HAS_CHILD]->()
WITH root, level1, collect(level2.feature)[0..3] as level2_sample
RETURN root.llm_tag as root_tag,
       level1.feature as level1_feature,
       level2_sample
ORDER BY level1.feature
LIMIT 5""",

                    'leaf_features': f"""
// Top 10 leaf features for: {tag}
MATCH (root:MiniTaxonomyNode {{id: '{root_id}'}})
WHERE NOT (()-[:HAS_CHILD]->(root))
MATCH (root)-[:HAS_CHILD*]->(leaf:MiniTaxonomyNode)
WHERE NOT (leaf)-[:HAS_CHILD]->() AND leaf.feature IS NOT NULL
RETURN root.llm_tag as taxonomy_tag,
       collect(DISTINCT leaf.feature)[0..10] as top_features,
       count(DISTINCT leaf) as total_leaf_count""",

                    'tree_depth': f"""
// Tree depth and structure metrics for: {tag}
MATCH (root:MiniTaxonomyNode {{id: '{root_id}'}})
WHERE NOT (()-[:HAS_CHILD]->(root))
OPTIONAL MATCH path = (root)-[:HAS_CHILD*]->(leaf)
WHERE NOT (leaf)-[:HAS_CHILD]->()
WITH root, 
     max(length(path)) as max_depth,
     count(DISTINCT leaf) as leaf_count,
     avg(length(path)) as avg_depth
RETURN root.llm_tag as taxonomy_tag,
       max_depth,
       leaf_count,
       round(avg_depth, 2) as average_depth""",

                    'parent_child_relationships': f"""
// Parent-child relationships for: {tag}
MATCH (parent:MiniTaxonomyNode)-[:HAS_CHILD]->(child:MiniTaxonomyNode)
WHERE parent.id = '{root_id}' OR 
      (parent)-[:HAS_CHILD*0..]-(:MiniTaxonomyNode {{id: '{root_id}'}})
RETURN parent.id as parent_id,
       parent.feature as parent_feature,
       parent.is_leaf as parent_is_leaf,
       child.id as child_id,
       child.feature as child_feature,
       child.is_leaf as child_is_leaf
ORDER BY parent.feature, child.feature"""
                }
            }

            queries['individual_queries'].append(individual_query)

        # Batch queries for analyzing multiple taxonomies
        root_ids_str = "', '".join(root_ids)

        queries['batch_queries'] = {
            'all_selected_taxonomies': f"""
// All selected taxonomies overview
MATCH (root:MiniTaxonomyNode)
WHERE root.id IN ['{root_ids_str}'] AND NOT (()-[:HAS_CHILD]->(root))
OPTIONAL MATCH (root)-[:HAS_CHILD*]->(leaf:MiniTaxonomyNode)
WHERE NOT (leaf)-[:HAS_CHILD]->()
WITH root,
     count(DISTINCT leaf) as leaf_count,
     collect(DISTINCT leaf.feature) as leaf_features
OPTIONAL MATCH path = (root)-[:HAS_CHILD*]->(descendant)
WITH root, leaf_count, leaf_features,
     max(length(path)) as max_depth
RETURN root.id as root_id,
       root.llm_tag as tag,
       root.session_id as session_id,
       max_depth as depth,
       leaf_count as leaves,
       leaf_features
ORDER BY root.llm_tag""",

            'feature_overlap_analysis': f"""
// Feature overlap analysis between selected taxonomies
MATCH (root1:MiniTaxonomyNode)-[:HAS_CHILD*]->(leaf1:MiniTaxonomyNode)
WHERE root1.id IN ['{root_ids_str}'] AND NOT (()-[:HAS_CHILD]->(root1)) AND NOT (leaf1)-[:HAS_CHILD]->()
MATCH (root2:MiniTaxonomyNode)-[:HAS_CHILD*]->(leaf2:MiniTaxonomyNode)
WHERE root2.id IN ['{root_ids_str}'] AND NOT (()-[:HAS_CHILD]->(root2)) AND NOT (leaf2)-[:HAS_CHILD]->()
      AND root1.id < root2.id AND leaf1.feature = leaf2.feature
RETURN root1.llm_tag as taxonomy1,
       root2.llm_tag as taxonomy2,
       collect(DISTINCT leaf1.feature) as shared_features,
       count(DISTINCT leaf1.feature) as overlap_count
ORDER BY overlap_count DESC""",

            'session_analysis': f"""
// Session analysis for selected taxonomies
MATCH (root:MiniTaxonomyNode)
WHERE root.id IN ['{root_ids_str}'] AND NOT (()-[:HAS_CHILD]->(root))
WITH root.session_id as session_id, collect(root.llm_tag) as taxonomy_tags, count(*) as count
RETURN session_id,
       count as taxonomies_in_session,
       taxonomy_tags
ORDER BY count DESC""",

            'comparative_metrics': f"""
// Comparative metrics for selected taxonomies
MATCH (root:MiniTaxonomyNode)
WHERE root.id IN ['{root_ids_str}'] AND NOT (()-[:HAS_CHILD]->(root))
OPTIONAL MATCH (root)-[:HAS_CHILD*]->(leaf:MiniTaxonomyNode)
WHERE NOT (leaf)-[:HAS_CHILD]->()
WITH root, count(DISTINCT leaf) as leaf_count
OPTIONAL MATCH path = (root)-[:HAS_CHILD*]->(descendant)
WITH root, leaf_count, max(length(path)) as max_depth
RETURN root.llm_tag as taxonomy_tag,
       root.id as root_id,
       leaf_count,
       max_depth,
       CASE 
         WHEN leaf_count = 0 THEN 'Empty'
         WHEN leaf_count <= 5 THEN 'Small' 
         WHEN leaf_count <= 20 THEN 'Medium'
         WHEN leaf_count <= 50 THEN 'Large'
         ELSE 'Very Large'
       END as size_category,
       CASE
         WHEN max_depth <= 2 THEN 'Shallow'
         WHEN max_depth <= 5 THEN 'Medium'
         WHEN max_depth <= 8 THEN 'Deep'
         ELSE 'Very Deep'
       END as depth_category
ORDER BY leaf_count DESC, max_depth DESC"""
        }

        return queries

    def save_results(self, results, filename_prefix="taxonomy_analysis"):
        """Save analysis results to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate Cypher queries for best and worst taxonomies
        logger.info("Generating Cypher queries...")
        best_queries = self.generate_cypher_queries(
            results['best_taxonomies']['taxonomies'],
            "best_taxonomies"
        )
        worst_queries = self.generate_cypher_queries(
            results['worst_taxonomies']['taxonomies'],
            "worst_taxonomies"
        )

        # Save complete results
        complete_file = f"{filename_prefix}_complete_{timestamp}.json"
        with open(complete_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save best taxonomies only
        best_file = f"{filename_prefix}_best_20_{timestamp}.json"
        with open(best_file, 'w') as f:
            json.dump(results['best_taxonomies'], f, indent=2, default=str)

        # Save worst taxonomies only
        worst_file = f"{filename_prefix}_worst_20_{timestamp}.json"
        with open(worst_file, 'w') as f:
            json.dump(results['worst_taxonomies'], f, indent=2, default=str)

        # Save Cypher queries for best taxonomies
        best_queries_file = f"{filename_prefix}_best_cypher_queries_{timestamp}.json"
        with open(best_queries_file, 'w') as f:
            json.dump(best_queries, f, indent=2, default=str)

        # Save Cypher queries for worst taxonomies
        worst_queries_file = f"{filename_prefix}_worst_cypher_queries_{timestamp}.json"
        with open(worst_queries_file, 'w') as f:
            json.dump(worst_queries, f, indent=2, default=str)

        # Create a readable Cypher queries file
        readable_queries_file = f"{filename_prefix}_cypher_queries_{timestamp}.txt"
        with open(readable_queries_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CYPHER QUERIES FOR TOP 20 BEST TAXONOMIES\n")
            f.write("=" * 80 + "\n\n")

            for query in best_queries['individual_queries'][:10]:  # Top 10 for readability
                f.write(f"TAXONOMY #{query['taxonomy_rank']}: {query['tag']}\n")
                f.write(f"Root ID: {query['root_id']}\n")
                f.write(f"Quality Score: {query['quality_score']:.2f}\n")
                f.write("-" * 60 + "\n")
                f.write(query['queries']['full_tree'])
                f.write("\n" + "-" * 60 + "\n")
                f.write(query['queries']['leaf_features'])
                f.write("\n" + "=" * 80 + "\n\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("BATCH QUERIES FOR ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            for query_name, query_text in best_queries['batch_queries'].items():
                f.write(f"QUERY: {query_name.upper()}\n")
                f.write("-" * 60 + "\n")
                f.write(query_text)
                f.write("\n" + "=" * 80 + "\n\n")

        logger.info(f"Results saved to: {complete_file}, {best_file}, {worst_file}")
        logger.info(f"Cypher queries saved to: {best_queries_file}, {worst_queries_file}, {readable_queries_file}")
        return complete_file, best_file, worst_file, best_queries_file, worst_queries_file, readable_queries_file
        """Save analysis results to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save complete results
        complete_file = f"{filename_prefix}_complete_{timestamp}.json"
        with open(complete_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save best taxonomies only
        best_file = f"{filename_prefix}_best_20_{timestamp}.json"
        with open(best_file, 'w') as f:
            json.dump(results['best_taxonomies'], f, indent=2, default=str)

        # Save worst taxonomies only
        worst_file = f"{filename_prefix}_worst_20_{timestamp}.json"
        with open(worst_file, 'w') as f:
            json.dump(results['worst_taxonomies'], f, indent=2, default=str)

        logger.info(f"Results saved to: {complete_file}, {best_file}, {worst_file}")
        return complete_file, best_file, worst_file


def main():
    try:
        logger.info("Starting taxonomy analysis...")
        analyzer = WorkingTaxonomyAnalyzer()

        # Extract and analyze taxonomies
        results = analyzer.extract_best_worst_taxonomies()

        if results:
            # Save results
            logger.info("Saving results...")
            files = analyzer.save_results(results)

            # Print summary
            print("\n=== TAXONOMY ANALYSIS SUMMARY ===")
            print(f"Total taxonomies analyzed: {results['metadata']['total_taxonomies_analyzed']}")
            print(f"Best taxonomies found: {results['metadata']['best_taxonomies_count']}")
            print(f"Worst taxonomies found: {results['metadata']['worst_taxonomies_count']}")
            print(f"Mean quality score: {results['quality_distribution']['mean_score']:.2f}")
            print(
                f"Score range: {results['quality_distribution']['min_score']:.2f} - {results['quality_distribution']['max_score']:.2f}")

            print("\n=== TOP 10 BEST TAXONOMIES (Overall) ===")
            for i, taxonomy in enumerate(results['best_taxonomies']['taxonomies'][:10]):
                tag = taxonomy.get('tag', 'N/A')
                depth = taxonomy.get('depth', 'N/A')
                leaves = taxonomy.get('leaves', 'N/A')
                score = taxonomy.get('quality_score', 'N/A')
                print(f"{i + 1:2d}. {tag[:60]:<60} (D:{depth:2d}, L:{leaves:3d}, S:{score:6.1f})")

            print("\n=== TOP 10 BEST SMALL TAXONOMIES (≤50 features, Perfect for Screenshots) ===")
            for i, taxonomy in enumerate(results['small_taxonomies']['taxonomies'][:10]):
                tag = taxonomy.get('tag', 'N/A')
                depth = taxonomy.get('depth', 'N/A')
                leaves = taxonomy.get('leaves', 'N/A')
                score = taxonomy.get('quality_score', 'N/A')
                print(f"{i + 1:2d}. {tag[:60]:<60} (D:{depth:2d}, L:{leaves:3d}, S:{score:6.1f})")

            print("\n=== TOP 10 WORST TAXONOMIES ===")
            for i, taxonomy in enumerate(results['worst_taxonomies']['taxonomies'][-10:]):
                tag = taxonomy.get('tag', 'N/A')
                depth = taxonomy.get('depth', 'N/A')
                leaves = taxonomy.get('leaves', 'N/A')
                score = taxonomy.get('quality_score', 'N/A')
                print(f"{i + 1:2d}. {tag[:60]:<60} (D:{depth:2d}, L:{leaves:3d}, S:{score:6.1f})")

            # Show semantic analysis summary
            best_analysis = results['best_taxonomies']['semantic_analysis']
            worst_analysis = results['worst_taxonomies']['semantic_analysis']

            print(f"\n=== SEMANTIC ANALYSIS SUMMARY ===")
            print(f"Similar tag pairs (best): {len(best_analysis.get('tag_clusters', []))}")
            print(f"Similar tag pairs (worst): {len(worst_analysis.get('tag_clusters', []))}")
            print(
                f"Most common words in best tags: {best_analysis.get('feature_patterns', {}).get('most_common_words', [])[:5]}")
            print(
                f"Most common words in worst tags: {worst_analysis.get('feature_patterns', {}).get('most_common_words', [])[:5]}")

            print(f"\n=== FILES CREATED ===")
            for file in files:
                print(f"- {file}")

            print(f"\n=== CYPHER QUERIES ===")
            print("Ready-to-use Cypher queries have been generated!")
            print("Use these files:")
            print(f"- JSON format: {files[3]} (best), {files[4]} (worst)")
            print(f"- Readable format: {files[5]}")
            print("\nExample queries to run in Neo4j Browser:")
            print("1. Full tree structure for top taxonomy")
            print("2. Leaf features analysis")
            print("3. Feature overlap between taxonomies")
            print("4. Comparative metrics")

        else:
            logger.error("Analysis failed - no results generated")
            print("Analysis failed. Check the logs above for details.")

    except Exception as e:
        logger.error(f"Analysis failed with exception: {e}", exc_info=True)
        print(f"Analysis failed with error: {e}")


if __name__ == "__main__":
    main()