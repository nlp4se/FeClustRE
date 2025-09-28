import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def load_session_data(session_file: str) -> Dict[str, Any]:
    with open(session_file, 'r') as f:
        return json.load(f)


def extract_and_summarize_metrics(session_data: Dict[str, Any]) -> Dict[str, Any]:
    all_metrics = []

    apps_processed = session_data.get('apps_processed', {})

    for config_key, config in apps_processed.items():
        processing_results = config.get('processing_results', {}).get('results', {})

        for app_name, result in processing_results.items():
            candidates = result.get('clustering_results', {}).get('candidates', [])

            if candidates:
                best_candidate = candidates[0]
                summary = best_candidate.get('summary', {})
                candidate_metrics = summary.get('metrics', {})

                embedding_type = config.get('embedding_type', 'allmini')

                all_metrics.append({
                    'model_type': config['model_type'],
                    'embedding_type': embedding_type,
                    'sample_size': config['sample_size'],
                    'selection_strategy': config['selection_strategy'],
                    'silhouette_score': candidate_metrics.get('silhouette_score', 0),
                    'davies_bouldin_score': candidate_metrics.get('davies_bouldin_score', float('inf')),
                    'n_clusters': summary.get('n_clusters', 0),
                    'processing_time': config.get('processing_time', 0),
                    'unique_features': result.get('unique_features', 0)
                })

    # Group by configuration
    config_results = {}

    for metric in all_metrics:
        key = (metric['model_type'], metric['embedding_type'],
               metric['sample_size'], metric['selection_strategy'])

        if key not in config_results:
            config_results[key] = {
                'silhouette_scores': [],
                'davies_bouldin_scores': [],
                'n_clusters': [],
                'processing_times': [],
                'unique_features': [],
                'experiments': 0
            }

        config_results[key]['silhouette_scores'].append(metric['silhouette_score'])
        config_results[key]['n_clusters'].append(metric['n_clusters'])
        config_results[key]['processing_times'].append(metric['processing_time'])
        config_results[key]['unique_features'].append(metric['unique_features'])
        config_results[key]['experiments'] += 1

        # Handle Davies-Bouldin scores
        db_score = metric['davies_bouldin_score']
        if db_score != float('inf') and not np.isnan(db_score):
            config_results[key]['davies_bouldin_scores'].append(db_score)

    return config_results


def generate_comprehensive_table(config_results: Dict) -> str:

    latex = """\\begin{table*}[htbp]
\\centering
\\caption{Comprehensive Experimental Results: Feature Clustering Performance Across Model Configurations}
\\label{tab:experimental_results}
\\footnotesize
\\begin{tabular}{|l|l|l|l|c|c|c|c|c|}
\\hline
\\textbf{Model} & \\textbf{Embedding} & \\textbf{Sample} & \\textbf{Strategy} & \\textbf{Exps} & \\textbf{Silhouette} & \\textbf{Davies-B} & \\textbf{Clusters} & \\textbf{Time (s)} \\\\
\\hline
"""

    # Sort configurations for logical grouping
    sorted_configs = sorted(config_results.items(),
                            key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3]))

    current_model = None
    current_embedding = None

    for (model_type, embedding_type, sample_size, strategy), data in sorted_configs:
        # Add separator lines between different model/embedding combinations
        if current_model != model_type or current_embedding != embedding_type:
            if current_model is not None:
                latex += "\\hline\n"
            current_model = model_type
            current_embedding = embedding_type

        # Format display names
        model_display = {
            't-frex': 'T-FREX',
            'transfeatex': 'TransfeatEx',
            'hybrid': 'Hybrid'
        }.get(model_type, model_type.upper())

        embedding_display = {
            'allmini': 'AllMiniLM',
            'sentence-t5': 'Sentence-T5'
        }.get(embedding_type, embedding_type)

        # Format sample size
        if sample_size >= 1000000:
            sample_display = f"{sample_size // 1000000}M"
        elif sample_size >= 1000:
            sample_display = f"{sample_size // 1000}K"
        else:
            sample_display = str(sample_size)

        strategy_display = strategy.capitalize()

        # Calculate statistics
        avg_silhouette = np.mean(data['silhouette_scores'])
        std_silhouette = np.std(data['silhouette_scores'])

        if data['davies_bouldin_scores']:
            avg_db = np.mean(data['davies_bouldin_scores'])
            db_display = f"{avg_db:.2f}"
        else:
            db_display = "N/A"

        avg_clusters = np.mean(data['n_clusters'])
        avg_time = np.mean(data['processing_times'])

        # Format silhouette with std dev
        silhouette_display = f"{avg_silhouette:.3f}$\\pm${std_silhouette:.2f}"

        latex += f"{model_display} & {embedding_display} & {sample_display} & {strategy_display} & {data['experiments']} & {silhouette_display} & {db_display} & {avg_clusters:.1f} & {avg_time:.1f} \\\\\n"

    latex += """\\hline
\\end{tabular}
\\begin{tablenotes}
\\footnotesize
\\item \\textbf{Model}: Feature extraction method (T-FREX, TransfeatEx, Hybrid)
\\item \\textbf{Embedding}: Sentence embedding model (AllMiniLM-L6-v2, Sentence-T5)  
\\item \\textbf{Sample}: Dataset sample size (K=thousands, M=millions)
\\item \\textbf{Strategy}: Clustering selection strategy (Balanced, Silhouette, Conservative)
\\item \\textbf{Exps}: Number of app experiments per configuration
\\item \\textbf{Silhouette}: Mean silhouette score ± standard deviation (higher better)
\\item \\textbf{Davies-B}: Davies-Bouldin index (lower better, N/A if undefined)
\\item \\textbf{Clusters}: Average number of clusters generated
\\item \\textbf{Time}: Average processing time in seconds
\\end{tablenotes}
\\end{table*}
"""

    return latex


def generate_summary_stats(config_results: Dict) -> str:
    """Generate summary statistics as LaTeX comments."""

    all_silhouette = []
    all_db = []
    all_times = []
    total_experiments = 0

    for data in config_results.values():
        all_silhouette.extend(data['silhouette_scores'])
        all_db.extend(data['davies_bouldin_scores'])
        all_times.extend(data['processing_times'])
        total_experiments += data['experiments']

    best_silhouette = max(all_silhouette)
    worst_silhouette = min(all_silhouette)
    avg_silhouette = np.mean(all_silhouette)

    if all_db:
        best_db = min(all_db)
        worst_db = max(all_db)
        avg_db = np.mean(all_db)
    else:
        best_db = worst_db = avg_db = "N/A"

    fastest_time = min(all_times)
    slowest_time = max(all_times)
    avg_time = np.mean(all_times)

    unique_configs = len(config_results)

    summary = f"""% EXPERIMENTAL SUMMARY STATISTICS
% Total configurations tested: {unique_configs}
% Total app experiments: {total_experiments}
% Silhouette scores - Best: {best_silhouette:.3f}, Worst: {worst_silhouette:.3f}, Avg: {avg_silhouette:.3f}
% Davies-Bouldin scores - Best: {best_db}, Worst: {worst_db}, Avg: {avg_db}
% Processing times - Fastest: {fastest_time:.1f}s, Slowest: {slowest_time:.1f}s, Avg: {avg_time:.1f}s

"""

    return summary


def main():
    if len(sys.argv) != 2:
        print("Usage: python session_to_latex.py <session_file.json|session_file.json.gz>", file=sys.stderr)
        sys.exit(1)

    session_file = sys.argv[1]

    if not Path(session_file).exists():
        print(f"Error: Session file '{session_file}' not found", file=sys.stderr)
        sys.exit(1)

    try:
        session_data = load_session_data(session_file)
        config_results = extract_and_summarize_metrics(session_data)

        if not config_results:
            print("Error: No experimental results found in session data", file=sys.stderr)
            sys.exit(1)

        # Generate summary stats as comments
        summary_stats = generate_summary_stats(config_results)

        # Generate the comprehensive table
        latex_table = generate_comprehensive_table(config_results)

        # Output everything
        print(summary_stats)
        print(latex_table)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()