import json
import pandas as pd
import numpy as np
import requests
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystematicTester:
    def __init__(self, base_url='http://localhost:3000', cache_dir='test_cache'):
        self.base_url = base_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Results storage
        self.results_dir = Path('evaluation_results')
        self.results_dir.mkdir(exist_ok=True)

        # Session tracking
        self.session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_file = self.results_dir / f"{self.session_id}.json"

        # Initialize session data
        self.session_data = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'apps_processed': {},
            'evaluation_metrics': {},
            'configurations': []
        }

    def save_session(self):
        with open(self.session_file, 'w') as f:
            json.dump(self.session_data, f, indent=2, default=str)

    def load_session(self, session_file):
        with open(session_file, 'r') as f:
            self.session_data = json.load(f)
        self.session_id = self.session_data['session_id']
        logger.info(f"Loaded session: {self.session_id}")

    def process_csv_smart(self, csv_file, model_type='tfrex', sample_size=None, skip_processing=False):
        csv_path = Path(csv_file)
        cache_key = f"{csv_path.stem}_{model_type}_{sample_size}_{hash(csv_path.stat().st_mtime)}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check cache first
        if cache_file.exists() and not skip_processing:
            logger.info(f"Loading cached results for {csv_path.name}")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Read and sample CSV if needed
        logger.info(f"Reading CSV: {csv_path.name}")
        df = pd.read_csv(csv_path)

        if sample_size and len(df) > sample_size:
            logger.info(f"Sampling {sample_size} from {len(df)} rows")
            if 'app_name' in df.columns:
                df = df.groupby('app_name').apply(
                    lambda x: x.sample(min(len(x), max(1, sample_size // df['app_name'].nunique())))
                ).reset_index(drop=True)
            else:
                df = df.sample(sample_size)

        # Save sampled data
        temp_file = self.cache_dir / f"temp_{cache_key}.csv"
        df.to_csv(temp_file, index=False)

        try:
            logger.info(f"Processing {len(df)} rows with model {model_type}")
            with open(temp_file, 'rb') as f:
                files = {'file': (csv_path.name, f, 'text/csv')}
                params = {'model_type': model_type}

                response = requests.post(
                    f"{self.base_url}/process_reviews/upload",
                    files=files,
                    params=params,
                    timeout=None
                )

            if response.ok:
                results = response.json()

                # Cache results
                with open(cache_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                # Clean up temp file
                temp_file.unlink()

                return results
            else:
                logger.error(f"Processing failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            return None

    def auto_select_best_candidate(self, app_results, selection_strategy='balanced'):
        best_selections = {}

        for app_name, result in app_results.items():
            candidates = result.get('clustering_results', {}).get('candidates', [])

            if not candidates:
                logger.warning(f"No candidates for {app_name}")
                continue

            # Score candidates based on strategy
            scored_candidates = []
            for i, candidate in enumerate(candidates):
                metrics = candidate.get('summary', {}).get('metrics', {})

                silhouette = metrics.get('silhouette_score', 0)
                davies_bouldin = 1 / (1 + metrics.get('davies_bouldin_score', 1))  # Invert DB score
                n_clusters = candidate.get('summary', {}).get('n_clusters', 1)
                avg_size = candidate.get('summary', {}).get('avg_cluster_size', 1)

                if selection_strategy == 'silhouette':
                    score = silhouette
                elif selection_strategy == 'balanced':
                    cluster_penalty = 1 / (1 + abs(n_clusters - 5))  # Prefer ~5 clusters
                    size_penalty = 1 / (1 + abs(avg_size - 10))  # Prefer ~10 features per cluster
                    score = (silhouette * 0.4 + davies_bouldin * 0.3 +
                             cluster_penalty * 0.15 + size_penalty * 0.15)
                elif selection_strategy == 'conservative':
                    # Prefer fewer, larger clusters
                    score = silhouette * (avg_size / max(n_clusters, 1))
                else:
                    score = silhouette  # Default to silhouette

                scored_candidates.append((i, score, candidate))

            # Select best candidate
            best_idx, best_score, best_candidate = max(scored_candidates, key=lambda x: x[1])
            best_selections[app_name] = {
                'candidate_index': best_idx,
                'score': best_score,
                'candidate': best_candidate,
                'selection_strategy': selection_strategy
            }

            logger.info(f"{app_name}: Selected candidate {best_idx} (score: {best_score:.4f})")

        return best_selections

    def save_selected_clusterings(self, best_selections):
        saved_results = {}

        for app_name, selection in best_selections.items():
            try:
                clustering_data = selection['candidate']['clustering']

                response = requests.post(
                    f"{self.base_url}/save_selected_clustering/{app_name}",
                    json={"clustering": clustering_data},
                    timeout=None
                )

                if response.ok:
                    result = response.json()
                    saved_results[app_name] = {
                        'status': 'success',
                        'result': result,
                        'selection_info': selection
                    }
                    logger.info(f"Saved clustering for {app_name}")
                else:
                    saved_results[app_name] = {
                        'status': 'error',
                        'error': response.text,
                        'selection_info': selection
                    }
                    logger.error(f"Failed to save {app_name}: {response.text}")

            except Exception as e:
                saved_results[app_name] = {
                    'status': 'error',
                    'error': str(e),
                    'selection_info': selection
                }
                logger.error(f"Error saving {app_name}: {e}")

        return saved_results

    def run_full_pipeline(self, csv_files, model_types=['tfrex'], sample_sizes=[1000],
                          selection_strategies=['balanced'], skip_existing=True):

        all_configurations = []

        for csv_file in csv_files:
            for model_type in model_types:
                for sample_size in sample_sizes:
                    for strategy in selection_strategies:
                        config = {
                            'csv_file': str(csv_file),
                            'model_type': model_type,
                            'sample_size': sample_size,
                            'selection_strategy': strategy,
                            'timestamp': datetime.now().isoformat()
                        }

                        config_key = f"{Path(csv_file).stem}_{model_type}_{sample_size}_{strategy}"

                        if skip_existing and config_key in self.session_data.get('apps_processed', {}):
                            logger.info(f"Skipping existing configuration: {config_key}")
                            continue

                        logger.info(f"Processing configuration: {config_key}")

                        # Step 1: Process CSV
                        start_time = time.time()
                        processing_results = self.process_csv_smart(
                            csv_file, model_type, sample_size
                        )
                        processing_time = time.time() - start_time

                        if not processing_results:
                            logger.error(f"Failed to process {config_key}")
                            continue

                        # Step 2: Auto-select best candidates
                        app_results = processing_results.get('results', {})
                        best_selections = self.auto_select_best_candidate(
                            app_results, strategy
                        )

                        # Step 3: Save selected clusterings
                        saved_results = self.save_selected_clusterings(best_selections)

                        # Store configuration results
                        config.update({
                            'processing_time': processing_time,
                            'apps_found': len(app_results),
                            'successful_selections': len([r for r in saved_results.values()
                                                          if r['status'] == 'success']),
                            'processing_results': processing_results,
                            'best_selections': best_selections,
                            'saved_results': saved_results
                        })

                        self.session_data['apps_processed'][config_key] = config
                        all_configurations.append(config)

                        self.save_session()

                        logger.info(f"Completed {config_key} in {processing_time:.2f}s")

        self.session_data['configurations'] = all_configurations
        self.save_session()

        return all_configurations

    def evaluate_clustering_quality(self):
        logger.info("Starting comprehensive clustering quality evaluation...")

        evaluation_results = {
            'internal_metrics': {},
            'external_metrics': {},
            'app_comparisons': {},
            'threshold_analysis': {},
            'cohesion_separation': {},
            'temporal_analysis': {},
            'timestamp': datetime.now().isoformat()
        }

        # Collect all processed configurations
        all_configs = self.session_data.get('apps_processed', {})

        # Group by app for comparison
        apps_data = defaultdict(list)
        for config_key, config in all_configs.items():
            csv_name = Path(config['csv_file']).stem
            apps_data[csv_name].append(config)

        # Internal clustering quality metrics
        internal_metrics = []
        threshold_data = []
        cohesion_data = []
        separation_data = []

        for csv_name, configs in apps_data.items():
            for config in configs:
                app_results = config.get('processing_results', {}).get('results', {})

                for app_name, result in app_results.items():
                    candidates = result.get('clustering_results', {}).get('candidates', [])

                    for i, candidate in enumerate(candidates):
                        summary = candidate.get('summary', {})
                        metrics = summary.get('metrics', {})
                        clustering = candidate.get('clustering', {})

                        # Calculate cohesion and separation
                        cohesion, separation = self._calculate_cohesion_separation(clustering)

                        metric_data = {
                            'csv_file': csv_name,
                            'app_name': app_name,
                            'model_type': config['model_type'],
                            'sample_size': config['sample_size'],
                            'candidate_index': i,
                            'threshold': summary.get('threshold', 0),
                            'n_clusters': summary.get('n_clusters', 0),
                            'avg_cluster_size': summary.get('avg_cluster_size', 0),
                            'silhouette_score': metrics.get('silhouette_score', 0),
                            'davies_bouldin_score': metrics.get('davies_bouldin_score', float('inf')),
                            'cohesion': cohesion,
                            'separation': separation
                        }

                        internal_metrics.append(metric_data)
                        threshold_data.append(metric_data)

                        if cohesion is not None:
                            cohesion_data.append({
                                'csv_file': csv_name,
                                'model_type': config['model_type'],
                                'cohesion': cohesion,
                                'separation': separation,
                                'app_name': app_name
                            })

        if internal_metrics:
            import pandas as pd
            metrics_df = pd.DataFrame(internal_metrics)
            metrics_df = metrics_df.replace([float('inf'), -float('inf')], [999999, -999999])

            evaluation_results['internal_metrics'] = {
                'summary_statistics': {
                    'silhouette': {
                        'mean': float(metrics_df['silhouette_score'].mean()),
                        'std': float(metrics_df['silhouette_score'].std()),
                        'min': float(metrics_df['silhouette_score'].min()),
                        'max': float(metrics_df['silhouette_score'].max())
                    },
                    'davies_bouldin': {
                        'mean': float(
                            metrics_df[metrics_df['davies_bouldin_score'] != 999999]['davies_bouldin_score'].mean()),
                        'std': float(
                            metrics_df[metrics_df['davies_bouldin_score'] != 999999]['davies_bouldin_score'].std()),
                        'min': float(
                            metrics_df[metrics_df['davies_bouldin_score'] != 999999]['davies_bouldin_score'].min()),
                        'max': float(
                            metrics_df[metrics_df['davies_bouldin_score'] != 999999]['davies_bouldin_score'].max())
                    },
                    'cohesion': {
                        'mean': float(metrics_df[metrics_df['cohesion'].notna()]['cohesion'].mean()),
                        'std': float(metrics_df[metrics_df['cohesion'].notna()]['cohesion'].std()),
                        'min': float(metrics_df[metrics_df['cohesion'].notna()]['cohesion'].min()),
                        'max': float(metrics_df[metrics_df['cohesion'].notna()]['cohesion'].max())
                    },
                    'separation': {
                        'mean': float(metrics_df[metrics_df['separation'].notna()]['separation'].mean()),
                        'std': float(metrics_df[metrics_df['separation'].notna()]['separation'].std()),
                        'min': float(metrics_df[metrics_df['separation'].notna()]['separation'].min()),
                        'max': float(metrics_df[metrics_df['separation'].notna()]['separation'].max())
                    }
                },
                'by_model': self._convert_groupby_to_dict(metrics_df.groupby('model_type').agg({
                    'silhouette_score': ['mean', 'std'],
                    'davies_bouldin_score': ['mean', 'std'],
                    'n_clusters': ['mean', 'std'],
                    'cohesion': ['mean', 'std'],
                    'separation': ['mean', 'std']
                })),
                'correlation_matrix': metrics_df[['silhouette_score', 'davies_bouldin_score',
                                                  'n_clusters', 'avg_cluster_size', 'threshold',
                                                  'cohesion', 'separation']].corr().to_dict()
            }

            # Threshold analysis for optimal threshold finding
            evaluation_results['threshold_analysis'] = self._analyze_threshold_evolution(metrics_df)

        # Enhanced external metrics
        external_metrics = {
            'cluster_size_distribution': {},
            'app_diversity': {},
            'cluster_counts': {},
            'optimal_thresholds': {}
        }

        for csv_name, configs in apps_data.items():
            sizes = []
            cluster_counts = []
            thresholds = []

            for config in configs:
                best_selections = config.get('best_selections', {})
                for app_name, selection in best_selections.items():
                    clustering = selection['candidate']['clustering']
                    clusters = clustering.get('clusters', {})
                    sizes.extend([len(features) for features in clusters.values()])
                    cluster_counts.append(len(clusters))
                    thresholds.append(selection['candidate']['summary'].get('threshold', 0))

            if sizes:
                import numpy as np
                external_metrics['cluster_size_distribution'][csv_name] = {
                    'mean': float(np.mean(sizes)),
                    'median': float(np.median(sizes)),
                    'std': float(np.std(sizes)),
                    'min': int(np.min(sizes)),
                    'max': int(np.max(sizes)),
                    'percentiles': {
                        '25': float(np.percentile(sizes, 25)),
                        '75': float(np.percentile(sizes, 75)),
                        '90': float(np.percentile(sizes, 90))
                    }
                }

                external_metrics['cluster_counts'][csv_name] = {
                    'mean': float(np.mean(cluster_counts)),
                    'std': float(np.std(cluster_counts)),
                    'min': int(np.min(cluster_counts)),
                    'max': int(np.max(cluster_counts))
                }

                external_metrics['optimal_thresholds'][csv_name] = {
                    'mean': float(np.mean(thresholds)),
                    'std': float(np.std(thresholds)),
                    'most_common': float(max(set(thresholds), key=thresholds.count))
                }

        evaluation_results['external_metrics'] = external_metrics
        evaluation_results['cohesion_separation'] = cohesion_data

        # Store evaluation results
        self.session_data['evaluation_metrics'] = evaluation_results
        self.save_session()

        return evaluation_results

    def _calculate_cohesion_separation(self, clustering):
        try:
            clusters = clustering.get('clusters', {})
            if len(clusters) < 2:
                return None, None

            cluster_sizes = [len(features) for features in clusters.values()]

            # Cohesion: inverse of average cluster size variance
            cohesion = 1.0 / (1.0 + np.var(cluster_sizes))

            # Separation measure how distinct clusters are
            # Ratio of number of clusters to total features
            total_features = sum(cluster_sizes)
            n_clusters = len(clusters)
            separation = n_clusters / np.sqrt(total_features)  # More clusters = better separation

            return float(cohesion), float(separation)

        except Exception as e:
            logger.error(f"Error calculating cohesion/separation: {e}")
            return None, None
    def _analyze_threshold_evolution(self, metrics_df):
        threshold_analysis = {}

        # Group by threshold and calculate average scores
        threshold_groups = metrics_df.groupby('threshold').agg({
            'silhouette_score': ['mean', 'std', 'count'],
            'davies_bouldin_score': ['mean', 'std'],
            'n_clusters': ['mean', 'std'],
            'cohesion': ['mean', 'std'],
            'separation': ['mean', 'std']
        }).round(4)

        # Find optimal threshold based on silhouette score
        threshold_means = metrics_df.groupby('threshold')['silhouette_score'].mean()
        optimal_threshold = threshold_means.idxmax()

        # Calculate elbow point for threshold selection
        thresholds = sorted(threshold_means.index)
        scores = [threshold_means[t] for t in thresholds]

        # Simple elbow detection using second derivative
        elbow_threshold = None
        if len(scores) >= 3:
            second_derivatives = []
            for i in range(1, len(scores) - 1):
                second_deriv = scores[i - 1] - 2 * scores[i] + scores[i + 1]
                second_derivatives.append(abs(second_deriv))

            if second_derivatives:
                elbow_idx = second_derivatives.index(max(second_derivatives)) + 1
                elbow_threshold = thresholds[elbow_idx]

        threshold_analysis = {
            'optimal_threshold': float(optimal_threshold),
            'optimal_silhouette': float(threshold_means[optimal_threshold]),
            'elbow_threshold': float(elbow_threshold) if elbow_threshold else None,
            'threshold_evolution': self._convert_groupby_to_dict(threshold_groups),
            'threshold_range': {
                'min': float(min(thresholds)),
                'max': float(max(thresholds)),
                'count': len(thresholds)
            }
        }

        return threshold_analysis

    def _convert_groupby_to_dict(self, groupby_result):
        result_dict = {}

        for index, row in groupby_result.iterrows():
            key = str(index) if not isinstance(index, str) else index
            row_dict = {}
            for col_tuple, value in row.items():
                # Handle multi-level column names (tuples)
                if isinstance(col_tuple, tuple):
                    col_key = '_'.join(str(x) for x in col_tuple)
                else:
                    col_key = str(col_tuple)
                if hasattr(value, 'item'):
                    row_dict[col_key] = float(value.item()) if not pd.isna(value) else None
                else:
                    row_dict[col_key] = float(value) if not pd.isna(value) else None

            result_dict[key] = row_dict

        return result_dict

    def generate_visualizations(self):
        logger.info("Generating comprehensive evaluation visualizations...")

        viz_dir = self.results_dir / f"visualizations_{self.session_id}"
        viz_dir.mkdir(exist_ok=True)

        # Collect metrics data - TWO SEPARATE COLLECTIONS
        all_configs = self.session_data.get('apps_processed', {})

        all_candidates_metrics = []
        selected_metrics = []

        for config_key, config in all_configs.items():
            app_results = config.get('processing_results', {}).get('results', {})
            best_selections = config.get('best_selections', {})

            # Collect ALL candidates for threshold analysis (no strategy needed)
            for app_name, result in app_results.items():
                candidates = result.get('clustering_results', {}).get('candidates', [])

                for i, candidate in enumerate(candidates):
                    summary = candidate.get('summary', {})
                    metrics = summary.get('metrics', {})
                    clustering = candidate.get('clustering', {})

                    # Calculate cohesion and separation
                    cohesion, separation = self._calculate_cohesion_separation(clustering)

                    all_candidates_metrics.append({
                        'csv_file': Path(config['csv_file']).stem,
                        'app_name': app_name,
                        'model_type': config['model_type'],
                        'threshold': summary.get('threshold', 0),
                        'silhouette_score': metrics.get('silhouette_score', 0),
                        'davies_bouldin_score': metrics.get('davies_bouldin_score', float('inf')),
                        'n_clusters': summary.get('n_clusters', 0),
                        'avg_cluster_size': summary.get('avg_cluster_size', 0),
                        'cohesion': cohesion,
                        'separation': separation
                    })

            for app_name, selection in best_selections.items():
                candidate = selection['candidate']
                summary = candidate.get('summary', {})
                metrics = summary.get('metrics', {})
                clustering = candidate.get('clustering', {})

                cohesion, separation = self._calculate_cohesion_separation(clustering)

                selected_metrics.append({
                    'csv_file': Path(config['csv_file']).stem,
                    'app_name': app_name,
                    'model_type': config['model_type'],
                    'selection_strategy': config.get('selection_strategy', 'unknown'),
                    'threshold': summary.get('threshold', 0),
                    'silhouette_score': metrics.get('silhouette_score', 0),
                    'davies_bouldin_score': metrics.get('davies_bouldin_score', float('inf')),
                    'n_clusters': summary.get('n_clusters', 0),
                    'avg_cluster_size': summary.get('avg_cluster_size', 0),
                    'cohesion': cohesion,
                    'separation': separation,
                    'selection_score': selection.get('score', 0)
                })

        # Create two dataframes
        df_all = pd.DataFrame(all_candidates_metrics)
        df_selected = pd.DataFrame(selected_metrics)

        if df_all.empty and df_selected.empty:
            logger.warning("No data available for visualization")
            return

        # Filter out infinite values
        df_all = df_all[df_all['davies_bouldin_score'] != float('inf')]
        df_selected = df_selected[df_selected['davies_bouldin_score'] != float('inf')]

        # Add model_strategy column to selected data only
        if not df_selected.empty:
            df_selected['model_strategy'] = df_selected['model_type'] + '_' + df_selected['selection_strategy']

        plt.style.use('seaborn-v0_8')

        colors = {'transfeatex': 'blue', 't-frex': 'red', 'tfrex': 'green'}
        markers = {'balanced': 'o', 'silhouette': 's', 'conservative': '^', 'unknown': 'x'}
        linestyles = {'balanced': '-', 'silhouette': '--', 'conservative': ':', 'unknown': '-.'}


        # 1. Threshold Evolution (ALL candidates - shows natural clustering behavior)
        plt.figure(figsize=(10, 8))
        for model in df_all['model_type'].unique():
            model_data = df_all[df_all['model_type'] == model]
            threshold_avg = model_data.groupby('threshold')['silhouette_score'].mean()
            plt.plot(threshold_avg.index, threshold_avg.values,
                     marker='o', label=model,
                     color=colors.get(model, 'gray'),
                     linewidth=2, markersize=6)

            # Mark optimal threshold
            if len(threshold_avg) >= 3:
                optimal_idx = threshold_avg.values.argmax()
                plt.axvline(x=threshold_avg.index[optimal_idx],
                            linestyle='--', alpha=0.3, color=colors.get(model, 'gray'))

        plt.xlabel('Height Threshold')
        plt.ylabel('Average Silhouette Score')
        plt.title('Threshold Evolution by Model (All Candidates)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(viz_dir / 'threshold_evolution_all_candidates.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Elbow Analysis (ALL candidates)
        plt.figure(figsize=(10, 8))
        for model in df_all['model_type'].unique():
            model_data = df_all[df_all['model_type'] == model]
            threshold_clusters = model_data.groupby('threshold')['n_clusters'].mean()
            plt.plot(threshold_clusters.index, threshold_clusters.values,
                     marker='s', label=model,
                     color=colors.get(model, 'gray'),
                     linewidth=2, markersize=6)

        plt.xlabel('Height Threshold')
        plt.ylabel('Average Number of Clusters')
        plt.title('Elbow Analysis: Threshold vs Clusters by Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(viz_dir / 'threshold_vs_clusters_all_candidates.png', dpi=300, bbox_inches='tight')
        plt.close()

        if not df_selected.empty:
            # 3. Silhouette vs Davies-Bouldin (SELECTED only)
            plt.figure(figsize=(12, 8))
            for model in df_selected['model_type'].unique():
                for strategy in df_selected['selection_strategy'].unique():
                    model_data = df_selected[(df_selected['model_type'] == model) &
                                             (df_selected['selection_strategy'] == strategy)]
                    if not model_data.empty:
                        plt.scatter(model_data['silhouette_score'], model_data['davies_bouldin_score'],
                                    label=f'{model}_{strategy}',
                                    alpha=0.7,
                                    s=80,
                                    color=colors.get(model, 'gray'),
                                    marker=markers.get(strategy, 'o'))

            plt.xlabel('Silhouette Score (higher is better)')
            plt.ylabel('Davies-Bouldin Score (lower is better)')
            plt.title('Clustering Quality: Selected Candidates by Model and Strategy')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / 'silhouette_vs_davies_bouldin_selected.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 4. Cohesion vs Separation (SELECTED only)
            plt.figure(figsize=(12, 8))
            cohesion_data = df_selected.dropna(subset=['cohesion', 'separation'])
            if not cohesion_data.empty:
                for model in cohesion_data['model_type'].unique():
                    for strategy in cohesion_data['selection_strategy'].unique():
                        model_data = cohesion_data[(cohesion_data['model_type'] == model) &
                                                   (cohesion_data['selection_strategy'] == strategy)]
                        if not model_data.empty:
                            plt.scatter(model_data['cohesion'], model_data['separation'],
                                        label=f'{model}_{strategy}',
                                        alpha=0.7,
                                        s=80,
                                        color=colors.get(model, 'gray'),
                                        marker=markers.get(strategy, 'o'))

                plt.xlabel('Intra-cluster Cohesion (higher is better)')
                plt.ylabel('Inter-cluster Separation (higher is better)')
                plt.title('Cohesion vs Separation: Selected Candidates by Model and Strategy')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / 'cohesion_vs_separation_selected.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 5. Number of clusters distribution (SELECTED)
            plt.figure(figsize=(12, 8))
            df_selected.boxplot(column='n_clusters', by='model_strategy', ax=plt.gca())
            plt.title('Number of Clusters: Selected Candidates by Model-Strategy')
            plt.suptitle('')
            plt.xlabel('Model_Strategy')
            plt.ylabel('Number of Clusters')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(viz_dir / 'clusters_distribution_selected.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 6. Average cluster size distribution (SELECTED)
            plt.figure(figsize=(12, 8))
            df_selected.boxplot(column='avg_cluster_size', by='model_strategy', ax=plt.gca())
            plt.title('Average Cluster Size: Selected Candidates by Model-Strategy')
            plt.suptitle('')
            plt.xlabel('Model_Strategy')
            plt.ylabel('Average Cluster Size')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(viz_dir / 'cluster_size_distribution_selected.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 7. Strategy Performance Comparison
            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            strategy_performance = df_selected.groupby('selection_strategy')['silhouette_score'].mean().sort_values()
            strategy_performance.plot(kind='barh', color='steelblue')
            plt.xlabel('Average Silhouette Score')
            plt.ylabel('Selection Strategy')
            plt.title('Strategy Performance (Averaged Across Models)')
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            model_performance = df_selected.groupby('model_type')['silhouette_score'].mean().sort_values()
            model_performance.plot(kind='barh', color='coral')
            plt.xlabel('Average Silhouette Score')
            plt.ylabel('Model Type')
            plt.title('Model Performance (Averaged Across Strategies)')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(viz_dir / 'strategy_model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 8. Heatmap of model-strategy performance
            plt.figure(figsize=(10, 8))
            pivot_table = df_selected.pivot_table(values='silhouette_score',
                                                  index='model_type',
                                                  columns='selection_strategy',
                                                  aggfunc='mean')
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis',
                        cbar_kws={'label': 'Silhouette Score'})
            plt.title('Model-Strategy Performance Heatmap (Selected Candidates)')
            plt.ylabel('Model Type')
            plt.xlabel('Selection Strategy')
            plt.tight_layout()
            plt.savefig(viz_dir / 'model_strategy_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 9. Correlation matrices for both datasets
        plt.figure(figsize=(16, 7))

        # Correlation for all candidates
        plt.subplot(1, 2, 1)
        numeric_cols = ['silhouette_score', 'davies_bouldin_score', 'n_clusters',
                        'avg_cluster_size', 'threshold']
        if not df_all['cohesion'].isna().all():
            numeric_cols.extend(['cohesion', 'separation'])

        correlation_matrix = df_all[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, fmt='.3f')
        plt.title('Correlation Matrix: All Candidates')

        # Correlation for selected candidates
        if not df_selected.empty:
            plt.subplot(1, 2, 2)
            correlation_matrix_selected = df_selected[numeric_cols].corr()
            sns.heatmap(correlation_matrix_selected, annot=True, cmap='RdBu_r', center=0,
                        square=True, linewidths=0.5, fmt='.3f')
            plt.title('Correlation Matrix: Selected Candidates')

        plt.tight_layout()
        plt.savefig(viz_dir / 'correlation_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 10. Model comparison radar chart (selected candidates only)
        if not df_selected.empty:
            self._create_model_strategy_comparison_radar(df_selected, viz_dir)

        # 11. Additional detailed visualizations
        self._create_detailed_visualizations_with_strategies(df_all, df_selected, viz_dir)

        logger.info(f"Comprehensive visualizations saved to {viz_dir}")

    def _create_detailed_visualizations_with_strategies(self, df_all, df_selected, viz_dir):
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        df_all['threshold'].hist(bins=20, alpha=0.7, label='All Candidates', color='blue')
        if not df_selected.empty:
            df_selected['threshold'].hist(bins=20, alpha=0.7, label='Selected Candidates', color='red')
        plt.xlabel('Threshold')
        plt.ylabel('Frequency')
        plt.title('Threshold Distribution: All vs Selected')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        df_all.boxplot(column='silhouette_score', by='model_type', ax=plt.gca())
        plt.title('Silhouette Score Distribution by Model (All Candidates)')
        plt.suptitle('')
        plt.xlabel('Model Type')
        plt.ylabel('Silhouette Score')

        plt.tight_layout()
        plt.savefig(viz_dir / 'threshold_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Strategy effectiveness visualization (if selected data exists)
        if not df_selected.empty:
            plt.figure(figsize=(12, 8))

            # Create a scatter plot showing selection score vs silhouette score
            for strategy in df_selected['selection_strategy'].unique():
                strategy_data = df_selected[df_selected['selection_strategy'] == strategy]
                plt.scatter(strategy_data['selection_score'],
                            strategy_data['silhouette_score'],
                            label=strategy, alpha=0.6, s=50)

            plt.xlabel('Selection Score (Strategy-specific)')
            plt.ylabel('Silhouette Score')
            plt.title('Strategy Effectiveness: Selection Score vs Silhouette Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / 'strategy_effectiveness.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Performance by dataset
            if len(df_selected['csv_file'].unique()) > 1:
                plt.figure(figsize=(14, 10))

                # Create subplots for different metrics
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))

                # Silhouette by dataset and strategy
                ax = axes[0, 0]
                df_selected.boxplot(column='silhouette_score',
                                    by=['csv_file', 'selection_strategy'], ax=ax)
                ax.set_title('Silhouette Score by Dataset and Strategy')
                ax.set_xlabel('')
                plt.sca(ax)
                plt.xticks(rotation=45, ha='right')

                # Number of clusters by dataset and strategy
                ax = axes[0, 1]
                df_selected.boxplot(column='n_clusters',
                                    by=['csv_file', 'selection_strategy'], ax=ax)
                ax.set_title('Number of Clusters by Dataset and Strategy')
                ax.set_xlabel('')
                plt.sca(ax)
                plt.xticks(rotation=45, ha='right')

                # Cohesion by dataset and model
                ax = axes[1, 0]
                cohesion_data = df_selected.dropna(subset=['cohesion'])
                if not cohesion_data.empty:
                    cohesion_data.boxplot(column='cohesion',
                                          by=['csv_file', 'model_type'], ax=ax)
                    ax.set_title('Cohesion by Dataset and Model')
                ax.set_xlabel('')
                plt.sca(ax)
                plt.xticks(rotation=45, ha='right')

                # Separation by dataset and model
                ax = axes[1, 1]
                separation_data = df_selected.dropna(subset=['separation'])
                if not separation_data.empty:
                    separation_data.boxplot(column='separation',
                                            by=['csv_file', 'model_type'], ax=ax)
                    ax.set_title('Separation by Dataset and Model')
                ax.set_xlabel('')
                plt.sca(ax)
                plt.xticks(rotation=45, ha='right')

                plt.suptitle('')
                plt.tight_layout()
                plt.savefig(viz_dir / 'performance_by_dataset_detailed.png', dpi=300, bbox_inches='tight')
                plt.close()
    def _create_model_strategy_comparison_radar(self, df, viz_dir):
        try:
            import matplotlib.pyplot as plt
            from math import pi

            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

            # Metrics to compare
            metrics = ['silhouette_score', 'n_clusters', 'avg_cluster_size']
            if not df['cohesion'].isna().all():
                metrics.extend(['cohesion', 'separation'])

            # Normalize Davies-Bouldin
            df_normalized = df.copy()
            df_normalized['davies_bouldin_normalized'] = 1 / (1 + df['davies_bouldin_score'])
            metrics.append('davies_bouldin_normalized')

            # Calculate angles
            angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
            angles += angles[:1]

            # Colors for model-strategy combinations
            color_map = {
                'transfeatex_balanced': 'blue',
                'transfeatex_silhouette': 'lightblue',
                'transfeatex_conservative': 'darkblue',
                't-frex_balanced': 'red',
                't-frex_silhouette': 'lightcoral',
                't-frex_conservative': 'darkred',
                'tfrex_balanced': 'green',
                'tfrex_silhouette': 'lightgreen',
                'tfrex_conservative': 'darkgreen'
            }

            # Plot for each model-strategy combination
            for model_strategy in df['model_strategy'].unique():
                model_data = df_normalized[df_normalized['model_strategy'] == model_strategy]

                # Calculate mean values for each metric
                values = []
                for metric in metrics:
                    if metric in model_data.columns:
                        metric_values = model_data[metric].dropna()
                        if len(metric_values) > 0:
                            # Normalize to 0-1 scale
                            min_val = df_normalized[metric].min()
                            max_val = df_normalized[metric].max()
                            if max_val > min_val:
                                normalized = (metric_values.mean() - min_val) / (max_val - min_val)
                            else:
                                normalized = 0.5
                            values.append(normalized)
                        else:
                            values.append(0)
                    else:
                        values.append(0)

                values += values[:1]  # Complete the circle

                # Plot
                ax.plot(angles, values, 'o-', linewidth=2,
                        label=model_strategy,
                        color=color_map.get(model_strategy, 'gray'))
                ax.fill(angles, values, alpha=0.15,
                        color=color_map.get(model_strategy, 'gray'))

            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax.set_ylim(0, 1)
            ax.set_title('Model-Strategy Performance Comparison (Normalized)', size=16, y=1.1)
            ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))

            plt.tight_layout()
            plt.savefig(viz_dir / 'model_strategy_comparison_radar.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error creating radar chart: {e}")
            self._create_model_strategy_comparison_bars(df, viz_dir)

    def _create_model_strategy_comparison_bars(self, df, viz_dir):
        plt.figure(figsize=(16, 8))

        metrics = ['silhouette_score', 'n_clusters', 'avg_cluster_size']
        if not df['cohesion'].isna().all():
            metrics.extend(['cohesion', 'separation'])

        model_strategies = sorted(df['model_strategy'].unique())
        x = np.arange(len(metrics))
        width = 0.8 / len(model_strategies)

        for i, model_strategy in enumerate(model_strategies):
            model_data = df[df['model_strategy'] == model_strategy]
            values = [model_data[metric].mean() for metric in metrics]

            plt.bar(x + i * width, values, width, label=model_strategy, alpha=0.8)

        plt.xlabel('Metrics')
        plt.ylabel('Average Values')
        plt.title('Model-Strategy Performance Comparison')
        plt.xticks(x + width * len(model_strategies) / 2,
                   [m.replace('_', ' ').title() for m in metrics])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(viz_dir / 'model_strategy_comparison_bars.png', dpi=300, bbox_inches='tight')
        plt.close()
    def _create_detailed_visualizations(self, df, viz_dir):
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        numeric_cols = ['silhouette_score', 'davies_bouldin_score', 'n_clusters',
                        'avg_cluster_size', 'threshold']
        correlation_matrix = df[numeric_cols].corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('Clustering Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(viz_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Performance by CSV file
        if len(df['csv_file'].unique()) > 1:
            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            df.boxplot(column='silhouette_score', by='csv_file', ax=plt.gca())
            plt.title('Silhouette Score by Dataset')
            plt.xticks(rotation=45)

            plt.subplot(1, 2, 2)
            df.boxplot(column='davies_bouldin_score', by='csv_file', ax=plt.gca())
            plt.title('Davies-Bouldin Score by Dataset')
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(viz_dir / 'performance_by_dataset.png', dpi=300, bbox_inches='tight')
            plt.close()

    def generate_report(self):
        logger.info("Generating evaluation report...")

        report = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': self.session_data.get('start_time'),
                'end_time': datetime.now().isoformat(),
                'total_configurations': len(self.session_data.get('apps_processed', {}))
            },
            'summary': {},
            'detailed_results': self.session_data.get('evaluation_metrics', {}),
            'recommendations': []
        }

        # Generate summary
        all_configs = self.session_data.get('apps_processed', {})
        if all_configs:
            successful_configs = [c for c in all_configs.values()
                                  if c.get('successful_selections', 0) > 0]

            report['summary'] = {
                'total_configurations_tested': len(all_configs),
                'successful_configurations': len(successful_configs),
                'success_rate': len(successful_configs) / len(all_configs) if all_configs else 0,
                'total_apps_processed': sum(c.get('apps_found', 0) for c in all_configs.values()),
                'avg_processing_time': np.mean([c.get('processing_time', 0) for c in all_configs.values()])
            }

        # Generate recommendations
        evaluation_metrics = self.session_data.get('evaluation_metrics', {})
        internal_metrics = evaluation_metrics.get('internal_metrics', {})

        if internal_metrics:
            sil_stats = internal_metrics.get('summary_statistics', {}).get('silhouette', {})
            db_stats = internal_metrics.get('summary_statistics', {}).get('davies_bouldin', {})

            recommendations = []

            if sil_stats.get('mean', 0) < 0.3:
                recommendations.append({
                    'type': 'quality_warning',
                    'message': 'Low average silhouette score detected. Consider adjusting clustering parameters.',
                    'priority': 'high'
                })

            if db_stats.get('mean', float('inf')) > 2.0:
                recommendations.append({
                    'type': 'quality_warning',
                    'message': 'High Davies-Bouldin score indicates poor cluster separation.',
                    'priority': 'medium'
                })

            report['recommendations'] = recommendations

        # Save report
        report_file = self.results_dir / f"evaluation_report_{self.session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {report_file}")
        return report
