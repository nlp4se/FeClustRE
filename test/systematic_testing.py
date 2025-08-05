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
        logger.info("Starting clustering quality evaluation...")

        evaluation_results = {
            'internal_metrics': {},
            'external_metrics': {},
            'app_comparisons': {},
            'threshold_analysis': {},
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

        for csv_name, configs in apps_data.items():
            for config in configs:
                app_results = config.get('processing_results', {}).get('results', {})

                for app_name, result in app_results.items():
                    candidates = result.get('clustering_results', {}).get('candidates', [])

                    for i, candidate in enumerate(candidates):
                        summary = candidate.get('summary', {})
                        metrics = summary.get('metrics', {})

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
                            'davies_bouldin_score': metrics.get('davies_bouldin_score', float('inf'))
                        }

                        internal_metrics.append(metric_data)
                        threshold_data.append(metric_data)

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
                    }
                },
                # Convert groupby results to serializable format
                'by_model': self._convert_groupby_to_dict(metrics_df.groupby('model_type').agg({
                    'silhouette_score': ['mean', 'std'],
                    'davies_bouldin_score': ['mean', 'std'],
                    'n_clusters': ['mean', 'std']
                })),
                'correlation_matrix': metrics_df[['silhouette_score', 'davies_bouldin_score',
                                                  'n_clusters', 'avg_cluster_size', 'threshold']].corr().to_dict()
            }

        external_metrics = {
            'cluster_size_distribution': {},
            'app_diversity': {},
            'threshold_evolution': {}
        }

        for csv_name, configs in apps_data.items():
            sizes = []
            for config in configs:
                best_selections = config.get('best_selections', {})
                for app_name, selection in best_selections.items():
                    clustering = selection['candidate']['clustering']
                    clusters = clustering.get('clusters', {})
                    sizes.extend([len(features) for features in clusters.values()])

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

        evaluation_results['external_metrics'] = external_metrics

        # Store evaluation results
        self.session_data['evaluation_metrics'] = evaluation_results
        self.save_session()

        return evaluation_results

    def _convert_groupby_to_dict(self, groupby_result):
        result_dict = {}

        for index, row in groupby_result.iterrows():
            # Convert index to string if it's not already
            key = str(index) if not isinstance(index, str) else index

            # Convert the row to a dictionary with string keys
            row_dict = {}
            for col_tuple, value in row.items():
                # Handle multi-level column names (tuples)
                if isinstance(col_tuple, tuple):
                    col_key = '_'.join(str(x) for x in col_tuple)
                else:
                    col_key = str(col_tuple)

                # Convert numpy types to Python types
                if hasattr(value, 'item'):
                    row_dict[col_key] = float(value.item()) if not pd.isna(value) else None
                else:
                    row_dict[col_key] = float(value) if not pd.isna(value) else None

            result_dict[key] = row_dict

        return result_dict

    def generate_visualizations(self):
        logger.info("Generating evaluation visualizations...")

        viz_dir = self.results_dir / f"visualizations_{self.session_id}"
        viz_dir.mkdir(exist_ok=True)

        # Collect metrics data
        all_configs = self.session_data.get('apps_processed', {})
        internal_metrics = []

        for config_key, config in all_configs.items():
            app_results = config.get('processing_results', {}).get('results', {})

            for app_name, result in app_results.items():
                candidates = result.get('clustering_results', {}).get('candidates', [])

                for i, candidate in enumerate(candidates):
                    summary = candidate.get('summary', {})
                    metrics = summary.get('metrics', {})

                    internal_metrics.append({
                        'csv_file': Path(config['csv_file']).stem,
                        'app_name': app_name,
                        'model_type': config['model_type'],
                        'threshold': summary.get('threshold', 0),
                        'silhouette_score': metrics.get('silhouette_score', 0),
                        'davies_bouldin_score': metrics.get('davies_bouldin_score', float('inf')),
                        'n_clusters': summary.get('n_clusters', 0),
                        'avg_cluster_size': summary.get('avg_cluster_size', 0)
                    })

        df = pd.DataFrame(internal_metrics)

        if df.empty:
            logger.warning("No data available for visualization")
            return

        # Filter out infinite values
        df = df[df['davies_bouldin_score'] != float('inf')]

        plt.style.use('seaborn-v0_8')

        # 1. Silhouette vs Davies-Bouldin scatter plot
        plt.figure(figsize=(12, 8))

        # Create subplot for scatter plot by model type
        plt.subplot(2, 2, 1)
        for model in df['model_type'].unique():
            model_data = df[df['model_type'] == model]
            plt.scatter(model_data['silhouette_score'], model_data['davies_bouldin_score'],
                        label=model, alpha=0.7)

        plt.xlabel('Silhouette Score (higher is better)')
        plt.ylabel('Davies-Bouldin Score (lower is better)')
        plt.title('Clustering Quality: Silhouette vs Davies-Bouldin')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Threshold evolution
        plt.subplot(2, 2, 2)
        threshold_groups = df.groupby(['threshold', 'model_type']).agg({
            'silhouette_score': 'mean',
            'davies_bouldin_score': 'mean'
        }).reset_index()

        for model in threshold_groups['model_type'].unique():
            model_data = threshold_groups[threshold_groups['model_type'] == model]
            plt.plot(model_data['threshold'], model_data['silhouette_score'],
                     marker='o', label=f'{model} (Silhouette)')

        plt.xlabel('Height Threshold')
        plt.ylabel('Average Silhouette Score')
        plt.title('Threshold Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Cluster size distribution
        plt.subplot(2, 2, 3)
        df.boxplot(column='avg_cluster_size', by='model_type', ax=plt.gca())
        plt.title('Cluster Size Distribution by Model')
        plt.xlabel('Model Type')
        plt.ylabel('Average Cluster Size')

        # 4. Number of clusters distribution
        plt.subplot(2, 2, 4)
        df.boxplot(column='n_clusters', by='model_type', ax=plt.gca())
        plt.title('Number of Clusters by Model')
        plt.xlabel('Model Type')
        plt.ylabel('Number of Clusters')

        plt.tight_layout()
        plt.savefig(viz_dir / 'clustering_quality_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Additional detailed visualizations
        self._create_detailed_visualizations(df, viz_dir)

        logger.info(f"Visualizations saved to {viz_dir}")

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


if __name__ == "__main__":
    tester = SystematicTester()

    csv_files = [
        "ChatGPT.csv",
        "Claude_by_Anthropic.csv",
        "DeepSeek_-_AI_Assistant.csv"
    ]

    configurations = tester.run_full_pipeline(
        csv_files=csv_files,
        model_types=['tfrex'],
        sample_sizes=[500, 1000],
        selection_strategies=['balanced', 'silhouette']
    )

    evaluation_results = tester.evaluate_clustering_quality()

    tester.generate_visualizations()

    report = tester.generate_report()

    print(f"Evaluation complete! Session ID: {tester.session_id}")
    print(f"Results saved in: {tester.results_dir}")
