import argparse
import sys
from pathlib import Path
from systematic_testing import SystematicTester

_DATA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "input"
    / "endpoint_1_process_reviews"
)

AI_ASSISTANTS_DIR = _DATA_ROOT / "ai_assistants"
MOBILE_APPS_DIR = _DATA_ROOT / "mobile_apps"

_AI_ASSISTANT_FILES = [
    "Claude_by_Anthropic.csv",
    "DeepSeek_-_AI_Assistant.csv",
    "Google_Gemini.csv",
    "Le_Chat_by_Mistral_AI.csv",
    "Microsoft_Copilot.csv",
    "Perplexity_-_Ask_Anything.csv",
]

_MOBILE_APPS_PIPELINE_CSV = MOBILE_APPS_DIR / "mobilerec_reviews_pipeline.csv"


def resolve_csv_files(dataset, custom_files=None):
    """Return the list of CSV paths for the requested dataset.

    Priority: explicit --csv-files > --dataset > default (ai_assistants).
    """
    if custom_files:
        return [str(Path(f).resolve()) for f in custom_files]

    if dataset == "mobile_apps":
        if not _MOBILE_APPS_PIPELINE_CSV.exists():
            print(
                f"ERROR: Pipeline CSV not found: {_MOBILE_APPS_PIPELINE_CSV}\n"
                "Generate it first by running the full MobileRec.ipynb notebook."
            )
            sys.exit(1)
        return [str(_MOBILE_APPS_PIPELINE_CSV)]

    # default: ai_assistants — individual per-app CSV files
    return [str(AI_ASSISTANTS_DIR / f) for f in _AI_ASSISTANT_FILES]


def mock_test(csv_files, dataset='ai_assistants'):
    """Minimal end-to-end smoke test — one config, smallest viable sample.

    Purpose: prove the full pipeline (upload → extract → cluster → label → save)
    runs without errors.  Not intended to produce meaningful quality metrics.
    """
    print("Running mock test (smoke only — 1 config)...")

    tester = SystematicTester()

    # mobile_apps is one big CSV with 100 apps.  500 rows → ~5 reviews per app,
    # the minimum needed for the NER model to produce enough features for clustering.
    # ai_assistants are individual per-app files; 50 rows per file is plenty.
    if dataset == 'mobile_apps':
        files = csv_files          # single aggregated file
        sample_sizes = [500]
    else:
        files = csv_files[:1]      # one app file is enough to smoke-test
        sample_sizes = [50]

    configurations = tester.run_full_pipeline(
        csv_files=files,
        model_types=['t-frex'],
        embedding_types=['allmini'],
        sample_sizes=sample_sizes,
        selection_strategies=['balanced']
    )

    evaluation_results = tester.evaluate_clustering_quality()
    tester.generate_visualizations()
    report = tester.generate_report()

    print(f"Mock test complete -> Session: {tester.session_id}")
    return tester.session_id


def full_test(csv_files):
    print("Running full comprehensive test...")

    tester = SystematicTester()

    configurations = tester.run_full_pipeline(
        csv_files=csv_files,
        model_types=['t-frex', 'transfeatex', 'hybrid'],
        embedding_types=['allmini', 'sentence-t5'],
        sample_sizes=[1000, 2000, 5000, 50000],
        selection_strategies=['balanced', 'silhouette', 'conservative']
    )

    evaluation_results = tester.evaluate_clustering_quality()
    tester.generate_visualizations()
    report = tester.generate_report()

    print(f"Full test complete! Session: {tester.session_id}")
    return tester.session_id


def semantic_experiment(csv_files):
    print("Running semantic experiment...")
    print("Configuration: Hybrid model, AllMini embeddings, maximum sample size, no review limits")

    tester = SystematicTester()

    configurations = tester.run_full_pipeline(
        csv_files=csv_files,
        model_types=['hybrid'],
        embedding_types=['allmini'],
        sample_sizes=[None],  # None means no limit - maximum available reviews
        selection_strategies=['balanced']
    )

    evaluation_results = tester.evaluate_clustering_quality()
    tester.generate_visualizations()
    report = tester.generate_report()

    print(f"Semantic experiment complete! Session: {tester.session_id}")
    return tester.session_id


def main():
    parser = argparse.ArgumentParser(description='Systematic Testing Framework for Feature Clustering')
    parser.add_argument('mode',
                        choices=['mock', 'full', 'semantic'],
                        help='Test mode to run')
    parser.add_argument('--dataset',
                        choices=['ai_assistants', 'mobile_apps'],
                        default='ai_assistants',
                        help='Dataset to use (default: ai_assistants). '
                             'mobile_apps requires mobilerec_reviews_pipeline.csv '
                             'to be generated first via MobileRec.ipynb.')
    parser.add_argument('--csv-files', nargs='+',
                        help='Explicit CSV file paths to process. Overrides --dataset.')
    parser.add_argument('--session-file', type=str, help='Session file to resume')
    parser.add_argument('--models', nargs='+', default=['t-frex'], help='Models to test')
    parser.add_argument('--samples', nargs='+', type=int, default=[1000], help='Sample sizes to test')
    parser.add_argument('--strategies', nargs='+', default=['balanced'], help='Selection strategies')

    args = parser.parse_args()

    csv_files = resolve_csv_files(args.dataset, args.csv_files)
    print(f"Dataset: {args.dataset} ({len(csv_files)} file(s))")

    try:
        if args.mode == 'mock':
            session_id = mock_test(csv_files, dataset=args.dataset)
        elif args.mode == 'full':
            session_id = full_test(csv_files)
        elif args.mode == 'semantic':
            session_id = semantic_experiment(csv_files)

        print(f"\nTest completed successfully!")
        print(f"Session ID: {session_id}")
        print(f"Results available in: evaluation_results/")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running test: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
