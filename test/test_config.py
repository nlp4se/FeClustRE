import argparse
import sys
from systematic_testing import SystematicTester


def medium_test():
    print("Running medium test...")

    tester = SystematicTester()

    csv_files = [
        "../data/input/endpoint_1_process_reviews/ai_assistants/cleaned_data/Claude_by_Anthropic.csv",
    ]

    configurations = tester.run_full_pipeline(
        csv_files=csv_files,
        model_types=['transfeatex'],
        sample_sizes=[100],
        selection_strategies=['silhouette']
    )

    evaluation_results = tester.evaluate_clustering_quality()
    tester.generate_visualizations()
    report = tester.generate_report()

    print(f"Medium test complete! Session: {tester.session_id}")
    return tester.session_id


def full_test():
    print("Running full comprehensive test...")

    tester = SystematicTester()

    csv_files = [
        "../data/input/endpoint_1_process_reviews/ai_assistants/cleaned_data/ChatGPT.csv",
        "../data/input/endpoint_1_process_reviews/ai_assistants/cleaned_data/Claude_by_Anthropic.csv",
        "../data/input/endpoint_1_process_reviews/ai_assistants/cleaned_data/DeepSeek_-_AI_Assistant.csv",
        "../data/input/endpoint_1_process_reviews/ai_assistants/cleaned_data/Google_Gemini.csv",
        "../data/input/endpoint_1_process_reviews/ai_assistants/cleaned_data/Le_Chat_by_Mistral.csv",
        "../data/input/endpoint_1_process_reviews/ai_assistants/cleaned_data/Microsoft_Copilot.csv",
        "../data/input/endpoint_1_process_reviews/ai_assistants/cleaned_data/Perplexity_-_Ask_Anything.csv"
    ]

    configurations = tester.run_full_pipeline(
        csv_files=csv_files,
        model_types=['tfrex', 'transfeatex'],
        sample_sizes=[1000, 2000, 5000, 99999999],
        selection_strategies=['balanced', 'silhouette', 'conservative']
    )

    evaluation_results = tester.evaluate_clustering_quality()
    tester.generate_visualizations()
    report = tester.generate_report()

    print(f"Full test complete! Session: {tester.session_id}")
    return tester.session_id


def custom_test(csv_files, model_types, sample_sizes, strategies):
    print("Running custom test...")

    tester = SystematicTester()

    configurations = tester.run_full_pipeline(
        csv_files=csv_files,
        model_types=model_types,
        sample_sizes=sample_sizes,
        selection_strategies=strategies
    )

    evaluation_results = tester.evaluate_clustering_quality()
    tester.generate_visualizations()
    report = tester.generate_report()

    print(f"Custom test complete! Session: {tester.session_id}")
    return tester.session_id


def ablation_study():
    print("Running ablation study...")

    tester = SystematicTester()

    test_cases = [
        {
            'name': 'tfrex_vs_transfeatex',
            'csv_files': ["ChatGPT.csv", "Claude_by_Anthropic.csv"],
            'model_types': ['tfrex', 'transfeatex'],
            'sample_sizes': [1000],
            'strategies': ['balanced']
        },
        {
            'name': 'sample_size_impact',
            'csv_files': ["ChatGPT.csv"],
            'model_types': ['tfrex'],
            'sample_sizes': [500, 1000, 2000, 5000],
            'strategies': ['balanced']
        },
        {
            'name': 'selection_strategy_comparison',
            'csv_files': ["Claude_by_Anthropic.csv"],
            'model_types': ['tfrex'],
            'sample_sizes': [1000],
            'strategies': ['balanced', 'silhouette', 'conservative']
        }
    ]

    all_results = {}

    for test_case in test_cases:
        print(f"\nRunning test case: {test_case['name']}")

        case_tester = SystematicTester()
        case_tester.session_id = f"ablation_{test_case['name']}_{case_tester.session_id}"

        configurations = case_tester.run_full_pipeline(
            csv_files=test_case['csv_files'],
            model_types=test_case['model_types'],
            sample_sizes=test_case['sample_sizes'],
            selection_strategies=test_case['strategies']
        )

        evaluation_results = case_tester.evaluate_clustering_quality()
        case_tester.generate_visualizations()
        report = case_tester.generate_report()

        all_results[test_case['name']] = {
            'session_id': case_tester.session_id,
            'configurations': configurations,
            'evaluation': evaluation_results,
            'report': report
        }

        print(f"Test case '{test_case['name']}' complete!")

    # Generate comparative analysis
    generate_comparative_analysis(all_results)

    return all_results


def generate_comparative_analysis(ablation_results):
    import pandas as pd
    import matplotlib.pyplot as plt

    print("Generating comparative analysis...")

    comparison_data = []

    for test_name, results in ablation_results.items():
        evaluation = results.get('evaluation', {})
        internal_metrics = evaluation.get('internal_metrics', {})
        summary_stats = internal_metrics.get('summary_statistics', {})

        if summary_stats:
            comparison_data.append({
                'test_case': test_name,
                'avg_silhouette': summary_stats.get('silhouette', {}).get('mean', 0),
                'avg_davies_bouldin': summary_stats.get('davies_bouldin', {}).get('mean', 0),
                'silhouette_std': summary_stats.get('silhouette', {}).get('std', 0),
                'davies_bouldin_std': summary_stats.get('davies_bouldin', {}).get('std', 0)
            })

    if comparison_data:
        df = pd.DataFrame(comparison_data)

        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Silhouette comparison
        ax1.bar(df['test_case'], df['avg_silhouette'], yerr=df['silhouette_std'], capsize=5)
        ax1.set_title('Average Silhouette Score by Test Case')
        ax1.set_ylabel('Silhouette Score')
        ax1.tick_params(axis='x', rotation=45)

        # Davies-Bouldin comparison
        ax2.bar(df['test_case'], df['avg_davies_bouldin'], yerr=df['davies_bouldin_std'], capsize=5)
        ax2.set_title('Average Davies-Bouldin Score by Test Case')
        ax2.set_ylabel('Davies-Bouldin Score')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'evaluation_results/ablation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save comparison data
        df.to_csv('evaluation_results/ablation_comparison.csv', index=False)

        print("Comparative analysis saved!")


def main():
    parser = argparse.ArgumentParser(description='Systematic Testing Framework for Feature Clustering')
    parser.add_argument('mode',
                        choices=['medium', 'full', 'custom', 'ablation'],
                        help='Test mode to run')
    parser.add_argument('--session-file', type=str, help='Session file to resume (for resume mode)')
    parser.add_argument('--csv-files', nargs='+', help='CSV files to process (for custom mode)')
    parser.add_argument('--models', nargs='+', default=['tfrex'], help='Models to test')
    parser.add_argument('--samples', nargs='+', type=int, default=[1000], help='Sample sizes to test')
    parser.add_argument('--strategies', nargs='+', default=['balanced'], help='Selection strategies')

    args = parser.parse_args()

    try:
        if args.mode == 'medium':
            session_id = medium_test()
        elif args.mode == 'full':
            session_id = full_test()
        elif args.mode == 'custom':
            if not args.csv_files:
                print("Error: --csv-files required for custom mode")
                sys.exit(1)
            session_id = custom_test(args.csv_files, args.models, args.samples, args.strategies)
        elif args.mode == 'ablation':
            results = ablation_study()
            print("Ablation study complete!")
            return

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
