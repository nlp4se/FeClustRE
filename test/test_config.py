import argparse
import sys
from systematic_testing import SystematicTester

def mock_test():
    print("Running mock test...")

    tester = SystematicTester()

    csv_files = [
        "../data/input/endpoint_1_process_reviews/ai_assistants/cleaned_data/ChatGPT.csv",
        "../data/input/endpoint_1_process_reviews/ai_assistants/cleaned_data/Claude_by_Anthropic.csv",
    ]

    configurations = tester.run_full_pipeline(
        csv_files=csv_files,
        model_types=['transfeatex', 't-frex'],
        sample_sizes=[100, 200, 500],
        selection_strategies=['balanced', 'silhouette', 'conservative']
    )

    evaluation_results = tester.evaluate_clustering_quality()
    tester.generate_visualizations()
    report = tester.generate_report()

    print(f"Mock test complete -> Session: {tester.session_id}")
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



def main():
    parser = argparse.ArgumentParser(description='Systematic Testing Framework for Feature Clustering')
    parser.add_argument('mode',
                        choices=['mock', 'full'],
                        help='Test mode to run')
    parser.add_argument('--session-file', type=str, help='Session file to resume (for resume mode)')
    parser.add_argument('--csv-files', nargs='+', help='CSV files to process (for custom mode)')
    parser.add_argument('--models', nargs='+', default=['tfrex'], help='Models to test')
    parser.add_argument('--samples', nargs='+', type=int, default=[1000], help='Sample sizes to test')
    parser.add_argument('--strategies', nargs='+', default=['balanced'], help='Selection strategies')

    args = parser.parse_args()

    try:
        if args.mode == 'mock':
            session_id = mock_test()
        elif args.mode == 'full':
            session_id = full_test()

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
