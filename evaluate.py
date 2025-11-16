#!/usr/bin/env python
"""
Evaluation Script
Main script for evaluating stored test data and calculating prompt stability.
"""

import argparse
import sys
from evaluator import Evaluator
from data_storage import JSONDataStorage
from stability_calculator import StabilityCalculator, JaccardSimilarity, LengthSimilarity


def main():
    """Main function for running evaluations."""
    parser = argparse.ArgumentParser(
        description='Evaluate test results and calculate prompt stability using Monte-Carlo simulation'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='test_data',
        help='Directory containing test data (default: test_data)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Number of Monte-Carlo iterations (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for the report (default: print to stdout)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        choices=['jaccard', 'length'],
        default='jaccard',
        help='Similarity metric to use (default: jaccard)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Evaluate only a specific prompt (default: evaluate all)'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Print only summary statistics'
    )
    
    args = parser.parse_args()
    
    # Initialize components
    data_storage = JSONDataStorage(storage_dir=args.data_dir)
    
    # Choose similarity metric
    if args.metric == 'jaccard':
        similarity_metric = JaccardSimilarity()
    elif args.metric == 'length':
        similarity_metric = LengthSimilarity()
    else:
        similarity_metric = JaccardSimilarity()
    
    stability_calculator = StabilityCalculator(similarity_metric=similarity_metric)
    
    # Create evaluator
    evaluator = Evaluator(
        data_storage=data_storage,
        stability_calculator=stability_calculator,
        monte_carlo_iterations=args.iterations
    )
    
    # Check if data exists
    test_results = data_storage.load_test_results()
    if not test_results:
        print("No test data found. Please run tests first using uuak.py")
        sys.exit(1)
    
    print(f"Loaded {len(test_results)} test results")
    print(f"Using {args.metric} similarity metric")
    print(f"Running Monte-Carlo simulation with {args.iterations} iterations...")
    print()
    
    # Generate report
    if args.summary_only:
        summary = evaluator.get_summary_statistics()
        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Overall Mean Stability: {summary['overall_mean_stability']:.4f}")
        print(f"Overall Min Stability: {summary['overall_min_stability']:.4f}")
        print(f"Overall Max Stability: {summary['overall_max_stability']:.4f}")
        print("\nAgent Averages:")
        for agent, avg_stability in summary['agent_averages'].items():
            print(f"  {agent}: {avg_stability:.4f}")
    else:
        report = evaluator.generate_report(output_file=args.output)
        if not args.output:
            print(report)
        else:
            print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()

