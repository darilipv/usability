#!/usr/bin/env python
"""
Evaluator Module
Main evaluation class that orchestrates data loading, analysis, and stability calculation.
"""

from typing import Dict, List, Any, Optional
from data_storage import DataStorage, JSONDataStorage
from stability_calculator import StabilityCalculator, JaccardSimilarity, LengthSimilarity
import json


class TestResultAggregator:
    """
    Aggregates test results by base prompt and agent.
    Organizes data for stability analysis.
    """
    
    def __init__(self):
        self._data = {}  # {base_prompt: {agent_name: [responses]}}
    
    def add_result(self, base_prompt: str, agent_name: str, response: str, 
                   style_combination: str = None, sentiment: Dict = None):
        """
        Add a test result to the aggregator.
        
        Args:
            base_prompt: The base prompt without style modifications
            agent_name: Name of the agent/model
            response: The response text
            style_combination: The style combination used
            sentiment: Sentiment analysis results
        """
        if base_prompt not in self._data:
            self._data[base_prompt] = {}
        
        if agent_name not in self._data[base_prompt]:
            self._data[base_prompt][agent_name] = []
        
        self._data[base_prompt][agent_name].append({
            'response': response,
            'style_combination': style_combination,
            'sentiment': sentiment
        })
    
    def get_response_sets(self, base_prompt: str) -> Dict[str, List[str]]:
        """
        Get response sets for a specific base prompt.
        
        Args:
            base_prompt: The base prompt to get responses for
            
        Returns:
            Dictionary mapping agent names to lists of response texts
        """
        if base_prompt not in self._data:
            return {}
        
        response_sets = {}
        for agent_name, results in self._data[base_prompt].items():
            response_sets[agent_name] = [r['response'] for r in results]
        
        return response_sets
    
    def get_all_prompts(self) -> List[str]:
        """Get all base prompts that have been aggregated."""
        return list(self._data.keys())
    
    def get_full_data(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Get the complete aggregated data structure."""
        return self._data


class Evaluator:
    """
    Main evaluator class that loads test data and calculates stability metrics.
    Uses Monte-Carlo simulation for robust stability estimation.
    """
    
    def __init__(self, 
                 data_storage: DataStorage = None,
                 stability_calculator: StabilityCalculator = None,
                 monte_carlo_iterations: int = 1000):
        """
        Initialize the evaluator.
        
        Args:
            data_storage: Storage instance for loading data
            stability_calculator: Calculator instance for stability metrics
            monte_carlo_iterations: Number of iterations for Monte-Carlo simulation
        """
        self._data_storage = data_storage or JSONDataStorage()
        self._stability_calculator = stability_calculator or StabilityCalculator()
        self._monte_carlo_iterations = monte_carlo_iterations
        self._aggregator = TestResultAggregator()
    
    def load_and_aggregate_data(self) -> None:
        """
        Load test results from storage and aggregate them by prompt and agent.
        """
        test_results = self._data_storage.load_test_results()
        
        for result in test_results:
            base_prompt = result.get('base_prompt', '')
            agent_name = result.get('agent_name', '')
            response = result.get('response', '')
            style_combination = result.get('style_combination')
            sentiment = result.get('sentiment')
            
            if base_prompt and agent_name and response:
                self._aggregator.add_result(
                    base_prompt, agent_name, response, 
                    style_combination, sentiment
                )
    
    def evaluate_stability(self, base_prompt: str = None) -> Dict[str, Any]:
        """
        Evaluate stability for a specific prompt or all prompts.
        
        Args:
            base_prompt: Specific prompt to evaluate (None = evaluate all)
            
        Returns:
            Dictionary containing stability metrics
        """
        if base_prompt:
            prompts = [base_prompt]
        else:
            prompts = self._aggregator.get_all_prompts()
        
        results = {}
        
        for prompt in prompts:
            response_sets = self._aggregator.get_response_sets(prompt)
            
            if not response_sets:
                continue
            
            # Calculate comprehensive stability using Monte-Carlo
            stability_metrics = self._stability_calculator.calculate_comprehensive_stability(
                response_sets, 
                n_iterations=self._monte_carlo_iterations
            )
            
            results[prompt] = {
                'stability_metrics': stability_metrics,
                'num_responses_per_agent': {
                    agent: len(responses) 
                    for agent, responses in response_sets.items()
                }
            }
        
        return results
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_file: Optional file path to save the report
            
        Returns:
            Report text as string
        """
        self.load_and_aggregate_data()
        evaluation_results = self.evaluate_stability()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PROMPT STABILITY EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"Monte-Carlo Iterations: {self._monte_carlo_iterations}")
        report_lines.append(f"Total Prompts Evaluated: {len(evaluation_results)}")
        report_lines.append("")
        
        for prompt, data in evaluation_results.items():
            report_lines.append("-" * 80)
            report_lines.append(f"Base Prompt: {prompt}")
            report_lines.append("-" * 80)
            
            stability_metrics = data['stability_metrics']
            num_responses = data['num_responses_per_agent']
            
            for agent_name, metrics in stability_metrics.items():
                report_lines.append(f"\nAgent: {agent_name}")
                report_lines.append(f"  Number of Responses: {num_responses.get(agent_name, 0)}")
                report_lines.append(f"  Mean Stability: {metrics['mean_stability']:.4f}")
                report_lines.append(f"  Stability Std Dev: {metrics['std_dev']:.4f}")
                report_lines.append(f"  Stability Variance: {metrics['variance']:.4f}")
                report_lines.append(f"  Min Stability: {metrics['min_stability']:.4f}")
                report_lines.append(f"  Max Stability: {metrics['max_stability']:.4f}")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all prompts and agents.
        
        Returns:
            Dictionary with summary statistics
        """
        self.load_and_aggregate_data()
        evaluation_results = self.evaluate_stability()
        
        all_stabilities = []
        agent_stabilities = {}
        
        for prompt, data in evaluation_results.items():
            stability_metrics = data['stability_metrics']
            
            for agent_name, metrics in stability_metrics.items():
                mean_stability = metrics['mean_stability']
                all_stabilities.append(mean_stability)
                
                if agent_name not in agent_stabilities:
                    agent_stabilities[agent_name] = []
                agent_stabilities[agent_name].append(mean_stability)
        
        summary = {
            'overall_mean_stability': sum(all_stabilities) / len(all_stabilities) if all_stabilities else 0.0,
            'overall_min_stability': min(all_stabilities) if all_stabilities else 0.0,
            'overall_max_stability': max(all_stabilities) if all_stabilities else 0.0,
            'agent_averages': {
                agent: sum(stabilities) / len(stabilities) 
                for agent, stabilities in agent_stabilities.items()
            }
        }
        
        return summary

