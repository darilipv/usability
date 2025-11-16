#!/usr/bin/env python
"""
Stability Calculator Module
Calculates prompt stability using various metrics and Monte-Carlo simulation.
"""

import random
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
from abc import ABC, abstractmethod


class SimilarityMetric(ABC):
    """
    Abstract base class for similarity metrics.
    Defines interface for comparing responses.
    """
    
    @abstractmethod
    def calculate(self, response1: str, response2: str) -> float:
        """
        Calculate similarity between two responses.
        
        Args:
            response1: First response text
            response2: Second response text
            
        Returns:
            Similarity score between 0 and 1 (1 = identical, 0 = completely different)
        """
        pass


class JaccardSimilarity(SimilarityMetric):
    """
    Jaccard similarity based on word sets.
    """
    
    def calculate(self, response1: str, response2: str) -> float:
        """Calculate Jaccard similarity of word sets."""
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class LengthSimilarity(SimilarityMetric):
    """
    Similarity based on response length.
    """
    
    def calculate(self, response1: str, response2: str) -> float:
        """Calculate similarity based on length ratio."""
        len1 = len(response1)
        len2 = len(response2)
        
        if len1 == 0 and len2 == 0:
            return 1.0
        
        max_len = max(len1, len2)
        min_len = min(len1, len2)
        
        return min_len / max_len if max_len > 0 else 0.0


class StabilityCalculator:
    """
    Calculates prompt stability using Monte-Carlo simulation.
    Stability measures how consistent model responses are across different style variations.
    """
    
    def __init__(self, similarity_metric: SimilarityMetric = None):
        """
        Initialize stability calculator.
        
        Args:
            similarity_metric: Metric to use for comparing responses
        """
        self._similarity_metric = similarity_metric or JaccardSimilarity()
    
    def calculate_pairwise_similarity(self, responses: List[str]) -> List[float]:
        """
        Calculate pairwise similarities between all responses.
        
        Args:
            responses: List of response texts
            
        Returns:
            List of similarity scores for all pairs
        """
        similarities = []
        n = len(responses)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._similarity_metric.calculate(responses[i], responses[j])
                similarities.append(sim)
        
        return similarities
    
    def calculate_stability_score(self, responses: List[str]) -> float:
        """
        Calculate overall stability score from a list of responses.
        Higher score = more stable (consistent responses).
        
        Args:
            responses: List of response texts
            
        Returns:
            Stability score between 0 and 1
        """
        if len(responses) < 2:
            return 1.0  # Single response is perfectly stable
        
        similarities = self.calculate_pairwise_similarity(responses)
        
        if not similarities:
            return 0.0
        
        # Average similarity is the stability score
        return np.mean(similarities)
    
    def monte_carlo_stability(self, 
                             response_sets: Dict[str, List[str]], 
                             n_iterations: int = 1000,
                             sample_size: int = None) -> Dict[str, float]:
        """
        Calculate stability using Monte-Carlo simulation.
        Randomly samples response combinations and calculates stability distribution.
        
        Args:
            response_sets: Dictionary mapping agent names to lists of responses
            n_iterations: Number of Monte-Carlo iterations
            sample_size: Number of responses to sample per iteration (None = use all)
            
        Returns:
            Dictionary mapping agent names to stability scores
        """
        results = {}
        
        for agent_name, responses in response_sets.items():
            if len(responses) < 2:
                results[agent_name] = 1.0
                continue
            
            stability_scores = []
            
            for _ in range(n_iterations):
                # Sample responses (with replacement if sample_size is specified)
                if sample_size and sample_size < len(responses):
                    sampled = random.sample(responses, min(sample_size, len(responses)))
                else:
                    sampled = responses
                
                # Calculate stability for this sample
                stability = self.calculate_stability_score(sampled)
                stability_scores.append(stability)
            
            # Mean stability across all iterations
            results[agent_name] = np.mean(stability_scores)
        
        return results
    
    def calculate_stability_variance(self, 
                                    response_sets: Dict[str, List[str]], 
                                    n_iterations: int = 1000) -> Dict[str, float]:
        """
        Calculate variance in stability scores using Monte-Carlo simulation.
        Higher variance indicates less consistent stability.
        
        Args:
            response_sets: Dictionary mapping agent names to lists of responses
            n_iterations: Number of Monte-Carlo iterations
            
        Returns:
            Dictionary mapping agent names to stability variance
        """
        results = {}
        
        for agent_name, responses in response_sets.items():
            if len(responses) < 2:
                results[agent_name] = 0.0
                continue
            
            stability_scores = []
            
            for _ in range(n_iterations):
                # Sample a subset of responses
                sample_size = max(2, len(responses) // 2)
                sampled = random.sample(responses, min(sample_size, len(responses)))
                
                stability = self.calculate_stability_score(sampled)
                stability_scores.append(stability)
            
            results[agent_name] = np.var(stability_scores)
        
        return results
    
    def calculate_comprehensive_stability(self, 
                                        response_sets: Dict[str, List[str]], 
                                        n_iterations: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive stability metrics including mean and variance.
        
        Args:
            response_sets: Dictionary mapping agent names to lists of responses
            n_iterations: Number of Monte-Carlo iterations
            
        Returns:
            Dictionary mapping agent names to metric dictionaries
        """
        results = {}
        
        for agent_name, responses in response_sets.items():
            if len(responses) < 2:
                results[agent_name] = {
                    'mean_stability': 1.0,
                    'variance': 0.0,
                    'std_dev': 0.0
                }
                continue
            
            stability_scores = []
            
            for _ in range(n_iterations):
                sample_size = max(2, len(responses) // 2)
                sampled = random.sample(responses, min(sample_size, len(responses)))
                stability = self.calculate_stability_score(sampled)
                stability_scores.append(stability)
            
            mean_stability = np.mean(stability_scores)
            variance = np.var(stability_scores)
            std_dev = np.std(stability_scores)
            
            results[agent_name] = {
                'mean_stability': mean_stability,
                'variance': variance,
                'std_dev': std_dev,
                'min_stability': np.min(stability_scores),
                'max_stability': np.max(stability_scores)
            }
        
        return results

