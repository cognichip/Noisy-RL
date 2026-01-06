#!/usr/bin/env python3
"""
reward_sandbox_with_noise.py – Python coding scorer with synthetic noise

🎯 NOISE-ENABLED VERSION: Extends reward_sandbox_simple.py with synthetic noise

This version adds controlled noise to rewards to simulate:
- False Positives: Solutions that should fail but get a passing reward
- False Negatives: Solutions that should pass but get a failing reward

Key Features:
- All features from reward_sandbox_simple.py
- Configurable false positive and false negative rates
- Detailed noise application statistics
- Support for both binary and percentage rewards
"""

import numpy as np
import os
import sys
from typing import List, Any, Dict, Optional

# Import the original reward calculation functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reward_sandbox_simple import (
    compute_score_batch as original_compute_score_batch,
    _sandbox_rewards,
    _sandbox_percentage_rewards,
    extract_code
)

# ────────────────────────────────────────────────────────────
# Noise Application Functions
# ────────────────────────────────────────────────────────────

def apply_noise(true_reward: float, false_positive_rate: float, false_negative_rate: float) -> float:
    """
    Apply synthetic noise to a reward value.
    
    Args:
        true_reward: The original reward value (0.0 or 1.0)
        false_positive_rate: Probability to convert a 0 to 1 (false positive)
        false_negative_rate: Probability to convert a 1 to 0 (false negative)
        
    Returns:
        float: The potentially modified reward value
    """
    if true_reward < 0.5:  # Truly negative
        # Flip to positive with false positive rate
        return 1.0 if np.random.random() < false_positive_rate else 0.0
    else:  # Truly positive
        # Flip to negative with false negative rate
        return 0.0 if np.random.random() < false_negative_rate else 1.0

def _sandbox_rewards_with_noise(
    solution_strs: List[str], 
    verification_infos: List[Dict], 
    *, 
    false_positive_rate: float = 0.0, 
    false_negative_rate: float = 0.0,
    num_workers: int = 64, 
    return_percentage: bool = False
) -> List[float]:
    """
    Wrapper around _sandbox_rewards to add synthetic noise.
    
    Args:
        solution_strs: List of solution strings to evaluate
        verification_infos: List of verification info dicts containing test cases
        false_positive_rate: Probability to convert a failing reward to passing
        false_negative_rate: Probability to convert a passing reward to failing
        num_workers: Number of parallel workers for sandbox execution
        return_percentage: If True, return raw percentage scores (0.0-1.0)
                          If False, return binary (0.0 or 1.0)
    
    Returns:
        List of scores with synthetic noise applied
    """
    # Get true rewards first
    true_rewards = _sandbox_rewards(
        solution_strs, 
        verification_infos, 
        num_workers=num_workers, 
        return_percentage=return_percentage
    )
    
    # Apply noise if needed
    if false_positive_rate > 0 or false_negative_rate > 0:
        noisy_rewards = []
        flipped_count = 0
        fp_count = 0
        fn_count = 0
        
        for reward in true_rewards:
            if return_percentage:
                # For percentage rewards, we'll binarize to apply noise
                binary_reward = 1.0 if reward >= 0.99 else 0.0
                noisy_binary = apply_noise(
                    binary_reward, 
                    false_positive_rate, 
                    false_negative_rate
                )
                
                # Track flips
                original_binary = 1.0 if reward >= 0.99 else 0.0
                if original_binary != noisy_binary:
                    flipped_count += 1
                    if original_binary < 0.5 and noisy_binary >= 0.5:
                        fp_count += 1
                    elif original_binary >= 0.5 and noisy_binary < 0.5:
                        fn_count += 1
                
                # For percentage rewards, keep the original percentage for true positives
                # or noise-flipped positives, otherwise 0
                noisy_reward = reward if noisy_binary > 0.5 else 0.0
            else:
                # For binary rewards, simply flip based on probabilities
                noisy_reward = apply_noise(
                    reward, 
                    false_positive_rate, 
                    false_negative_rate
                )
                
                # Track flips
                if abs(reward - noisy_reward) > 0.01:
                    flipped_count += 1
                    if reward < 0.5 and noisy_reward >= 0.5:
                        fp_count += 1
                    elif reward >= 0.5 and noisy_reward < 0.5:
                        fn_count += 1
                    
            noisy_rewards.append(noisy_reward)
        
        # Log noise application stats
        print(f"[Noise] 🔄 Applied noise: FP rate={false_positive_rate:.2f}, FN rate={false_negative_rate:.2f}")
        print(f"[Noise] 📊 Results: {flipped_count}/{len(true_rewards)} rewards flipped "
              f"({fp_count} false positives, {fn_count} false negatives)")
        
        return noisy_rewards
    else:
        # No noise to apply
        return true_rewards

def _sandbox_percentage_rewards_with_noise(
    solution_strs: List[str], 
    verification_infos: List[Dict], 
    *,
    false_positive_rate: float = 0.0, 
    false_negative_rate: float = 0.0,
    num_workers: int = 64
) -> List[float]:
    """
    Convenience function for percentage-based sandbox evaluation with noise.
    """
    return _sandbox_rewards_with_noise(
        solution_strs, 
        verification_infos, 
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        num_workers=num_workers, 
        return_percentage=True
    )

# ────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────

def compute_score_batch(
    data_sources: Any,
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[dict],
    *,
    # Noise parameters
    false_positive_rate: float = 0.0,
    false_negative_rate: float = 0.0,
    # Original parameters
    sim_tool_test: str = "sandbox",
    sim_tool_train: str = "sandbox",
    timeout: int = 10,
    batch_size: int = 1024,
    max_batch_retries: int = 0,
    retry_delay: float = 5,
    log_dir: str = "./store_data_py",
    num_parallel_calls: int = 64,
    **kwargs,
) -> List[float]:
    """
    Main entry point for batch scoring of Python solutions with sandbox evaluation and synthetic noise.
    
    Args:
        false_positive_rate: Probability to convert a failing reward to passing (0.0 to 1.0)
        false_negative_rate: Probability to convert a passing reward to failing (0.0 to 1.0)
        sim_tool_test: Evaluation method for test split ("sandbox" or "sandbox_percentage")
        sim_tool_train: Evaluation method for train split ("sandbox" or "sandbox_percentage")
        
    Returns:
        List[float]: Scores for each solution with noise applied ONLY to train split
    """
    # If no noise requested, use the original function for efficiency
    if false_positive_rate <= 0 and false_negative_rate <= 0:
        print(f"[Batch Scorer] ℹ️ No noise requested, using original scoring method")
        return original_compute_score_batch(
            data_sources=data_sources,
            solution_strs=solution_strs,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
            sim_tool_test=sim_tool_test,
            sim_tool_train=sim_tool_train,
            timeout=timeout,
            batch_size=batch_size,
            max_batch_retries=max_batch_retries,
            retry_delay=retry_delay,
            log_dir=log_dir,
            num_parallel_calls=num_parallel_calls,
            **kwargs
        )
    
    # With noise enabled, we need to handle the scoring ourselves
    n_items = len(solution_strs)
    print(f"[Batch Scorer] 🚀 Processing {n_items} Python solutions with noisy evaluation (on train split only)")
    print(f"[Batch Scorer] 🔊 Noise settings: FP rate={false_positive_rate:.2f}, FN rate={false_negative_rate:.2f}")
    
    # Group items by split type
    train_indices = []
    test_indices = []
    split_counts = {"train": 0, "test": 0}
    
    for i in range(n_items):
        extra_info = extra_infos[i] if i < len(extra_infos) else {}
        split = extra_info.get("split", "train")
        
        if split == "test":
            test_indices.append(i)
            split_counts["test"] += 1
        else:
            train_indices.append(i)
            split_counts["train"] += 1
    
    print(f"[Batch Scorer] 📊 Split distribution: {split_counts}")
    print(f"[Batch Scorer] 🔊 Noise will ONLY be applied to train split ({split_counts.get('train', 0)} items)")
    
    # Initialize results
    results = [0.0] * n_items
    
    # Process test indices WITHOUT noise (using original function)
    if test_indices:
        print(f"[Batch Scorer] 🔧 Processing {len(test_indices)} test items WITHOUT noise")
        
        # Extract data for test items
        test_solutions = [solution_strs[i] for i in test_indices]
        test_ground_truths = [ground_truths[i] if i < len(ground_truths) else None for i in test_indices]
        test_extra_infos = [extra_infos[i] if i < len(extra_infos) else {} for i in test_indices]
        
        # Use original function for test items
        test_results = original_compute_score_batch(
            data_sources=data_sources,
            solution_strs=test_solutions,
            ground_truths=test_ground_truths,
            extra_infos=test_extra_infos,
            sim_tool_test=sim_tool_test,
            sim_tool_train=sim_tool_train,
            timeout=timeout,
            batch_size=batch_size,
            max_batch_retries=max_batch_retries,
            retry_delay=retry_delay,
            log_dir=log_dir,
            num_parallel_calls=num_parallel_calls,
            **kwargs
        )
        
        # Store test results
        for idx, result in zip(test_indices, test_results):
            results[idx] = result
    
    # Process train indices WITH noise
    if train_indices:
        print(f"[Batch Scorer] 🔧 Processing {len(train_indices)} train items WITH noise")
        
        # Extract data for train items
        train_solutions = [solution_strs[i] for i in train_indices]
        train_verification_infos = [extra_infos[i].get("verification_info", {}) if i < len(extra_infos) else {} for i in train_indices]
        
        # Determine evaluation method
        method = sim_tool_train
        
        # Evaluate based on method
        if method == "sandbox":
            # Binary sandbox evaluation (0.0 or 1.0) with noise
            print(f"[Batch Scorer] 🔴 Using binary sandbox evaluation with noise")
            train_results = _sandbox_rewards_with_noise(
                train_solutions, 
                train_verification_infos, 
                false_positive_rate=false_positive_rate,
                false_negative_rate=false_negative_rate,
                num_workers=num_parallel_calls, 
                return_percentage=False
            )
            
        elif method == "sandbox_percentage":
            # Percentage-based sandbox evaluation (0.0 to 1.0) with noise
            print(f"[Batch Scorer] 📊 Using percentage-based sandbox evaluation with noise")
            train_results = _sandbox_percentage_rewards_with_noise(
                train_solutions, 
                train_verification_infos, 
                false_positive_rate=false_positive_rate,
                false_negative_rate=false_negative_rate,
                num_workers=num_parallel_calls
            )
            
        else:
            print(f"[Batch Scorer] ⚠️ Unknown method: {method}, defaulting to binary sandbox with noise")
            train_results = _sandbox_rewards_with_noise(
                train_solutions, 
                train_verification_infos, 
                false_positive_rate=false_positive_rate,
                false_negative_rate=false_negative_rate,
                num_workers=num_parallel_calls, 
                return_percentage=False
            )
        
        # Store train results
        for idx, result in zip(train_indices, train_results):
            results[idx] = result
    
    # Summary
    avg_score = np.mean(results) if results else 0
    print(f"[Batch Scorer] ✅ Completed: average score = {avg_score:.3f}")
    
    return results

# Example usage
if __name__ == "__main__":
    print("Noisy Reward Sandbox Example")
    print("----------------------------")
    print("This module extends reward_sandbox_simple.py with synthetic noise")
    print("Example usage:")
    print("  from reward_sandbox_with_noise import compute_score_batch")
    print("  rewards = compute_score_batch(..., false_positive_rate=0.1, false_negative_rate=0.05)")
