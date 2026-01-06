#!/usr/bin/env python3
"""
reward_sandbox_simple.py – Simplified Python coding scorer using only sandbox evaluation

- Sandbox execution of Python code with test cases
- Binary rewards (0.0 or 1.0) - traditional pass/fail
- Percentage rewards (0.0 to 1.0) - based on test pass rate

"""

import asyncio
import json
import logging
import os
import numpy as np
from typing import Any, Dict, List, Optional
import concurrent.futures

# Suppress E2B logging noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("e2b").setLevel(logging.WARNING)
logging.getLogger("e2b_code_interpreter").setLevel(logging.WARNING)



# ────────────────────────────────────────────────────────────
# Patch
# ────────────────────────────────────────────────────────────
"""
sitecustomize.py – polyfill for google.protobuf.runtime_version

Ensures **any** runtime (protobuf 3.x, 4.x, or 5.x) satisfies E2B’s
‘from google.protobuf import runtime_version’ import.
"""

import importlib, sys, types, enum, google.protobuf as _pb

try:
    _pb.runtime_version  # already present? great.
except AttributeError:
    try:
        _pb.runtime_version = importlib.import_module(
            "google.protobuf.runtime_version"  # protobuf ≥5 but attr missing
        )
    except ModuleNotFoundError:
        # protobuf ≤4 → fabricate a minimal stub
        _rv = types.ModuleType("google.protobuf.runtime_version")

        class _Domain(enum.Enum):
            GOOGLE_INTERNAL = 1
            PUBLIC = 2

        _rv.Domain = _Domain
        _rv.PROTOBUF_RUNTIME_VERSION = getattr(_pb, "__version__", "unknown")

        def ValidateProtobufRuntimeVersion(*_a, **_kw):
            return None  # no-op on older runtimes

        _rv.ValidateProtobufRuntimeVersion = ValidateProtobufRuntimeVersion
        sys.modules["google.protobuf.runtime_version"] = _rv
        _pb.runtime_version = _rv

# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────

# E2B Configuration (for sandbox evaluation)
E2B_API_KEY = "<YOUR_E2B_API_KEY>"
os.environ['E2B_API_KEY'] = E2B_API_KEY

# ────────────────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """Extract code from markdown blocks"""
    lines = text.split('\n')
    in_code_block = False
    code_lines = []
    
    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            code_lines.append(line)
    
    return '\n'.join(code_lines) if code_lines else text

# ────────────────────────────────────────────────────────────
# Sandbox execution
# ────────────────────────────────────────────────────────────

EVAL_TEMPLATE = r"""
import subprocess, json, textwrap, sys, tempfile
code = textwrap.dedent({code_json})
fp = tempfile.NamedTemporaryFile(delete=False, suffix='.py'); fp.write(code.encode()); fp.close()

def run_case(inp: str, out: str, timeout: float = 5.0) -> bool:
    try:
        proc = subprocess.run([sys.executable, fp.name], input=inp, text=True,
                              capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            return False
        return proc.stdout.rstrip('\n').strip() == out.strip()
    except Exception:
        return False

cases = json.loads({cases_json})
passed = sum(run_case(c['input'], c['output']) for c in cases)
passed / len(cases)
"""

async def _sandbox_run(script: str, language: str, sem: asyncio.Semaphore) -> float:
    """Run code in E2B sandbox using AsyncSandbox"""
    from e2b_code_interpreter import AsyncSandbox
    SANDBOX_TIMEOUT, MARGIN = 30, 2
    async with sem:
        try:
            if not os.environ.get('E2B_API_KEY'):
                os.environ['E2B_API_KEY'] = E2B_API_KEY
            
            sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=SANDBOX_TIMEOUT - 1)
            execution = await asyncio.wait_for(sandbox.run_code(script, language=language), timeout=SANDBOX_TIMEOUT + MARGIN)
            raw = execution.text or (execution.logs.stdout[-1] if execution.logs else "")
            result = float(str(raw).strip())
            print(f"[Sandbox]  Execution result: {result}")
            return result
        except Exception as e:
            print(f"[Sandbox]  Sandbox execution failed: {e}")
            return 0.0
        finally:
            try:
                await sandbox.kill()
            except Exception:
                pass

async def _run_scripts_async(scripts, language, workers):
    """Run multiple scripts in parallel"""
    sem = asyncio.Semaphore(workers)
    return await asyncio.gather(*[_sandbox_run(s, language, sem) for s in scripts])

def _sandbox_rewards(solution_strs, verification_infos, *, num_workers=64, return_percentage=False):
    """
    Sandbox-based evaluation using E2B pattern
    
    Args:
        solution_strs: List of solution strings to evaluate
        verification_infos: List of verification info dicts containing test cases
        num_workers: Number of parallel workers for sandbox execution
        return_percentage: If True, return raw percentage scores (0.0-1.0). If False, return binary (0.0 or 1.0)
    
    Returns:
        List of scores - either percentage-based (0.0-1.0) or binary (0.0 or 1.0)
    """
    print(f"[Sandbox] 🚀 Starting sandbox evaluation of {len(solution_strs)} items (return_percentage={return_percentage})")
    
    try:
        from e2b_code_interpreter import AsyncSandbox
        print(f"[Sandbox]  E2B AsyncSandbox import successful")
    except ImportError as e:
        print(f"[Sandbox]  E2B AsyncSandbox import failed: {e}")
        return [0.0] * len(solution_strs)
    
    # Extract clean code and calculate token savings
    codes = [extract_code(s) for s in solution_strs]
    total_original = sum(len(s) for s in solution_strs)
    total_clean = sum(len(c) for c in codes)
    if total_original > 0:
        print(f"[Sandbox] 🔧 Code extraction saved {total_original - total_clean} chars ({(total_original - total_clean)/total_original*100:.1f}%)")
    
    scripts = []
    
    for i, (code, verification_info) in enumerate(zip(codes, verification_infos)):
        if verification_info and "test_cases" in verification_info:
            test_cases = verification_info["test_cases"]
            script = EVAL_TEMPLATE.format(
                code_json=json.dumps(code), 
                cases_json=json.dumps(json.dumps(test_cases))
            )
        else:
            script = code
            print(f"[Sandbox] 📝 Item {i}: No test cases found, using code only")
        scripts.append(script)
    
    language = verification_infos[0].get("language", "python") if verification_infos else "python"
    
    try:
        rewards = asyncio.run(_run_scripts_async(scripts, language, num_workers))
    except Exception as e:
        print(f"[Sandbox]  Async execution failed: {e}")
        rewards = [0.0] * len(codes)
    
    if return_percentage:
        # Return raw percentage scores (0.0 to 1.0)
        results = [max(0.0, min(1.0, float(r))) for r in rewards]  # Clamp to [0,1] range
        avg_percentage = sum(results) / len(results) if results else 0.0
        print(f"[Sandbox]  Percentage scoring completed: average {avg_percentage:.3f} ({avg_percentage*100:.1f}% pass rate)")
        print(f"[Sandbox]  Score distribution: min={min(results):.3f}, max={max(results):.3f}, std={np.std(results):.3f}")
    else:
        # Return binary scores (0.0 or 1.0) - original behavior
        results = [1.0 if r >= 0.99 else 0.0 for r in rewards]
        print(f"[Sandbox]  Binary scoring completed: {sum(results)}/{len(results)} passed ({sum(results)/len(results)*100:.1f}%)")
    
    return results

def _sandbox_percentage_rewards(solution_strs, verification_infos, *, num_workers=64):
    """
    Convenience function for percentage-based sandbox evaluation.
    
    This function returns raw percentage scores (0.0 to 1.0) based on the fraction
    of test cases that pass for each solution. Unlike binary scoring, this provides
    granular rewards that reflect partial correctness.
    
    Example:
        - Solution passes 8/10 test cases → score = 0.8
        - Solution passes 5/10 test cases → score = 0.5
        - Solution passes 0/10 test cases → score = 0.0
        - Solution passes 10/10 test cases → score = 1.0
    
    Args:
        solution_strs: List of solution strings to evaluate
        verification_infos: List of verification info dicts containing test cases
        num_workers: Number of parallel workers for sandbox execution
    
    Returns:
        List[float]: Percentage scores between 0.0 and 1.0
    """
    return _sandbox_rewards(solution_strs, verification_infos, num_workers=num_workers, return_percentage=True)

# ────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────

def compute_score_batch(
    data_sources: Any,
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[dict],
    *,
    # Evaluation modes: "sandbox" (binary) or "sandbox_percentage" (0.0-1.0)
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
    Main entry point for batch scoring of Python solutions with sandbox evaluation.
    
    Args:
        sim_tool_test: Evaluation method for test split ("sandbox" or "sandbox_percentage")
        sim_tool_train: Evaluation method for train split ("sandbox" or "sandbox_percentage")
        
        Methods:
        - "sandbox": Binary scoring (0.0 or 1.0) - pass/fail only
        - "sandbox_percentage": Percentage scoring (0.0 to 1.0) - based on test pass rate
        
    Returns:
        List[float]: Scores for each solution
            - Binary mode: 0.0 (fail) or 1.0 (pass)
            - Percentage mode: 0.0 to 1.0 (fraction of tests passed)
    """
    
    n_items = len(solution_strs)
    print(f"[Batch Scorer] 🚀 Processing {n_items} Python solutions with sandbox evaluation")
    
    # Determine evaluation method based on split
    def _get_method(i):
        extra_info = extra_infos[i] if i < len(extra_infos) else {}
        split = extra_info.get("split", "train")
        
        if split == "test":
            return sim_tool_test
        else:
            return sim_tool_train
    
    # Group items by evaluation method
    method_groups = {}
    split_counts = {}
    for i in range(n_items):
        method = _get_method(i)
        if method not in method_groups:
            method_groups[method] = []
        method_groups[method].append(i)
        
        extra_info = extra_infos[i] if i < len(extra_infos) else {}
        split = extra_info.get("split", "train")
        split_counts[split] = split_counts.get(split, 0) + 1
    
    print(f"[Batch Scorer]  Split distribution: {split_counts}")
    print(f"[Batch Scorer]  Method groups: {[(method, len(indices)) for method, indices in method_groups.items()]}")
    
    # Initialize results
    results = [0.0] * n_items
    
    # Process each method group
    for method, indices in method_groups.items():
        if not indices:
            continue
            
        print(f"[Batch Scorer] 🔧 Processing {len(indices)} items with method: {method}")
        
        # Extract data for this group
        group_solutions = [solution_strs[i] for i in indices]
        group_verification_infos = [extra_infos[i].get("verification_info", {}) if i < len(extra_infos) else {} for i in indices]
        
        # Evaluate based on method
        if method == "sandbox":
            # Binary sandbox evaluation (0.0 or 1.0)
            print(f"[Batch Scorer]  Using binary sandbox evaluation")
            group_results = _sandbox_rewards(group_solutions, group_verification_infos, num_workers=num_parallel_calls, return_percentage=False)
            
        elif method == "sandbox_percentage":
            # Percentage-based sandbox evaluation (0.0 to 1.0 based on test pass rate)
            print(f"[Batch Scorer]  Using percentage-based sandbox evaluation")
            group_results = _sandbox_percentage_rewards(group_solutions, group_verification_infos, num_workers=num_parallel_calls)
            
        else:
            print(f"[Batch Scorer]  Unknown method: {method}, defaulting to binary sandbox")
            group_results = _sandbox_rewards(group_solutions, group_verification_infos, num_workers=num_parallel_calls, return_percentage=False)
        
        # Store results
        for idx, result in zip(indices, group_results):
            results[idx] = result
    
    # Summary
    avg_score = np.mean(results) if results else 0
    print(f"[Batch Scorer]  Completed: average score = {avg_score:.3f}")
    
    return results

