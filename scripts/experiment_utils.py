#!/usr/bin/env python3
"""
Shared utilities for experiment runners.

This module contains common functions used by:
- wrappers/run_full_benchmark.py
- experiments/run_all_experiments.py

Functions include:
- Constraint satisfaction calculation
- Timing parsing from subprocess output
- Safe filename generation
- Time formatting utilities
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd


# =============================================================================
# Constraint Satisfaction Rate (CSR) Calculation
# =============================================================================

def calculate_csr(
    df: pd.DataFrame,
    columns: List[str],
    values: List[Any]
) -> Tuple[float, int, int]:
    """
    Calculate Constraint Satisfaction Rate for a DataFrame.
    
    Args:
        df: DataFrame to evaluate
        columns: List of column names for constraints
        values: List of expected values for each column
    
    Returns:
        Tuple of (csr_percentage, satisfied_count, total_count)
    """
    
    if df.empty or not columns or not values:
        return 0.0, 0, len(df)
        
    # Check for missing columns
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Columns {missing_cols} not found in dataframe for CSR calculation. Returning 0.0.")
        return 0.0, 0, len(df)
    
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col, val in zip(columns, values):
        if df[col].dtype == 'object':
            # String comparison with whitespace normalization
            mask &= (df[col].astype(str).str.strip() == str(val).strip())
        else:
            # Numeric comparison
            try:
                num_val = float(val)
                mask &= (df[col] == num_val)
            except (ValueError, TypeError):
                mask &= (df[col].astype(str).str.strip() == str(val).strip())
    
    satisfied = int(mask.sum())
    total = len(df)
    csr = (satisfied / total) * 100 if total > 0 else 0.0
    
    return round(csr, 2), satisfied, total


def calculate_csr_from_constraint_strings(
    df: pd.DataFrame,
    constraints: List[str]
) -> Tuple[float, int, int]:
    """
    Calculate CSR from constraint strings in 'col= value' format.
    
    Args:
        df: DataFrame to evaluate
        constraints: List of constraint strings like "education= Bachelors"
    
    Returns:
        Tuple of (csr_percentage, satisfied_count, total_count)
    """
    columns = []
    values = []
    
    for c in constraints:
        if '=' not in c:
            continue
        col, val = c.split('=', 1)
        columns.append(col.strip())
        values.append(val.strip())
    
    return calculate_csr(df, columns, values)


# =============================================================================
# Timing Parsing
# =============================================================================

def parse_timing_from_output(output_text: str, method: str = None) -> Dict[str, Optional[float]]:
    """
    Parse timing information from subprocess output.
    
    Args:
        output_text: Combined stdout/stderr from subprocess
        method: Method name (optional, for method-specific patterns)
    
    Returns:
        Dict with training_time, controller_time, sampling_time (in seconds)
    """
    timing = {
        'training_time': None,
        'controller_time': None,
        'sampling_time': None
    }
    
    # Training patterns
    training_patterns = [
        r'Training.*?(\d+\.?\d*)\s*(?:s|seconds)',
        r'train.*?time[:\s]+(\d+\.?\d*)\s*(?:s|sec)',
        r'Diffuser training.*?(\d+\.?\d*)',
        r'Base model.*?(\d+\.?\d*)\s*s',
    ]
    
    # Controller patterns (RelDDPM specific)
    controller_patterns = [
        r'Controller.*?training.*?(\d+\.?\d*)\s*(?:s|seconds)',
        r'controller.*?(\d+\.?\d*)\s*(?:s|sec)',
        r'Training controller.*?(\d+\.?\d*)',
    ]
    
    # Sampling patterns
    sampling_patterns = [
        r'Sampling.*?(\d+\.?\d*)\s*(?:s|seconds)',
        r'Generation.*?(\d+\.?\d*)\s*(?:s|sec)',
        r'sample.*?time[:\s]+(\d+\.?\d*)',
        r'sampling done.*?(\d+\.?\d*)',
    ]
    
    for pattern in training_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            timing['training_time'] = float(match.group(1))
            break
            
    for pattern in controller_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            timing['controller_time'] = float(match.group(1))
            break
            
    for pattern in sampling_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            timing['sampling_time'] = float(match.group(1))
            break
    
    return timing


# =============================================================================
# Hashing & Filename Utilities
# =============================================================================

def get_constraint_hash(constraints: List[Dict[str, Any]], length: int = 8) -> str:
    """
    Generate a deterministic hash for a list of constraints.
    
    Args:
        constraints: List of dicts, e.g. [{"column": "age", "value": 30}]
        length: Length of the hash string
    
    Returns:
        Short hash string
    """
    import hashlib
    
    # Sort to ensure order independence
    sorted_cons = sorted(constraints, key=lambda x: str(x.get("column")) + str(x.get("value")))
    
    # Create canonical string
    canonical = []
    for c in sorted_cons:
        col = c.get("column", "")
        # Normalize strings to avoid subtle differences
        val = str(c.get("value", "")).strip()
        # Handle float/int mismatch (e.g., "1.0" vs "1")
        if val.endswith(".0"):
            val = val[:-2] 
        canonical.append(f"{col}={val}")
    
    encoded = "|".join(canonical).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:length]

def sanitize_constraint_string(constraints: List[str]) -> str:
    """
    Legacy: Sanitize constraint strings for use in filenames.
    Matches the bash script logic: replace '=' and ' ' with '_'.
    """
    suffix = ""
    for c in constraints:
        clean = re.sub(r'[= ]', '_', c)
        suffix += f"_{clean}"
    return suffix


# =============================================================================
# Time Formatting
# =============================================================================

def format_time(seconds: Optional[float]) -> str:
    """
    Format seconds into human readable string.
    
    Args:
        seconds: Time in seconds (or None)
    
    Returns:
        Formatted string like "1.5s", "2.3m", "1.2h", or "N/A"
    """
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# =============================================================================
# Data Loading Utilities
# =============================================================================

def find_real_data_path(dataset: str, project_root: Path = None) -> Optional[Path]:
    """
    Find the path to real training data for a dataset.
    
    Searches in order:
    1. data/processed/{dataset}/train.csv
    2. sub/TabDiff/data/{dataset}/train.csv
    3. sub/27-11-2025/datasets/{dataset}_train.csv
    
    Args:
        dataset: Dataset name
        project_root: Project root path (defaults to auto-detection)
    
    Returns:
        Path to training data or None if not found
    """
    if project_root is None:
        # Try to find project root
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "README.md").exists() and (parent / "sub").exists():
                project_root = parent
                break
        else:
            project_root = Path.cwd()
    
    search_paths = [
        project_root / "data" / "processed" / dataset / "train.csv",
        project_root / "sub" / "TabDiff" / "data" / dataset / "train.csv",
        project_root / "sub" / "27-11-2025" / "datasets" / f"{dataset}_train.csv",
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def load_synthetic_data(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load synthetic data with error handling.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame or None if loading fails
    """
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def to_native_types(obj):
    """Convert numpy types to native Python types for clean serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, str):
        return obj.strip()
    else:
        return obj
