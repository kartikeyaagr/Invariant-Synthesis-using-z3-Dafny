#!/usr/bin/env python3
"""
Week 10: Disjunctive Invariant Synthesis
Main entry point.

Usage:
  python main.py <dafny_file>           # Synthesize invariants
  python main.py <dafny_file> --analyze # Only analyze paths
  python main.py --test                 # Run tests
  python main.py --benchmark            # Run benchmarks
"""

import sys
import os

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from disjunctive_synthesis import main

if __name__ == "__main__":
    sys.exit(main())
