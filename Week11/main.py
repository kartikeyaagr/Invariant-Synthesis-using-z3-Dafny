#!/usr/bin/env python3
"""
Week 11: Quadratic Invariant Synthesis
Main entry point.

Usage:
  python main.py <dafny_file>           # Synthesize invariants
  python main.py --test                 # Run tests
  python main.py --benchmark            # Run benchmarks
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quadratic_synthesis import main

if __name__ == "__main__":
    sys.exit(main())
