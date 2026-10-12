"""PSNG: unsupervised clustering of polysomnography signal windows."""

import os

__version__ = "1.1.0"

# Set before numpy / sklearn load their threading runtimes.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # loky cannot count physical cores on Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoids the KMeans + MKL memory leak on Windows; data here is small
