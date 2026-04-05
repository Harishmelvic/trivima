"""Shared pytest configuration and markers."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "critical: tests that must pass before proceeding")
    config.addinivalue_line("markers", "phase1: cell struct tests")
    config.addinivalue_line("markers", "phase2: perception pipeline tests")
    config.addinivalue_line("markers", "phase3: cell grid construction tests")
    config.addinivalue_line("markers", "phase4: shell extension tests")
    config.addinivalue_line("markers", "synthetic: tests that don't need ML models")
    config.addinivalue_line("markers", "no_model: image tests that don't need ML models")
    config.addinivalue_line("markers", "needs_model: tests requiring Depth Pro / SAM on GPU")
    config.addinivalue_line("markers", "needs_training: tests requiring VLM distillation training (7-12 days)")
