# Contributing to Arabic EEG Imagined Speech Classification

Thank you for your interest in contributing to this project! This document provides guidelines for contributions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](../../issues)
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - Your environment (OS, Python version, GPU)

### Suggesting Features

1. Open an issue with the "enhancement" label
2. Describe the feature and its use case
3. Explain why it would be useful

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests: `python test_installation.py`
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to functions and classes
- Keep functions focused and modular

## Project Structure

```
├── train_advanced_model.py    # Main training script
├── predict.py                 # Inference script
├── visualize_results.py       # Visualization utilities
├── eda/                       # Exploratory data analysis
└── data/                      # Dataset (not in repo)
```

## Testing

Before submitting a PR, ensure:

```bash
# Check installation
python test_installation.py

# Test training (dry run)
python train_advanced_model.py --dry_run --epochs 5
```

## Questions?

Open an issue with the "question" label.

---

Thank you for contributing!
