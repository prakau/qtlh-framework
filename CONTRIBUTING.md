# Contributing to QTL-H Framework

We love your input! We want to make contributing to QTL-H Framework as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Development Process
We use GitHub flow, so all code changes happen through pull requests:

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code is properly formatted
6. Issue that pull request!

## Code Style
- Use black for Python code formatting
- Follow PEP 8 guidelines
- Add type hints to function signatures
- Write docstrings for all public functions and classes

## Testing
We use pytest for our test suite. To run tests:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Documentation
- Use Google-style docstrings
- Update README.md with any needed changes
- Update documentation under /docs if necessary
- Create example notebooks for new features

## Reporting Bugs
We use GitHub issues to track public bugs. Report a bug by opening a new issue.

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Feature Requests
We love feature requests! Please use the issue tracker with the "enhancement" label.

## License
By contributing, you agree that your contributions will be licensed under its MIT License.

## References
This document was based on contributing guidelines from various open source projects.

## Contact
For major changes, please open an issue first to discuss what you would like to change.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/qtlh-framework.git
cd qtlh-framework

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Check code formatting
black .
flake8
mypy src/
```

## Project Structure
```
qtlh-framework/
├── src/
│   └── qtlh/
│       ├── quantum/
│       ├── hd/
│       ├── topology/
│       ├── language/
│       ├── integration/
│       └── validation/
├── tests/
├── docs/
└── examples/
```

## Commit Messages
- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Pull Request Process
1. Update the README.md with details of changes to the interface
2. Update documentation and examples
3. The PR will be merged once you have the sign-off of two other developers
4. Make sure CI checks pass

Thank you for contributing to QTL-H Framework!
