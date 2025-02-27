# Contributing Guidelines

Thank you for your interest in contributing to the Medical Simulation Framework! We welcome contributions from everyone.

## Code Style

We use the following tools to maintain code quality:

### Ruff

We use [Ruff](https://github.com/charliermarsh/ruff) for linting and formatting Python code. 
To ensure your code meets our standards, install ruff and run it before submitting:

```bash
pip install ruff
ruff check .
```

### Black

We follow the [Black](https://github.com/psf/black) code style for Python. 
Black is an uncompromising code formatter that ensures consistency across the project.
To format your code according to our standards:

```bash
pip install black
black .
```

## Contribution Process

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes, following our code style guidelines
4. Add tests for your changes
5. Run the test suite to ensure all tests pass
6. Run ruff and black to ensure code quality
7. Submit a pull request with a clear description of your changes

## Pull Request Guidelines

- Update documentation as necessary
- Add or update tests as appropriate
- Keep pull requests focused on a single topic
- Reference any relevant issues in your PR description

We look forward to your contributions!