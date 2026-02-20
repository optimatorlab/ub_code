# Publishing ub-code to PyPI

This guide explains how to publish the `ub-code` package to PyPI so users can install it with `pip install ub-code`.

## Prerequisites

1. **Create PyPI account**: Register at [https://pypi.org/account/register/](https://pypi.org/account/register/)
2. **Create TestPyPI account** (for testing): Register at [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/)
3. **Install build tools**:
   ```bash
   pip install build twine
   ```

## Pre-Publishing Checklist

Before publishing, ensure:

- [ ] `pyproject.toml` is complete with all metadata
- [ ] `README.md` is up-to-date and informative
- [ ] Version in `ub_camera/_version.py` is correct (auto-updated by GitHub Actions)
- [ ] All dependencies are correctly specified
- [ ] Package installs correctly locally: `pip install -e .`
- [ ] SSL certificates are included in the package
- [ ] License is specified (currently MIT in `pyproject.toml`)

## Optional: Add More Metadata

You may want to enhance `pyproject.toml` with additional PyPI metadata:

```toml
[project]
# ... existing fields ...
keywords = ["camera", "opencv", "aruco", "computer-vision", "ros", "yolo"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

[project.urls]
Homepage = "https://github.com/optimatorlab/ub_code"
Repository = "https://github.com/optimatorlab/ub_code"
Issues = "https://github.com/optimatorlab/ub_code/issues"
```

## Build the Package

Build the distribution files:

```bash
# Make sure you're in the ub_code directory
cd ~/Projects/ub_code

# Clean any previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build
```

This creates two files in the `dist/` directory:
- `ub_code-YYYY-MM-DD.N.tar.gz` (source distribution)
- `ub_code-YYYY-MM-DD.N-py3-none-any.whl` (wheel distribution)

## Test on TestPyPI First

Before publishing to the real PyPI, test on TestPyPI:

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# You'll be prompted for your TestPyPI username and password
```

Test the installation from TestPyPI:

```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ub-code

# Test imports
python -c "import ub_camera; import ub_utils; print(ub_camera.__version__)"

# Deactivate and remove test environment
deactivate
rm -rf test_env
```

## Publish to PyPI

Once you've verified everything works on TestPyPI:

```bash
# Upload to real PyPI
python -m twine upload dist/*

# You'll be prompted for your PyPI username and password
```

**Note**: You can only upload a specific version once. If you need to make changes, you must bump the version number.

## Using API Tokens (Recommended)

Instead of using your password every time, use API tokens:

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/) → API tokens
2. Create a new token with scope for this project
3. Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YourAPITokenHere

[testpypi]
username = __token__
password = pypi-YourTestPyPITokenHere
```

Set proper permissions:
```bash
chmod 600 ~/.pypirc
```

## Automated Publishing with GitHub Actions

You can automate PyPI publishing when you create a new release:

1. Add your PyPI API token as a GitHub secret:
   - Go to repository Settings → Secrets → Actions
   - Add new secret: `PYPI_API_TOKEN`

2. Create `.github/workflows/publish-to-pypi.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  pypi-publish:
    runs-on: ubuntu-latest

    permissions:
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.x'

      - name: Install build tools
        run: pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: python -m twine upload dist/*
```

Then to publish:
1. Push changes to main (version auto-bumps)
2. Create a GitHub release
3. Package automatically publishes to PyPI

## After Publishing

Once published, users can install with:

```bash
pip install ub-code
```

Or with optional dependencies:

```bash
pip install ub-code[yolo]
pip install ub-code[ros]
pip install ub-code[all]
```

## Updating the Package

To release a new version:

1. Push changes to `main` branch
2. GitHub Actions automatically bumps the version in `ub_camera/_version.py`
3. Build and publish following the steps above

## Troubleshooting

**SSL certificates not included?**
- Verify `MANIFEST.in` includes the ssl directory
- Check that `include-package-data = true` is in `pyproject.toml`

**Import errors after installation?**
- Check that both `ub_camera` and `ub_utils` are listed in `packages`
- Verify `__init__.py` files exist in both package directories

**Version not updating?**
- Ensure `_version.py` is included in the package
- Check dynamic version configuration in `pyproject.toml`
