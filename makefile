# Define variables
VENV_DIR := venv
PIP := $(VENV_DIR)\Scripts\pip.exe

.PHONY: init lint test clean

# Set up virtual environment and install requirements
init:
	@echo "Setting up virtual environment..."
	if not exist $(VENV_DIR) python -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Lint the codebase
lint:
	@echo "Linting codebase with flake8 and black..."
	$(VENV_DIR)\Scripts\flake8 .
	$(VENV_DIR)\Scripts\black --check .

# Run tests
test:
	@echo "Running tests with pytest..."
	$(VENV_DIR)\Scripts\pytest

# Clean up temporary and cache files
clean:
	@echo "Removing Python file artifacts..."
	if exist *.pyc for /r %%i in (*.pyc) do del %%i
	if exist __pycache__ for /d %%i in (__pycache__) do rmdir /s /q %%i
	@echo "Removing build-related artifacts..."
	if exist build rmdir /s /q build
	if exist dist rmdir /s /q dist
	if exist *.egg-info rmdir /s /q *.egg-info