# Use STORAGE_DIR to specify a custom storage directory for the virtual environment, cache, and models.
# Example: make install STORAGE_DIR=/mnt/storage/callmemaybe
STORAGE_DIR ?= $(CURDIR)

VENV_DIR      := $(STORAGE_DIR)/.venv
UV_CACHE_DIR  ?= $(STORAGE_DIR)/.uv-cache
MODEL_DIR     ?= $(STORAGE_DIR)/models

UV := $(shell command -v uv 2>/dev/null || \
	printf '%s' "$(HOME)/.local/bin/uv")

export UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT := $(VENV_DIR)
export HF_HOME := $(MODEL_DIR)

.PHONY: ensure-uv install run debug clean lint lint-strict

ensure-uv:
	@if ! command -v uv >/dev/null 2>&1 && [ ! -x "$(UV)" ]; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

install: ensure-uv
	$(UV) sync

run: ensure-uv
	$(UV) run python -m src

debug: ensure-uv
	$(UV) run python -m pdb -m src

clean:
	@find . -type d \( -name ".git" -o -name ".venv" \) -prune -o \
		-type d \( -name "__pycache__" -o -name ".mypy_cache" \
		-o -name ".pytest_cache" \) -prune -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete

lint: ensure-uv
	$(UV) run flake8 .
	$(UV) run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict: ensure-uv
	$(UV) run flake8 .
	$(UV) run mypy . --strict