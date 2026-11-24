PYTHON_VERSION ?= 3.11
VENV := .venv
BIN  := $(VENV)/bin

.PHONY: setup ingest profile audit-verify test eval eval-offline eval-record eval-compare lint format clean

setup:  ## Create .venv on Python 3.11 with uv and install the project with dev tools
	uv venv --python $(PYTHON_VERSION) $(VENV)
	uv pip install --python $(BIN)/python -e ".[dev,openai,google]"

ingest:  ## Validate the raw dataset and build the SQLite database
	$(BIN)/fleet-copilot ingest

profile:  ## Write the data profile report to reports/data_profile.md
	$(BIN)/fleet-copilot profile

audit-verify:  ## Check that the audit log hash chain is intact
	$(BIN)/fleet-copilot audit-verify

test:
	$(BIN)/pytest

eval:  ## Run all eval suites; agent cases replay recorded model responses (no API key)
	$(BIN)/fleet-copilot eval --mode replay

eval-offline:  ## Only the suites that need no model at all
	$(BIN)/fleet-copilot eval --mode offline

eval-record:  ## Run agent cases against the configured model and record responses (needs its API key)
	$(BIN)/fleet-copilot eval --mode record

eval-compare:  ## Compare the latest agent eval run per provider/model
	$(BIN)/fleet-copilot eval-compare

lint:
	$(BIN)/ruff check src tests

format:
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

clean:  ## Remove generated databases, logs and reports (keeps the raw dataset)
	rm -rf data/processed data/audit data/quarantine data/actions data/checkpoints logs reports .pytest_cache .ruff_cache
