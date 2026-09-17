# Thin shim for Linux/CI. Windows users: `uv run poe <task>` (same task names).
.PHONY: help lint fmt typecheck test test-unit test-integration cov check dev up down logs ps

help:
	@echo "Targets: lint fmt typecheck test test-unit test-integration cov check dev up down logs ps"

lint fmt typecheck test test-unit test-integration cov check dev up down logs ps:
	uv run poe $@
