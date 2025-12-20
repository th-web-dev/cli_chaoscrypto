.PHONY: test fmt lint

test:
\tpython -m pytest -q

# Placeholder; can be extended with black/isort if desired.
fmt:
\t@echo "No formatter configured. Add one if needed."

lint:
\t@echo "No linter configured. Add one if needed."
