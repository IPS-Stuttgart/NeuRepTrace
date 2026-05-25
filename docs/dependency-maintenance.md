# Dependency maintenance

NeuRepTrace uses Poetry for Python dependency locking and GitHub Actions for continuous integration. Keep dependency updates small and reviewable so regressions in numerical workflows can be isolated quickly.

Suggested routine:

- update the lock file separately from functional code changes;
- run the unit-test matrix after lock-file updates;
- keep optional machine-learning extras separate from core workflow fixes.
