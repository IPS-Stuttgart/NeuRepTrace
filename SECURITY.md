# Security Policy

## Supported versions

NeuRepTrace is an early-stage research toolkit. Security fixes are considered for the current `main` branch and the latest published release, when a release exists. Older experimental snapshots are not supported unless they are needed to reproduce a published result and the fix can be applied safely.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Report it privately by using GitHub private vulnerability reporting when available, or email Florian Pfaff at <pfaff@ias.uni-stuttgart.de>.

Include as much of the following as possible:

- the affected NeuRepTrace version, commit, or workflow;
- a minimal reproducer or command line;
- whether private data, credentials, artifacts, or GitHub Actions secrets could be exposed;
- the expected and observed behavior;
- any known workaround.

Use synthetic or redacted data whenever possible. Do not send private participant data, credentials, or institution-internal artifacts unless explicitly requested through a secure channel.

## Response expectations

We aim to acknowledge credible reports within 7 days and provide an initial assessment or mitigation plan within 30 days. Timelines may vary for research-only workflows that require private datasets, but reports involving dependency compromise, arbitrary code execution, credential exposure, or unsafe CI behavior are prioritized.

## Disclosure

Please coordinate public disclosure until a fix, mitigation, or clear non-issue assessment is available. Security-related fixes should include regression tests or workflow checks whenever practical.
