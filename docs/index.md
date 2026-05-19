# NeuRepTrace

NeuRepTrace is a small Python toolkit for calibrated, time-resolved decoding of
neural representations from M/EEG data.

The project starts with public decoding benchmarks such as NOD-MEG/NOD-EEG and
THINGS-EEG/MEG. The longer-term target is to produce probability traces that
can support planning and replay analyses in task data.

For private or project-specific datasets, prefer a small adapter plus a
declarative dataset description over hard-coded experiment scripts. The
FieldTrip MAT dataset page describes the MATLAB raw/trial structure used by
PyMEGDec-style files, and the PyMEGDec migration page documents the planned
old-to-new command mapping.
