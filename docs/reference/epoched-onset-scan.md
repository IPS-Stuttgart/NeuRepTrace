# Epoched Onset Scan

The epoched onset-scan adapter converts validation-trial prediction traces into
NeuRepTrace probability/score observations.  Each trial is treated as a separate
pseudo-continuous sequence with times expressed relative to the known event
onset.  Detection receives only sequence, time, prediction, and score columns;
time zero is used only when the resulting events are evaluated.

::: neureptrace.epoched_onset_scan
