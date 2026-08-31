# Audit result

The private audit completed successfully over all 4,407 studies and 24,371 unique series. Every series was readable and had at least 80% geometry-projection coverage.

Canonical normal-vector ordering alone is unlikely to help: no instance/projection order disagreements, nonuniform-spacing series, or orientation drifts above one degree were found. The useful signal is protocol harmonization:

- In-plane pixel spacing spans 0.1367 to 0.7031 mm at the 1st and 99th percentiles.
- Through-plane spacing spans 0.6000 to 6.1858 mm at the 1st and 99th percentiles.
- 3,917 of 4,407 studies contain fluid-sensitive sequences in all three planes.
- 164 of 11,699 comparable series-level tag/geometry observations disagree on laterality.

Decision: retain geometry-projection ordering as a correctness invariant, but focus the paired GPU experiment on physical-spacing harmonization, clinical sequence selection, and laterality normalization. No competition submission was produced.
