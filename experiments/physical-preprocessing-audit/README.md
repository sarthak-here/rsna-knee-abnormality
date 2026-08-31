# Physical preprocessing audit

Private header-only preflight for the next materially distinct experiment. It measures DICOM ordering, orientation, laterality, physical spacing, and clinical sequence coverage across the full training corpus. It cannot create a competition submission and does not use gold labels for fitting.

The follow-up paired experiment will train identical lightweight heads independently on incumbent and physically canonical representations, preserving all 58 organizer-labelled studies for evaluation only.

Method choices are grounded in published knee-MRI preprocessing work: PMC11565144, PMC7343178, and PMC8521982.
