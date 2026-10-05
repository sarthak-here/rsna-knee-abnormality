# Cache validation

Private CPU validation of the complete 288px/12-slice HDF5 cache. It verifies the
published SHA-256, byte size, study identity, tensor shape, dtype, chunking,
compression, masks, finite values, and non-empty occupied slots. It cannot create
a competition submission.

Version 2 passed the full contract: 4,407 unique studies, tensor shape
`[4407, 6, 12, 288, 288]`, exact 17,051,786,047-byte size and SHA-256,
gzip-2 per-slot chunking, 23,089 occupied slots, valid binary masks, and zero
all-zero occupied slots. The validated report is preserved beside this file.
