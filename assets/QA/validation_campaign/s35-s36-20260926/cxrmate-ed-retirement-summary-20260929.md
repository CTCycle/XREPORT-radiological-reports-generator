# CXRMate-ED retirement summary

Date: 2026-09-29

CXRMate-ED was retired from the selectable XREPORT catalogue. The current
XREPORT compatibility adapter was replayed against the three approved fixtures
using the published preparation shape and decoding contract from the
[CXRMate-ED model card](https://huggingface.co/aehrc/cxrmate-ed):
`max_length=256`, `num_beams=4`, deterministic decoding, and one shared
clinical context. All three runs completed, but the full Findings/Impression
report hash was identical for all three distinct image fixtures.

The separate upstream-reference process was attempted in a disposable
local-files-only environment but could not reach generation. No independently
pinned Transformers environment was available, and the isolated dynamic-module
cache was missing the transitive `create_section_files.py` source file. This is
recorded as a limitation, not as a reference-model PASS or FAIL.

Because no corrected qualification gate passed, the model was removed rather
than retained as a permanent `degraded` selectable entry. The active manifest,
adapter registry, ED-only compatibility code, and ED-specific validation
wrapper were removed. Backend generation admission and the Angular page now
reject/disable any future public model whose validation status is `degraded`.

Historical report rows and provenance remain readable. The existing downloaded
CXRMate-ED snapshot was not deleted and remains available for an explicit
cleanup action.

Evidence: [diagnostic receipt](cxrmate-ed-retirement-diagnostic-20260929.json),
[historical S35 receipt](s35-cxrmate-ed-technical-receipt.json), and the
[approved case manifest](case-manifest.json).
