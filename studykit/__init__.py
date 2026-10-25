"""Shared infrastructure for the registered studies of plan v2 (studies/brain-supplement-v2, studies/brain-pathology-v1).

registry   study records: protocol, registration, schedule, task lock, amendments, STOP, locks, events, code provenance
stats_ext  TOST, exact sign-flip, simulated power, 4-parameter logistic fits, hierarchical bootstrap, partial Spearman
testplan   test families declared before any test is computed (with a post hoc flag)

research.py at the project root is the command line. The frozen study studies/brain-sokoban-v1 does not use this
package and is never modified by it.
"""
