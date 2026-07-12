"""GTO integration: drive TexasSolver on logged spots, extract strategies,
and compare the agent's play to the solver (leak-finding).

Public entry points are assembled in the submodules:
  - action_mapping : solver action strings <-> project action indices
  - range_builder  : empirical (opp_range.sqlite) / tier-fallback range strings
  - spot_inputs    : action_debug spot JSON -> GTOSpotInputs (solve setup)
  - solver         : GTOSolver — runs the console_solver binary
  - extract        : navigate the solved tree to hero's node + read strategy
  - writer         : write gto_* fields into a spot JSON, preserving annotations
"""
