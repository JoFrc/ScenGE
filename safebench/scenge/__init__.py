"""ScenGE-specific logic.

This subpackage holds everything that is unique to the ScenGE paper and is
*not* part of the borrowed SafeBench / ChatScene infrastructure:

- :mod:`safebench.scenge.msgen` -- LLM + RAG generation of Scenic scripts.
- :mod:`safebench.scenge.threat` -- attention-based segment selection and
  PGD trajectory perturbation used to amplify scenario threat.
"""
