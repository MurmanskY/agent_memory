"""Our method lives here.

Start from `method.py` (a subclass of BaseMemory wired into the same registry and
evaluation path as every baseline), and add submodules as the design grows, e.g.:

    ours/
    ├── method.py        # entry class, registered as "ours"
    ├── consolidation.py # write-time memory construction / compression
    ├── linking.py       # memory organization (graph, clusters, ...)
    ├── retrieval.py     # query-time retrieval / reasoning over memory
    └── prompts.py       # all LLM prompts in one place

Keep every external knob in a config dict so experiments are reproducible from
`configs/ours/*.yaml` alone.
"""
