"""ObservationProcessGenerator: the single API through which every observation process is produced (OPERA 5, 23).

Training views, evaluation conditions, teacher (rich) views and hard-mining candidates all go through this
class, so no experiment may hard-code a mask transformation of its own.

Config (see configs/*.yaml):

    observation:
      train_processes: ["random_dropout:0.1|random_dropout:0.4", "temporal_stride:2", ...]
      num_views: 2
      rich_process: "identity"          # the teacher's view
      candidate_processes: [...]        # hard-mining pool; defaults to train_processes
      constraints: {min_density: 0.15, ...}
"""
import torch

from src.observation import processes as P
from src.observation.constraints import ObservationConstraints
from src.observation.metadata import compute_metadata, descriptor

# Named regimes used by the configs and the evaluation suite. TRAIN and TEST are deliberately disjoint in
# severity and partly in type (OPERA section 23).
TRAIN_POOL = [
    "random_dropout:0.1", "random_dropout:0.25", "random_dropout:0.4",
    "temporal_stride:2",
    "burst:0.2", "burst:0.3",
    "variable_dropout:0.2",
    "local_mask:0.3",
]
# Seen types at unseen severities + types absent from training.
TEST_CONDITIONS = [
    "identity",
    "random_dropout:0.3", "random_dropout:0.5", "random_dropout:0.7",
    "variable_dropout:0.3", "variable_dropout:0.5",
    "temporal_stride:2", "temporal_stride:3", "temporal_stride:4",
    "burst:0.3", "burst:0.5",
    "long_gap:0.4",
    "variable_group_dropout:0.3",
    "density:0.5",
    "random_dropout:0.5+variable_dropout:0.3",
    "temporal_stride:3+variable_group_dropout:0.3",
]
# Conditions never seen in training, used for the "unseen observation process" table.
UNSEEN_CONDITIONS = [
    "random_dropout:0.7", "variable_dropout:0.5", "temporal_stride:3", "temporal_stride:4",
    "long_gap:0.4", "variable_group_dropout:0.3", "burst:0.5",
    "random_dropout:0.5+variable_dropout:0.3", "temporal_stride:3+variable_group_dropout:0.3",
]
# Matched-density triples for the density-vs-structure ablation (OPERA section 29): same expected density,
# different temporal / variable structure.
MATCHED_DENSITY = {
    0.5: ["density:0.5", "temporal_stride:2", "local_mask:0.5", "burst:0.5", "variable_dropout:0.5"],
}


class ObservationProcessGenerator:
    def __init__(self, train_processes=None, num_views=2, rich_process="identity",
                 candidate_processes=None, constraints=None, seed=0, device="cpu"):
        self.train_specs = list(train_processes or TRAIN_POOL)
        self.train_procs = [P.make_process(s) for s in self.train_specs]
        self.num_views = int(num_views)
        self.rich_spec = rich_process
        self.rich_proc = P.make_process(rich_process)
        self.candidate_specs = list(candidate_processes or self.train_specs)
        self.candidate_procs = [P.make_process(s) for s in self.candidate_specs]
        self.constraints = constraints if isinstance(constraints, ObservationConstraints) \
            else ObservationConstraints.from_config(constraints)
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))

    @classmethod
    def from_config(cls, cfg, seed=0, device="cpu"):
        cfg = (cfg or {}).copy()
        return cls(train_processes=cfg.get("train_processes"), num_views=cfg.get("num_views", 2),
                   rich_process=cfg.get("rich_process", "identity"),
                   candidate_processes=cfg.get("candidate_processes"),
                   constraints=cfg.get("constraints"), seed=seed, device=device)

    # ---------------- sampling ----------------
    def sample_process(self):
        i = int(torch.randint(len(self.train_procs), (1,), generator=self.gen, device=self.device))
        return self.train_procs[i]

    def views(self, data, tp, mask, k=None, with_meta=False):
        """k training views of the same underlying sample. Returns a list of (data, mask[, meta, spec])."""
        out = []
        for _ in range(k or self.num_views):
            proc = self.sample_process()
            d, m = P.apply(data, tp, mask, proc, self.gen)
            out.append((d, m, compute_metadata(tp, m, regime=proc.spec), proc.spec) if with_meta else (d, m))
        return out

    def rich_view(self, data, tp, mask):
        """The teacher's privileged observation (by default the full original observation)."""
        return P.apply(data, tp, mask, self.rich_proc, self.gen)

    def candidates(self, data, tp, mask, pool=None):
        """Hard-mining candidate pool: a list of (data, mask, spec) plus a (B, K) validity tensor."""
        procs = self.candidate_procs if pool is None else [P.make_process(s) for s in pool]
        items, masks = [], []
        for proc in procs:
            d, m = P.apply(data, tp, mask, proc, self.gen)
            items.append((d, m, proc.spec))
            masks.append(m)
        valid = self.constraints.filter_candidates(tp, masks, base_mask=mask)
        return items, valid

    def condition(self, data, tp, mask, spec, seed=12345):
        """Deterministic evaluation view: the same re-observation for every model."""
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        return P.apply(data, tp, mask, P.make_process(spec), g)

    # ---------------- descriptor ----------------
    @staticmethod
    def descriptor(tp, mask):
        return descriptor(tp, mask)
