"""Candidate-pool hard observation-process mining (OPERA sections 12-15).

We want S* = argmax_S L(f(S(X)), Y) subject to validity constraints, approximated without any differentiable
mask generator:

    1. generate K candidate observation processes for the batch (through ObservationProcessGenerator);
    2. evaluate the task loss of each candidate under torch.no_grad();
    3. drop candidates that violate the constraints;
    4. keep the top-k hardest;
    5. the trainer backpropagates only through those.

Curriculum (section 15): P(S) = (1 - alpha) P_random + alpha P_hard, with alpha ramped by the scheduler, and
stage 1 (pure random) for `warmup_epochs`.
"""
import torch

from src.evaluation.metrics import compute_error


class HardMiner:
    def __init__(self, obs_gen, candidate_pool=8, top_k=2, alpha=0.5, warmup_epochs=0, pool_specs=None):
        self.obs_gen = obs_gen
        self.K = int(candidate_pool)
        self.top_k = int(top_k)
        self.alpha = float(alpha)
        self.warmup_epochs = int(warmup_epochs)
        self.pool_specs = pool_specs
        self.stats = {"n_mined": 0, "n_random": 0, "picked": {}}

    def active(self, epoch):
        return self.alpha > 0 and epoch >= self.warmup_epochs

    @torch.no_grad()
    def _candidate_losses(self, model, batch, items):
        """Per-sample loss of every candidate: (B, K)."""
        was_training = model.training
        model.eval()
        losses = []
        target, tmask = batch["data_to_predict"], batch["mask_predicted_data"]
        for data, mask, _ in items:
            pred = model(model.with_observation(batch, data, mask))
            err = ((target - pred) ** 2 * tmask).reshape(target.shape[0], -1).sum(dim=1)
            cnt = tmask.reshape(target.shape[0], -1).sum(dim=1).clamp_min(1.0)
            losses.append(err / cnt)
        if was_training:
            model.train()
        return torch.stack(losses, dim=1)

    def mine(self, model, batch, n_select=None):
        """Return a list of (data, mask, spec) views to train on for this batch.

        A fraction (1 - alpha) of the requested views is drawn at random and alpha from the hardest valid
        candidates, so the model never trains exclusively on hard cases.
        """
        n_select = int(n_select or self.top_k)
        data, tp, mask = model.observed(batch)
        items, valid = self.obs_gen.candidates(data, tp, mask, pool=self.pool_specs)

        # never ask for more hard views than there are (valid) candidates; the rest are drawn at random
        n_hard = min(max(int(round(self.alpha * n_select)), 0), len(items))
        n_rand = n_select - n_hard
        chosen = []

        if n_hard > 0:
            losses = self._candidate_losses(model, batch, items)          # (B, K)
            losses = losses.masked_fill(~valid, float("-inf"))
            # rank candidates by their mean loss over the batch (one observation process per batch keeps the
            # implementation simple and the selected process interpretable)
            per_cand = torch.where(valid, losses, torch.zeros_like(losses)).sum(0) / valid.sum(0).clamp_min(1)
            per_cand = torch.where(valid.any(0), per_cand, torch.full_like(per_cand, float("-inf")))
            order = torch.argsort(per_cand, descending=True)
            for idx in order[:n_hard]:
                d, m, spec = items[int(idx)]
                chosen.append((d, m, spec))
                self.stats["picked"][spec] = self.stats["picked"].get(spec, 0) + 1
            self.stats["n_mined"] += n_hard

        for _ in range(n_rand):
            proc = self.obs_gen.sample_process()
            from src.observation.processes import apply
            d, m = apply(data, tp, mask, proc, self.obs_gen.gen)
            chosen.append((d, m, proc.spec))
            self.stats["n_random"] += 1
        return chosen

    def summary(self):
        top = sorted(self.stats["picked"].items(), key=lambda kv: -kv[1])[:5]
        return {"mining/n_mined": self.stats["n_mined"], "mining/n_random": self.stats["n_random"],
                "mining/top_specs": ", ".join(f"{k}({v})" for k, v in top)}
