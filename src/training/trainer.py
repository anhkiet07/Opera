"""OPERA trainer (OPERA sections 16, 31, 34).

One class drives every experiment E0-E5; which mechanisms are active comes entirely from the config:

    observation.num_views > 1 or train_processes != ['identity']  -> augmentation
    distillation.enabled                                          -> privileged distillation
    hard_mining.enabled                                           -> hard observation mining

Deployment path (section 31): only the student and a sparse observation are needed at inference; the teacher
and the candidate pool are training-time only.
"""
import json
import os
import time

import numpy as np
import torch

from src.distillation.teacher import build_teacher, assert_no_future_leakage
from src.evaluation.observation_shift import evaluate, evaluate_conditions
from src.mining.hard_miner import HardMiner
from src.observation import ObservationProcessGenerator
from src.training.objectives import opera_step


class Trainer:
    def __init__(self, cfg, model, data_obj, data_info, next_batch, device, out_dir, logger=print,
                 wandb_run=None, build_fn=None):
        self.cfg = cfg
        self.model = model
        self.data = data_obj
        self.info = data_info
        self.next_batch = next_batch
        self.device = device
        self.out = out_dir
        self.log = logger
        self.wandb = wandb_run
        os.makedirs(out_dir, exist_ok=True)

        t = cfg.get("training", {})
        self.epochs = int(t.get("epochs", 30))
        self.patience = int(t.get("patience", 8))
        self.min_epochs = int(t.get("min_epochs", 10))
        self.lr = float(t.get("lr", 1e-3))
        self.opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        self.obs_gen = ObservationProcessGenerator.from_config(cfg.get("observation"), seed=cfg.get("seed", 0),
                                                               device=device)
        self.num_views = int((cfg.get("observation") or {}).get("num_views", 1))

        mc = cfg.get("hard_mining", {}) or {}
        self.miner = HardMiner(self.obs_gen, candidate_pool=mc.get("candidate_pool", 8),
                               top_k=mc.get("top_k", 2), alpha=mc.get("alpha", 0.5),
                               warmup_epochs=mc.get("warmup_epochs", 0),
                               pool_specs=mc.get("pool")) if mc.get("enabled", False) else None

        dc = cfg.get("distillation", {}) or {}
        self.teacher = build_teacher(cfg, model, device, build_fn=build_fn, data_info=data_info) \
            if dc.get("enabled", False) else None
        self.rich_spec = (cfg.get("observation") or {}).get("rich_process", "identity")
        self.checked_leakage = False

        self.best = {"val_mse": float("inf"), "epoch": -1}
        self.epoch_times = []

    # ------------------------------------------------------------------ training
    def _views_for(self, batch, epoch):
        """The observation views this batch is trained on."""
        if self.miner is not None and self.miner.active(epoch):
            return self.miner.mine(self.model, batch, n_select=max(self.num_views, 1))
        if self.num_views <= 1 and self.obs_gen.train_specs == ["identity"]:
            return []
        return [(d, m, s) for d, m, _meta, s in
                self.obs_gen.views(batch["observed_data"], batch["observed_tp"], batch["observed_mask"],
                                   k=self.num_views, with_meta=True)]

    def train_epoch(self, epoch):
        self.model.train()
        t0 = time.time()
        n = self.info["n_train_batches"]
        agg, count = {}, 0
        for it in range(n):
            batch = self.next_batch(self.data["train_dataloader"])
            if batch is None:
                continue
            if self.teacher is not None:
                batch["_rich"] = self.obs_gen.rich_view(batch["observed_data"], batch["observed_tp"],
                                                        batch["observed_mask"])
                if not self.checked_leakage:
                    assert_no_future_leakage(batch, self.model.with_observation(batch, *batch["_rich"]))
                    self.checked_leakage = True
            views = self._views_for(batch, epoch)
            self.opt.zero_grad()
            loss, logs = opera_step(self.model, batch, views, teacher=self.teacher, cfg=self.cfg,
                                    collect_diag=(it == 0))
            loss.backward()
            self.opt.step()
            if self.teacher is not None:
                self.teacher.update(self.model)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0.0) + v
            count += 1
        dt = time.time() - t0
        self.epoch_times.append(dt)
        out = {k: v / max(count, 1) for k, v in agg.items()}
        out["train/epoch_time_s"] = dt
        return out

    def fit(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ckpt = os.path.join(self.out, "model.pt")
        for epoch in range(self.epochs):
            logs = self.train_epoch(epoch)
            val = evaluate(self.model, self.data["val_dataloader"], self.info["n_val_batches"],
                           self.next_batch, condition="identity", obs_gen=self.obs_gen)
            logs.update({"val/mse": val["mse"], "val/mae": val["mae"], "epoch": epoch})
            if self.miner is not None:
                logs.update(self.miner.summary())
            if torch.cuda.is_available():
                logs["gpu/peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20

            if val["mse"] < self.best["val_mse"]:
                self.best = {"val_mse": val["mse"], "epoch": epoch}
                torch.save({"model": self.model.state_dict(), "cfg": self.cfg, "epoch": epoch}, ckpt)
            logs["best_epoch"] = self.best["epoch"]
            self.log(f"epoch {epoch:03d} loss {logs.get('train/loss', float('nan')):.5f} "
                     f"val_mse {val['mse']:.5f} best {self.best['val_mse']:.5f}@{self.best['epoch']} "
                     f"({logs['train/epoch_time_s']:.1f}s)")
            if self.wandb is not None:
                self.wandb.log(logs, step=epoch)

            if epoch - self.best["epoch"] >= self.patience and epoch + 1 >= self.min_epochs:
                self.log("early stop")
                break

        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state["model"])
        return self.best

    # ------------------------------------------------------------------ evaluation
    def test(self, conditions=None):
        res = evaluate_conditions(self.model, self.data["test_dataloader"], self.info["n_test_batches"],
                                  self.next_batch, conditions=conditions, obs_gen=self.obs_gen, log=self.log)
        res["cost"] = {
            "n_params": self.model.n_params(),
            "train_time_per_epoch_s": float(np.mean(self.epoch_times)) if self.epoch_times else None,
            "total_train_time_s": float(np.sum(self.epoch_times)) if self.epoch_times else None,
            "epochs_run": len(self.epoch_times),
            "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None,
            "infer_ms_per_series": res["per_condition"]["identity"]["infer_ms_per_series"],
        }
        res["best"] = self.best
        if self.wandb is not None:
            flat = {f"test/{c}/mse": m["mse"] for c, m in res["per_condition"].items()}
            flat.update({f"test/{k}": v for k, v in res["summary"].items() if isinstance(v, (int, float))})
            flat.update({f"cost/{k}": v for k, v in res["cost"].items() if isinstance(v, (int, float))})
            for k, v in flat.items():
                self.wandb.summary[k] = v
        return res

    def save(self, res):
        with open(os.path.join(self.out, "results.json"), "w") as f:
            json.dump({"config": self.cfg, "info": self.info, **res}, f, indent=2)
        return os.path.join(self.out, "results.json")
