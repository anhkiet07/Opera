"""BaseIrregularEncoder: the backbone-agnostic interface OPERA trains (OPERA sections 18-19).

An encoder consumes an irregular observation O = (values, timestamps, mask) over a history window and predicts
the target over a future window. The observation-process machinery (augmentation, distillation, mining) sits
entirely outside this interface, so a new backbone only has to implement `forward`.

    pred            = model(batch)                     # (B, L_pred, N)
    pred, z         = model(batch, return_repr=True)    # z: (B, D_flat) per-sample representation

`batch` is the dict produced by the dataset wrappers:
    observed_data (B, L, N) or (B, M, L_in, N), observed_tp, observed_mask, tp_to_predict,
    data_to_predict, mask_predicted_data.
"""
import torch
import torch.nn as nn


class BaseIrregularEncoder(nn.Module):
    """Interface + shared helpers. Subclasses implement `forward`."""

    #: does this backbone consume patch-shaped batches (B, M, L_in, N)?
    patch_input = True
    #: does this backbone accept an observation descriptor r = g(S)?
    accepts_descriptor = False

    def forward(self, batch, return_repr=False, obs_descriptor=None):
        raise NotImplementedError

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def observed(batch):
        return batch["observed_data"], batch["observed_tp"], batch["observed_mask"]

    @staticmethod
    def with_observation(batch, data, mask):
        """Shallow copy of `batch` whose *history* observation is replaced. Targets are untouched, which is the
        anti-leakage invariant of OPERA section 11."""
        out = dict(batch)
        out["observed_data"] = data
        out["observed_mask"] = mask
        return out


class QuITEEncoder(BaseIrregularEncoder):
    """Wraps the query-based irregular encoder in models/quite.py (iTransformer / PatchTST / ... backbones).

    The wrapper owns no observation logic: it forwards the (possibly re-observed) history to the encoder.
    """
    patch_input = True

    def __init__(self, args):
        super().__init__()
        from models.quite import Model, VARIATE_MODELS
        self.net = Model(args)
        self.backbone = args.model
        self.patch_input = args.model not in VARIATE_MODELS or bool(getattr(args, "irr_emb", False))
        self.accepts_descriptor = bool(getattr(args, "observation_aware", False))

    def forward(self, batch, return_repr=False, obs_descriptor=None):
        data, tp, mask = self.observed(batch)
        out = self.net.forecasting(batch["tp_to_predict"], data, tp, mask, return_repr=return_repr)
        return out


def build_model(cfg, data_info, device):
    """Build an encoder from a config dict + dataset info. Returns (model, args_namespace)."""
    import argparse
    import math

    m = cfg.get("model", {})
    args = argparse.Namespace(
        model=m.get("backbone", "itransformer"),
        mode=m.get("embedding", "quite"),
        irr_emb=bool(m.get("irr_emb", True)),
        hid_dim=int(m.get("hid_dim", 64)),
        nhead=int(m.get("nhead", 4)),
        nlayer=int(m.get("nlayer", 3)),
        dropout=float(m.get("dropout", 0.1)),
        adaptive_query=m.get("adaptive_query", "none"),
        observation_aware=bool(m.get("observation_aware", False)),
        patch_size=float(data_info["patch_size"]),
        stride=float(data_info["stride"]),
        history=data_info["history"],
        pred_window=data_info["pred_window"],
        ndim=data_info["input_dim"],
        maxlen=data_info.get("maxlen", 0),
        device=device,
        task="forecasting",
        seed=int(cfg.get("seed", 0)),
    )
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
    args.scale_patch_size = args.patch_size / (args.history + args.pred_window)

    def layer_of_patches(n):
        if n == 1:
            return 1
        return 1 + layer_of_patches(n / 2) if n % 2 == 0 else layer_of_patches(n + 1)

    args.patch_layer = layer_of_patches(args.npatch)
    model = QuITEEncoder(args).to(device)
    return model, args
