"""Minimal Weights & Biases helpers shared by the training entrypoints.

Usage is deliberately small: ``init`` once, ``log`` a flat dict per epoch,
``summary`` for final numbers, ``finish`` at exit. Every call is a no-op when
wandb is disabled (``--no_wandb``) or not installed, so the scripts keep
working without it.
"""
import os

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

_run = None


def add_args(parser):
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='quite', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (team/user)')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name (auto if omitted)')
    parser.add_argument('--wandb_offline', action='store_true',
                        help='Log offline (sync later with `wandb sync wandb/offline-run-*`)')
    return parser


def init(args, name=None, tags=None, group=None):
    """Start a run. Returns the wandb run or None when disabled."""
    global _run
    if wandb is None or getattr(args, 'no_wandb', False):
        return None
    if getattr(args, 'wandb_offline', False):
        os.environ.setdefault('WANDB_MODE', 'offline')
    config = {k: v for k, v in vars(args).items()
              if isinstance(v, (int, float, str, bool, type(None)))}
    _run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or name,
        tags=tags,
        group=group,
        config=config,
    )
    return _run


def log(metrics, step=None):
    if _run is not None:
        _run.log(metrics, step=step)


def summary(metrics):
    if _run is not None:
        for k, v in metrics.items():
            _run.summary[k] = v


def finish():
    global _run
    if _run is not None:
        _run.finish()
        _run = None
