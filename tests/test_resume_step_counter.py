"""A resumed from-scratch run must continue the baseline's step counters, not restart them at 0.

The training modules drop the checkpoint's loop state on load, so RestoreEpochCallback has to
restore trainer.global_step and the logging step itself. This trains a tiny model for 3 epochs,
resumes a second run from the epoch-0 checkpoint, and checks both counters batch by batch.
"""
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.scripts.training.train_from_scratch import RestoreEpochCallback  # noqa: E402


class _Tiny(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.layer(x), y)
        self.log("train/loss", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def on_load_checkpoint(self, checkpoint):
        checkpoint.pop("loops", None)  # as CLIPLightningModule / SupervisedLightningModule do


class _Recorder(pl.Callback):
    def __init__(self):
        self.seen = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.seen.append((trainer.current_epoch, batch_idx, trainer.global_step,
                          trainer.fit_loop.epoch_loop._batches_that_stepped))


def _loader():
    g = torch.Generator().manual_seed(0)
    data = torch.utils.data.TensorDataset(torch.randn(16, 2, generator=g), torch.randn(16, 1, generator=g))
    return torch.utils.data.DataLoader(data, batch_size=2)


def _trainer(tmp_path, accumulate, callbacks):
    return pl.Trainer(max_epochs=3, accelerator="cpu", devices=1, accumulate_grad_batches=accumulate,
                      logger=False, enable_progress_bar=False, enable_model_summary=False,
                      default_root_dir=tmp_path, callbacks=callbacks)


@pytest.mark.parametrize("accumulate", [1, 2])
def test_resume_continues_global_step_and_logging_step(tmp_path, accumulate):
    ckpt = pl.callbacks.ModelCheckpoint(dirpath=tmp_path / "ckpts", filename="epoch_{epoch:03d}",
                                        auto_insert_metric_name=False, save_top_k=-1, every_n_epochs=1)
    baseline = _Recorder()
    _trainer(tmp_path, accumulate, [ckpt, baseline]).fit(_Tiny(), _loader())

    resume_path = tmp_path / "ckpts" / "epoch_000.ckpt"
    meta = torch.load(resume_path, map_location="cpu", weights_only=False)
    fork = _Recorder()
    _trainer(tmp_path, accumulate, [RestoreEpochCallback(meta["epoch"], meta["global_step"]), fork]).fit(
        _Tiny(), _loader(), ckpt_path=str(resume_path))

    assert meta["global_step"] == 8 // accumulate
    assert fork.seen == [row for row in baseline.seen if row[0] >= 1]
