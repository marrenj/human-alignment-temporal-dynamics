"""LightningModule for supervised ImageNet-1K classification with perturbation injection.

Parallels lightning_module.py (CLIPLightningModule) but for N-way classification:
cross-entropy loss, top-1/top-5 accuracy, timm backbone. Perturbations are applied in
training_step on (images, labels) exactly as in the CLIP path — image_noise/uniform_images
replace the images, label_shuffle permutes the integer labels within the batch.
"""
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics

from temporal_perturbations.scratch_training.models_supervised import build_classifier
from temporal_perturbations.perturbations.perturbation_utils import choose_perturbation_strategy


class SupervisedLightningModule(pl.LightningModule):
    def __init__(self, model_config, training_config, train_dataset_size: int = 0):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])
        self.model_config = model_config
        self.training_config = training_config
        self.train_dataset_size = train_dataset_size

        self.model = build_classifier(
            arch=model_config.name,
            num_classes=model_config.num_classes,
            pretrained=model_config.pretrained,
            drop_path_rate=model_config.drop_path_rate,
            image_size=model_config.image_size,
        )

        label_smoothing = getattr(training_config.loss, "label_smoothing", 0.0)
        n = model_config.num_classes

        # MixUp + CutMix (DeiT / ConvNeXt recipe). When active, targets become a soft
        # (B, num_classes) distribution and the training loss switches to soft-target
        # CE. Label smoothing is rolled INTO the soft targets by Mixup, so the
        # train criterion does NOT apply additional smoothing in that branch.
        data_cfg = training_config.data
        mixup_alpha = float(getattr(data_cfg, "mixup_alpha", 0.0) or 0.0)
        cutmix_alpha = float(getattr(data_cfg, "cutmix_alpha", 0.0) or 0.0)
        if mixup_alpha > 0 or cutmix_alpha > 0:
            from timm.data import Mixup
            from timm.loss import SoftTargetCrossEntropy
            self.mixup_fn = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                cutmix_minmax=None,
                prob=float(getattr(data_cfg, "mixup_prob", 1.0) or 0.0),
                switch_prob=float(getattr(data_cfg, "mixup_switch_prob", 0.5) or 0.0),
                mode=getattr(data_cfg, "mixup_mode", "batch") or "batch",
                label_smoothing=label_smoothing,
                num_classes=n,
            )
            self.train_criterion = SoftTargetCrossEntropy()
        else:
            self.mixup_fn = None
            self.train_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Validation always uses hard targets — never mixed at val time.
        self.val_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Retained for back-compat; pre-mixup code path still references self.criterion.
        self.criterion = self.train_criterion

        self.train_acc1 = torchmetrics.Accuracy(task="multiclass", num_classes=n, top_k=1)
        self.val_acc1 = torchmetrics.Accuracy(task="multiclass", num_classes=n, top_k=1)
        self.val_acc5 = torchmetrics.Accuracy(task="multiclass", num_classes=n, top_k=5)

        self.perturbation_strategy = choose_perturbation_strategy(
            perturb_type=getattr(training_config, "perturb_type", "none"),
            perturb_epoch=getattr(training_config, "perturb_epoch", 0),
            perturb_length=getattr(training_config, "perturb_length", 0),
            perturb_seed=getattr(training_config, "perturb_seed", 42),
        )

    def forward(self, images):
        return self.model(images)

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        ptype = getattr(self.training_config, "perturb_type", "none")
        if self.perturbation_strategy.is_active_epoch(epoch):
            rank_zero_info(
                f"[PERTURB] Epoch {epoch}: {ptype} ACTIVE "
                f"(window epochs {self.perturbation_strategy.start_epoch}-{self.perturbation_strategy.end_epoch})"
            )
        elif ptype != "none":
            rank_zero_info(f"[PERTURB] Epoch {epoch}: {ptype} inactive")

    def training_step(self, batch, batch_idx):
        images, targets = batch
        # Apply MixUp/CutMix FIRST (data augmentation regularizer). When active, targets
        # become a soft (B, num_classes) distribution; the train criterion (SoftTargetCE)
        # handles that. Mixup is a no-op for validation, so this only runs during training.
        if self.mixup_fn is not None:
            images, targets = self.mixup_fn(images, targets)

        # Apply perturbations SECOND (training-disruption intervention). label_shuffle's
        # `targets[perm]` works on both hard and soft targets; image perturbations don't
        # touch targets. This keeps the perturbation semantics consistent regardless of
        # whether mixup is enabled.
        is_active = self.perturbation_strategy.is_active_epoch(self.current_epoch)
        pre_targets = targets[:4].detach().clone() if (batch_idx == 0 and is_active) else None
        images, targets = self.perturbation_strategy.apply_to_batch(
            images, targets, self.device, self.current_epoch, batch_idx
        )
        if batch_idx == 0 and is_active:
            self._log_perturb_debug(images, targets, pre_targets)

        logits = self.model(images)
        loss = self.train_criterion(logits, targets)
        # Accuracy needs integer labels — collapse soft targets via argmax. This makes
        # training accuracy slightly optimistic under mixup (correct iff predicted ==
        # dominant mixed class) but the trend is still informative.
        targets_for_metric = targets.argmax(dim=-1) if targets.dim() > 1 else targets
        self.train_acc1(logits, targets_for_metric)
        self.log("train/ce_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc1", self.train_acc1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, targets = batch
        logits = self.model(images)
        loss = self.val_criterion(logits, targets)
        self.val_acc1.update(logits, targets)
        self.val_acc5.update(logits, targets)
        self.log("val/imagenet/ce_loss", loss, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        return None

    def on_validation_epoch_end(self):
        acc1 = self.val_acc1.compute()
        acc5 = self.val_acc5.compute()
        self.log("val/imagenet/acc1", acc1, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        self.log("val/imagenet/acc5", acc5, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        rank_zero_info(f"Epoch {self.current_epoch}: ImageNet val acc@1={acc1.item():.4f} acc@5={acc5.item():.4f}")
        self.val_acc1.reset()
        self.val_acc5.reset()

    def _log_perturb_debug(self, images, targets, pre_targets):
        ptype = getattr(self.training_config, "perturb_type", "none")
        if ptype in ("uniform_images", "image_noise"):
            rank_zero_info(
                f"[PERTURB DEBUG] {ptype} | epoch {self.current_epoch} | "
                f"image mean={images.mean().item():.4f} std={images.std().item():.4f}"
            )
        elif ptype == "label_shuffle" and pre_targets is not None:
            post = targets[:4].tolist()
            rank_zero_info(
                f"[PERTURB DEBUG] label_shuffle | epoch {self.current_epoch} | "
                f"first 4 labels before={pre_targets} after={post} (changed={pre_targets != post})"
            )

    def _get_optimizer(self) -> Optimizer:
        cfg = self.training_config.optimizer
        with_wd, without_wd = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name.endswith(".bias") or "norm" in name:
                without_wd.append(p)
            else:
                with_wd.append(p)
        groups = [
            {"params": with_wd, "weight_decay": cfg.weight_decay},
            {"params": without_wd, "weight_decay": 0.0},
        ]
        if cfg.name == "adamw":
            return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
        if cfg.name == "adam":
            return torch.optim.Adam(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
        if cfg.name == "sgd":
            momentum = cfg.momentum or 0.9
            nesterov = bool(getattr(cfg, "nesterov", False)) and momentum > 0
            return torch.optim.SGD(groups, lr=cfg.lr, momentum=momentum, nesterov=nesterov)
        raise ValueError(f"Unsupported optimizer: {cfg.name}")

    def _total_steps(self) -> int:
        # configure_optimizers fires before the dataloader is materialized, so
        # self.trainer.num_training_batches is usually `inf` here. The CLIP module
        # handles this by falling back to a generous default; we mirror that.
        # IMPORTANT: the previous fallback used self.train_dataset_size // batch_size,
        # but train_dataset_size is passed in as 0 from train.py, which silently
        # produced total_steps == max_epochs. That collapsed the LR to ~min_lr after
        # the first few dozen batches and the model never learned (val acc stuck at
        # ~2% for 37 epochs of the first resnet50 baseline run).
        if not self.trainer or not self.trainer.max_epochs or self.trainer.max_epochs == -1:
            return 1_000_000
        if self.trainer.max_steps and self.trainer.max_steps != -1:
            return int(self.trainer.max_steps)

        if self.trainer.num_training_batches != float("inf"):
            steps_per_epoch = int(self.trainer.num_training_batches)
        else:
            # Estimate from a known dataset size if we have one; otherwise fall back
            # to a sensible large number tied to warmup_steps so the cosine schedule
            # at least covers the run instead of hitting min_lr immediately.
            ds_size = self.train_dataset_size
            if ds_size <= 0 and getattr(self.training_config.data, "dataset_type", "") == "imagenet1k":
                ds_size = 1_281_167  # ImageNet-1K train split
            if ds_size > 0:
                world = max(1, getattr(self.trainer, "world_size", 1) or 1)
                accum = max(1, getattr(self.trainer, "accumulate_grad_batches", 1) or 1)
                effective_batch = self.training_config.data.batch_size * world * accum
                steps_per_epoch = max(1, ds_size // effective_batch)
            else:
                warmup = (self.training_config.scheduler.warmup_steps or 0) or 1
                steps_per_epoch = max(10_000, warmup * 5)
        return max(1, steps_per_epoch * self.trainer.max_epochs)

    def _get_lr_scheduler(self, optimizer: Optimizer) -> Dict[str, Any]:
        cfg = self.training_config.scheduler
        total_steps = self._total_steps()
        warmup_steps = cfg.warmup_steps or 0
        if warmup_steps >= total_steps:
            warmup_steps = max(1, total_steps // 2)
        self._scheduler_total_steps = total_steps
        self._scheduler_warmup_steps = warmup_steps
        rank_zero_info(f"LR scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}")
        min_lr_ratio = (getattr(cfg, "min_lr", 0.0) or 0.0) / self.training_config.optimizer.lr

        def lr_lambda(step):
            ws, ts = self._scheduler_warmup_steps, self._scheduler_total_steps
            if ws > 0 and step < ws:
                return step / float(ws)
            progress = min(1.0, max(0.0, (step - ws) / float(max(1, ts - ws))))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return {"scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda), "interval": "step", "frequency": 1}

    def configure_optimizers(self):
        opt = self._get_optimizer()
        return {"optimizer": opt, "lr_scheduler": self._get_lr_scheduler(opt)}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["scheduler_total_steps"] = self._scheduler_total_steps
        checkpoint["scheduler_warmup_steps"] = self._scheduler_warmup_steps

    def on_load_checkpoint(self, checkpoint):
        if "scheduler_total_steps" in checkpoint:
            self._scheduler_total_steps = checkpoint["scheduler_total_steps"]
            self._scheduler_warmup_steps = checkpoint["scheduler_warmup_steps"]
        # Mirror the CLIP module: drop loop state so a fork-from-baseline resume
        # doesn't restore the old epoch counter (RestoreEpochCallback handles that).
        checkpoint.pop("loops", None)
