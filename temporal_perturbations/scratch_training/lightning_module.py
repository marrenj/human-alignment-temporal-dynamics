import os
import csv
import math
from typing import Dict, Optional, Any

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics

from temporal_perturbations.scratch_training.config import TrainingConfig, CLIPConfig
from temporal_perturbations.scratch_training.model import CLIP
from open_clip import ClipLoss
from temporal_perturbations.perturbations.perturbation_utils import choose_perturbation_strategy

class CLIPLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for image-text contrastive (CLIP-style) training on YFCC15M
    with optional perturbation injection. Includes CIFAR10/100 zero-shot validation.
    """
    def __init__(
        self,
        model_config: CLIPConfig,
        training_config: TrainingConfig,
        train_dataset_size: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_config'])

        self.model_config = model_config
        self.training_config = training_config
        self.train_dataset_size = train_dataset_size

        self.train_mode = training_config.data.dataset_type

        if self.train_mode not in ['yfcc15m', 'yfcc15m_litdata']:
            raise ValueError(f"Unsupported dataset_type in training_config: {self.train_mode}")

        self.model = CLIP(config=model_config)
        rank = self.global_rank if hasattr(self, 'global_rank') else int(os.environ.get("RANK", 0))
        world_size = self.world_size if hasattr(self, 'world_size') else int(os.environ.get("WORLD_SIZE", 1))
        loss_cfg = training_config.loss
        self.loss_fn = ClipLoss(
            local_loss=loss_cfg.local_loss,
            gather_with_grad=loss_cfg.gather_with_grad,
            cache_labels=loss_cfg.cache_labels,
            rank=rank,
            world_size=world_size,
        )
        self.logit_scale_value = self.model.logit_scale.item()
        self.classifier_weights = {}
        self._classifier_epoch = -1
        self.cifar10_classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.cifar10_templates = ["a photo of a {c}."]
        self.cifar100_classes = (
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        )
        self.cifar100_templates = ["a photo of a {c}."]
        num_classes_cifar10 = len(self.cifar10_classes)
        top_k_val_cifar10 = min(5, num_classes_cifar10) if num_classes_cifar10 > 1 else 1
        self.val_accuracy_top1_cifar10 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes_cifar10, top_k=1)
        if top_k_val_cifar10 >= 5:
            self.val_accuracy_top5_cifar10 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes_cifar10, top_k=top_k_val_cifar10)
        else:
            self.val_accuracy_top5_cifar10 = None
        num_classes_cifar100 = len(self.cifar100_classes)
        top_k_val_cifar100 = min(5, num_classes_cifar100) if num_classes_cifar100 > 1 else 1
        self.val_accuracy_top1_cifar100 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes_cifar100, top_k=1)
        if top_k_val_cifar100 >= 5:
            self.val_accuracy_top5_cifar100 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes_cifar100, top_k=top_k_val_cifar100)
        else:
            self.val_accuracy_top5_cifar100 = None
        self.val_accuracy_top1 = self.val_accuracy_top1_cifar10
        self.val_accuracy_top5 = self.val_accuracy_top5_cifar10

        # --- Perturbation Strategy ---
        self.perturbation_strategy = choose_perturbation_strategy(
            perturb_type=getattr(training_config, 'perturb_type', 'none'),
            perturb_epoch=getattr(training_config, 'perturb_epoch', 0),
            perturb_length=getattr(training_config, 'perturb_length', 0),
            perturb_seed=getattr(training_config, 'perturb_seed', 42),
        )

    def setup(self, stage: str):
        if stage == 'fit' and self.train_mode.startswith("yfcc15m"):
            pl.utilities.rank_zero_info("CIFAR10/100 classifier weights will be built at the start of each validation epoch.")
        # Override the rank/world_size that ClipLoss read from os.environ in __init__
        # (which only happens to be correct under torchrun) with values Lightning has now
        # finalized — so spawned strategies, single-GPU debug runs, or any setup where the
        # env-var path is wrong still get the right ranks for all_gather.
        if hasattr(self, "loss_fn") and hasattr(self.loss_fn, "rank"):
            self.loss_fn.rank = self.trainer.global_rank
            self.loss_fn.world_size = self.trainer.world_size

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        perturb_type = getattr(self.training_config, 'perturb_type', 'none')
        if self.perturbation_strategy.is_active_epoch(epoch):
            start = self.perturbation_strategy.start_epoch
            end = self.perturbation_strategy.end_epoch
            pl.utilities.rank_zero_info(
                f"[PERTURB] Epoch {epoch}: {perturb_type} perturbation ACTIVE "
                f"(window: epochs {start}–{end})"
            )
        else:
            if perturb_type != 'none':
                pl.utilities.rank_zero_info(
                    f"[PERTURB] Epoch {epoch}: {perturb_type} perturbation inactive"
                )

    def forward(self, image, text=None):
        if text is None:
            return self.model.encode_image(image, normalize=True)
        else:
            return self.model(image, text)

    def on_train_start(self):
        if not self.trainer.loggers:
             return
        def log_hyperparams_on_rank_zero():
            if hasattr(self.logger, "log_hyperparams"):
                hp_dict = {
                    "model_type": "ViT-B/32",
                    "embed_dim": self.model_config.embed_dim,
                    "vision_layers": self.model_config.vision_cfg.layers,
                    "vision_width": self.model_config.vision_cfg.width,
                    "vision_patch_size": self.model_config.vision_cfg.patch_size,
                    "lr": self.training_config.optimizer.lr,
                    "weight_decay": self.training_config.optimizer.weight_decay,
                    "batch_size": self.training_config.data.batch_size,
                    "epochs": self.training_config.epochs,
                    "optimizer": self.training_config.optimizer.name,
                    "scheduler": self.training_config.scheduler.name,
                    "warmup_steps": self.training_config.scheduler.warmup_steps,
                    "train_mode": self.train_mode,
                    "model/text/layers": getattr(self.model_config.text_cfg, 'layers', "N/A"),
                    "model/text/width": getattr(self.model_config.text_cfg, 'width', "N/A"),
                    "loss/type": self.loss_fn.__class__.__name__,
                }
                hp_dict["loss/logit_scale_init"] = self.model_config.init_logit_scale
                hp_dict["loss/local_loss"] = getattr(self.training_config.loss, 'local_loss', False)
                hp_dict["loss/gather_with_grad"] = getattr(self.training_config.loss, 'gather_with_grad', False)
                try:
                    self.logger.log_hyperparams(hp_dict)
                except Exception:
                    pass
        log_hyperparams_on_rank_zero()

    def training_step(self, batch, batch_idx):
        result = self._clip_step(batch, batch_idx, step_type="train")
        return result["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Dataloader convention (set up in dataset.py create_dataloaders):
        #   idx 0: YFCC15M val split (img, text)
        #   idx 1: CIFAR10
        #   idx 2: CIFAR100 (if multi_val)
        # If YFCC val isn't first, CIFAR shifts to idx 0/1.
        is_likely_yfcc_val = (isinstance(batch, (list, tuple)) and len(batch) == 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]))

        if dataloader_idx == 0 and is_likely_yfcc_val:
            try:
                loss_dict = self._clip_step(batch, batch_idx, step_type="val")
                self.log("val/yfcc15m/clip_loss", loss_dict["loss"].detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                return loss_dict
            except Exception as e:
                pl.utilities.rank_zero_warn(f"YFCC validation step failed: {e}")
                self.log("val/yfcc15m/clip_loss", 0.0, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                return None

        cifar_idx_offset = 1 if is_likely_yfcc_val else 0
        if dataloader_idx == cifar_idx_offset:
            dataset_name = "cifar10"
            accuracy_top1 = self.val_accuracy_top1_cifar10
            accuracy_top5 = self.val_accuracy_top5_cifar10
        elif dataloader_idx == cifar_idx_offset + 1:
            dataset_name = "cifar100"
            accuracy_top1 = self.val_accuracy_top1_cifar100
            accuracy_top5 = self.val_accuracy_top5_cifar100
        else:
            pl.utilities.rank_zero_warn(f"Unexpected dataloader_idx {dataloader_idx} in validation_step.")
            return None

        if dataset_name not in self.classifier_weights or self._classifier_epoch != self.current_epoch:
            try:
                self._build_zero_shot_classifier(dataset=dataset_name)
                self._classifier_epoch = self.current_epoch
            except Exception as e:
                pl.utilities.rank_zero_warn(f"Failed to build {dataset_name} classifier: {e}")
                self.log(f"val/{dataset_name}/epoch_acc1", 0.0, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
                if accuracy_top5 is not None:
                    self.log(f"val/{dataset_name}/epoch_acc5", 0.0, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
                return None

        images, target = batch
        try:
            images = images.to(self.device)
            target = target.to(self.device)
            image_features = self.model.encode_image(images, normalize=True)
            if dataset_name not in self.classifier_weights:
                raise RuntimeError(f"Classifier weights for {dataset_name} not found after build attempt.")
            classifier_weights = self.classifier_weights[dataset_name].to(image_features.device)
            logits = self.model.logit_scale.exp() * image_features @ classifier_weights
            accuracy_top1.update(logits, target)
            if accuracy_top5 is not None:
                accuracy_top5.update(logits, target)
            return None
        except Exception as e:
            pl.utilities.rank_zero_warn(f"CIFAR validation step failed for {dataset_name}: {e}")
            self.log(f"val/{dataset_name}/epoch_acc1", 0.0, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            if accuracy_top5 is not None:
                self.log(f"val/{dataset_name}/epoch_acc5", 0.0, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            return None

    def test_step(self, batch, batch_idx):
        self.log("test/loss", 0.0, sync_dist=True)
        return None

    def _log_perturbation_debug(self, images, texts, pre_texts=None):
        perturb_type = getattr(self.training_config, 'perturb_type', 'none')
        epoch = self.current_epoch
        if perturb_type == 'uniform_images':
            mean = images.mean().item()
            std = images.std().item()
            pl.utilities.rank_zero_info(
                f"[PERTURB DEBUG] uniform_images | epoch {epoch} | "
                f"image mean={mean:.4f} (expect 0.5), std={std:.6f} (expect 0.0)"
            )
        elif perturb_type == 'image_noise':
            mean = images.mean().item()
            std = images.std().item()
            pl.utilities.rank_zero_info(
                f"[PERTURB DEBUG] image_noise | epoch {epoch} | "
                f"image mean={mean:.4f} (expect ≈0.0), std={std:.4f} (expect ≈1.0)"
            )
        elif perturb_type == 'label_shuffle':
            if pre_texts is not None:
                pre_sample = pre_texts[:4, 0].tolist()
                post_sample = texts[:4, 0].tolist()
                shuffled = pre_sample != post_sample
                pl.utilities.rank_zero_info(
                    f"[PERTURB DEBUG] label_shuffle | epoch {epoch} | "
                    f"first token of first 4 texts: before={pre_sample} after={post_sample} "
                    f"(order changed={shuffled})"
                )

    def _clip_step(self, batch, batch_idx, step_type="train"):
        images, texts = batch
        if step_type == "train":
            is_active = self.perturbation_strategy.is_active_epoch(self.current_epoch)
            pre_texts = texts if (batch_idx == 0 and is_active) else None
            images, texts = self.perturbation_strategy.apply_to_batch(
                images, texts, self.device, self.current_epoch, batch_idx
            )
            if batch_idx == 0 and is_active:
                self._log_perturbation_debug(images, texts, pre_texts)
        image_features = self.model.encode_image(images, normalize=True)
        text_features = self.model.encode_text(texts, normalize=True)
        logit_scale = self.model.logit_scale.exp()
        loss = self.loss_fn(image_features, text_features, logit_scale)
        log_on_step = (step_type == "train")
        log_on_epoch = True
        prog_bar = (step_type == "train")
        self.log(f"{step_type}/clip_loss", loss, on_step=log_on_step, on_epoch=log_on_epoch, prog_bar=prog_bar, sync_dist=True)
        self.log(f"{step_type}/logit_scale", self.model.logit_scale.detach(), on_step=log_on_step, on_epoch=log_on_epoch, sync_dist=True)
        return {"loss": loss}

    def _write_val_metrics_csv(self, metrics: Dict[str, float]):
        # `trainer.log_dir` triggers an internal `strategy.broadcast(dirpath)` so
        # all ranks must access it before we early-return on non-zero ranks —
        # otherwise rank 0 calls the broadcast alone and the next collective
        # the other ranks issue (the metric reduce/barrier in
        # `_logger_connector.on_epoch_end`) deadlocks against it.
        log_dir = self.trainer.log_dir or self.trainer.default_root_dir
        if self.trainer.global_rank != 0:
            return
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "val_metrics.csv")
        row = {"epoch": self.current_epoch, **metrics}
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _get_callback_metric(self, *keys) -> Optional[float]:
        """Return the first matching key found in callback_metrics, or None."""
        for key in keys:
            if key in self.trainer.callback_metrics:
                return float(self.trainer.callback_metrics[key])
        return None

    def on_validation_epoch_end(self):
        csv_metrics: Dict[str, float] = {}
        train_loss = self._get_callback_metric("train/clip_loss_epoch", "train/clip_loss")
        if train_loss is not None:
            csv_metrics["train/clip_loss"] = train_loss

        # --- YFCC15M Loss Summary ---
        # Lightning automatically averages metrics logged with on_epoch=True
        # We retrieve it from trainer.callback_metrics for the console log
        if "val/yfcc15m/clip_loss" in self.trainer.callback_metrics:
            yfcc_loss_epoch = self.trainer.callback_metrics["val/yfcc15m/clip_loss"]
            pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: YFCC15M Val Loss = {yfcc_loss_epoch:.4f}")
            csv_metrics["val/yfcc15m/clip_loss"] = float(yfcc_loss_epoch)
        else:
            # This might happen if the first validation step failed consistently
            pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: YFCC15M Val Loss metric not found.")


        # --- CIFAR10 Accuracy Summary ---
        try:
            # Check if the metric object exists and has computed values
            if hasattr(self.val_accuracy_top1_cifar10, 'compute'):
                final_acc1_cifar10 = self.val_accuracy_top1_cifar10.compute()
                self.log("val/cifar10/epoch_acc1", final_acc1_cifar10, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: CIFAR10 Val Acc@1 = {final_acc1_cifar10.item():.4f}")
                csv_metrics["val/cifar10/epoch_acc1"] = float(final_acc1_cifar10)
                if self.val_accuracy_top5_cifar10 and hasattr(self.val_accuracy_top5_cifar10, 'compute'):
                    final_acc5_cifar10 = self.val_accuracy_top5_cifar10.compute()
                    self.log("val/cifar10/epoch_acc5", final_acc5_cifar10, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                    pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: CIFAR10 Val Acc@5 = {final_acc5_cifar10.item():.4f}")
                    csv_metrics["val/cifar10/epoch_acc5"] = float(final_acc5_cifar10)
            else:
                pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: CIFAR10 metric object not ready for compute.")
        except Exception as e:
            pl.utilities.rank_zero_warn(f"Error computing/logging CIFAR10 validation metrics: {e}")
        finally:
            # Always reset metrics
            if hasattr(self.val_accuracy_top1_cifar10, 'reset'): self.val_accuracy_top1_cifar10.reset()
            if self.val_accuracy_top5_cifar10 and hasattr(self.val_accuracy_top5_cifar10, 'reset'): self.val_accuracy_top5_cifar10.reset()


        # --- CIFAR100 Accuracy Summary ---
        try:
            if hasattr(self.val_accuracy_top1_cifar100, 'compute'):
                final_acc1_cifar100 = self.val_accuracy_top1_cifar100.compute()
                self.log("val/cifar100/epoch_acc1", final_acc1_cifar100, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: CIFAR100 Val Acc@1 = {final_acc1_cifar100.item():.4f}")
                csv_metrics["val/cifar100/epoch_acc1"] = float(final_acc1_cifar100)
                if self.val_accuracy_top5_cifar100 and hasattr(self.val_accuracy_top5_cifar100, 'compute'):
                    final_acc5_cifar100 = self.val_accuracy_top5_cifar100.compute()
                    self.log("val/cifar100/epoch_acc5", final_acc5_cifar100, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                    pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: CIFAR100 Val Acc@5 = {final_acc5_cifar100.item():.4f}")
                    csv_metrics["val/cifar100/epoch_acc5"] = float(final_acc5_cifar100)
            else:
                 pl.utilities.rank_zero_info(f"Epoch {self.current_epoch}: CIFAR100 metric object not ready for compute.")
        except Exception as e:
            pl.utilities.rank_zero_warn(f"Error computing/logging CIFAR100 validation metrics: {e}")
        finally:
            # Always reset metrics
            if hasattr(self.val_accuracy_top1_cifar100, 'reset'): self.val_accuracy_top1_cifar100.reset()
            if self.val_accuracy_top5_cifar100 and hasattr(self.val_accuracy_top5_cifar100, 'reset'): self.val_accuracy_top5_cifar100.reset()

        if csv_metrics:
            self._write_val_metrics_csv(csv_metrics)

    def _get_optimizer(self) -> Optimizer:
        cfg = self.training_config.optimizer
        params_with_wd = []
        params_without_wd = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or '.bias' in name or 'logit_scale' in name or 'norm' in name or '.ln' in name or 'positional_embedding' in name:
                 params_without_wd.append(param)
            else:
                 params_with_wd.append(param)
        param_groups = [
            {"params": params_with_wd, "weight_decay": cfg.weight_decay, "name": "params_with_wd"},
            {"params": params_without_wd, "weight_decay": 0.0, "name": "params_without_wd"},
        ]
        if cfg.name == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
            )
        elif cfg.name == "adam":
             optimizer = torch.optim.Adam(
                param_groups,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
            )
        elif cfg.name == "sgd":
             optimizer = torch.optim.SGD(
                param_groups,
                lr=cfg.lr,
                momentum=cfg.momentum if hasattr(cfg, 'momentum') else 0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.name}")
        return optimizer

    def _get_lr_scheduler(self, optimizer: Optimizer) -> Dict[str, Any]:
        cfg_scheduler = self.training_config.scheduler
        cfg_opt = self.training_config.optimizer
        cfg_data = self.training_config.data
        schedule_config = {"interval": "step", "frequency": 1}
        if self.trainer is None:
            if self.training_config.epochs and self.train_dataset_size > 0:
                effective_batch_size_per_step = cfg_data.batch_size * self.trainer.world_size * self.trainer.accumulate_grad_batches
                steps_per_epoch = math.ceil(self.train_dataset_size / max(1, effective_batch_size_per_step))
                total_steps = steps_per_epoch * self.training_config.epochs
            else:
                total_steps = 1_000_000
        else:
            if self.trainer.max_steps and self.trainer.max_steps != -1:
                total_steps = self.trainer.max_steps
            elif self.trainer.max_epochs and self.trainer.max_epochs != -1:
                if self.train_dataset_size and self.train_dataset_size > 0:
                    # Streaming (LitData) train set: Lightning reports num_training_batches
                    # as inf at configure_optimizers time, which otherwise forces the
                    # max(10000, warmup*5) fallback and a ~40x too-long cosine horizon
                    # (400k steps vs the real ~9.7k → the schedule never anneals). Derive
                    # steps/epoch from the known dataset size (OpenCLIP-style:
                    # dataset_size / global_batch), where global batch = per-GPU batch ×
                    # world_size × grad-accum.
                    effective_batch = cfg_data.batch_size * self.trainer.world_size * self.trainer.accumulate_grad_batches
                    steps_per_epoch = math.ceil(self.train_dataset_size / max(1, effective_batch))
                else:
                    limit_batches = self.trainer.limit_train_batches or 1.0
                    if isinstance(limit_batches, int):
                        steps_per_epoch = limit_batches
                    elif self.trainer.num_training_batches != float('inf'):
                        steps_per_epoch = int(self.trainer.num_training_batches * limit_batches)
                    else:
                        steps_per_epoch = max(10000, cfg_scheduler.warmup_steps * 5)
                total_steps = steps_per_epoch * self.trainer.max_epochs
            else:
                total_steps = 1_000_000
        warmup_steps = cfg_scheduler.warmup_steps
        if warmup_steps >= total_steps:
            warmup_steps = max(1, total_steps // 2)
        # Store as instance attributes so on_load_checkpoint can override them
        # before training starts. The lambdas below read from self at call time.
        self._scheduler_total_steps = total_steps
        self._scheduler_warmup_steps = warmup_steps
        pl.utilities.rank_zero_info(f"LR scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}")
        if self.trainer is not None and self.train_dataset_size and self.train_dataset_size > 0 and self.training_config.epochs:
            _implied = math.ceil(self.train_dataset_size / max(1, cfg_data.batch_size * self.trainer.world_size * self.trainer.accumulate_grad_batches)) * self.training_config.epochs
            if total_steps >= 10 * _implied:
                pl.utilities.rank_zero_warn(f"LR horizon total_steps={total_steps} is >=10x the dataset-implied {_implied} — cosine will not anneal; check dataset-size/max_steps wiring.")
        elif (self.trainer is not None and cfg_scheduler.name == "cosine"
              and getattr(self.trainer, "num_training_batches", 0) == float("inf")):
            pl.utilities.rank_zero_warn(f"LR horizon total_steps={total_steps} is a HEURISTIC GUESS: no train_dataset_size and streaming num_training_batches=inf, so the cosine may not anneal. Provide train_dataset_size for a correct horizon.")
        if cfg_scheduler.name == "cosine":
            min_lr_ratio = getattr(cfg_scheduler, 'min_lr', 0.0) / cfg_opt.lr if cfg_opt.lr > 0 else 0.0
            def lr_lambda(current_step):
                ts = self._scheduler_total_steps
                ws = self._scheduler_warmup_steps
                if ws > 0 and current_step < ws:
                    return float(current_step) / float(ws)
                decay_steps = max(1, ts - ws)
                progress = float(current_step - ws) / float(decay_steps)
                progress = max(0.0, min(1.0, progress))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                scale_factor = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
                return scale_factor
        elif cfg_scheduler.name == "const":
             def lr_lambda(current_step):
                 ws = self._scheduler_warmup_steps
                 if ws > 0 and current_step < ws:
                    return float(current_step) / float(ws)
                 return 1.0
        else:
             raise ValueError(f"Unsupported scheduler: {cfg_scheduler.name}")
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        schedule_config["scheduler"] = scheduler
        return schedule_config

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler_config = self._get_lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint['scheduler_total_steps'] = self._scheduler_total_steps
        checkpoint['scheduler_warmup_steps'] = self._scheduler_warmup_steps

    def on_load_checkpoint(self, checkpoint):
        if 'scheduler_total_steps' in checkpoint:
            self._scheduler_total_steps = checkpoint['scheduler_total_steps']
            self._scheduler_warmup_steps = checkpoint['scheduler_warmup_steps']
            rank_zero_info(
                f"Restored scheduler from checkpoint: "
                f"total_steps={self._scheduler_total_steps}, "
                f"warmup_steps={self._scheduler_warmup_steps}"
            )
        # Drop loop state so LitData's StreamingDataset does not re-validate
        # input_dir / num_workers against the original run's cache paths.
        # NOTE: this also prevents Lightning from restoring trainer.current_epoch
        # from epoch_progress.  The top-level "epoch" and "global_step" keys are
        # preserved; RestoreEpochCallback in train.py reads them and re-injects
        # the values into the fit_loop in on_train_start.
        checkpoint.pop('loops', None)

    def _build_zero_shot_classifier(self, dataset="cifar10"):
        # Check if already built for this epoch to avoid redundant computation/broadcast
        if dataset in self.classifier_weights and self._classifier_epoch == self.current_epoch:
             return

        pl.utilities.rank_zero_info(f"Building zero-shot classifier for {dataset} (Epoch {self.current_epoch})...")
        original_mode = self.model.training
        self.model.eval()

        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

        # Build the classifier in a wrapped try so any failure (tokenize, encode,
        # shape mismatch) on this rank is captured locally without raising before
        # the all-ranks coordination below. If we let the exception propagate
        # here, the failing rank skips the broadcast collective on line below
        # while the surviving ranks call it and deadlock.
        classifier_weights = None
        local_error: Optional[str] = None
        try:
            if dataset == "cifar10":
                classes = self.cifar10_classes
                templates = self.cifar10_templates
            elif dataset == "cifar100":
                classes = self.cifar100_classes
                templates = self.cifar100_templates
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")
            with torch.no_grad():
                all_texts = []
                for class_name in classes:
                    class_texts = [template.format(c=class_name) for template in templates]
                    all_texts.extend(class_texts)
                tokenized_output = self.model.tokenize(all_texts)
                text_input_ids = tokenized_output['input_ids'].to(self.device)
                text_features = self.model.encode_text(text_input_ids, normalize=True)
                embed_dim = text_features.shape[-1]
                n_classes = len(classes)
                n_templates = len(templates)
                if text_features.shape[0] == n_classes * n_templates:
                    text_features = text_features.view(n_classes, n_templates, embed_dim)
                    text_features = text_features.mean(dim=1)
                elif text_features.shape[0] != n_classes:
                    raise ValueError(
                        f"Cannot reconcile text feature shape {text_features.shape} "
                        f"with {n_classes} classes and {n_templates} templates."
                    )
                text_features = F.normalize(text_features, dim=-1)
                classifier_weights = text_features.t().contiguous()
        except Exception as e:
            local_error = f"{type(e).__name__}: {e}"

        # Cross-rank consensus: only broadcast when every rank built successfully.
        # all_reduce(MIN) on a 0/1 success flag becomes 0 if ANY rank failed.
        all_succeeded = classifier_weights is not None
        if is_distributed:
            flag = torch.tensor(
                [1 if all_succeeded else 0], dtype=torch.long, device=self.device
            )
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            all_succeeded = bool(flag.item())

        if not all_succeeded:
            pl.utilities.rank_zero_warn(
                f"At least one rank failed to build {dataset} classifier "
                f"(local_error_on_this_rank={local_error}); skipping broadcast and "
                f"leaving classifier_weights[{dataset}] unset for this epoch."
            )
            self.model.train(original_mode)
            return

        if is_distributed:
            classifier_weights = classifier_weights.to(self.device)
            torch.distributed.broadcast(classifier_weights, src=0)
        self.classifier_weights[dataset] = classifier_weights
        self.model.train(original_mode)
 
    def on_validation_epoch_start(self):
        # This function is now primarily for building the classifiers if needed
        # The actual check happens within validation_step before using the weights
        if self.train_mode.startswith("yfcc15m"):
             self.model.to(self.device) # Ensure model is on correct device
             # Attempt to build both classifiers if they haven't been built this epoch
             # Errors during build are handled within validation_step where they are used
             if "cifar10" not in self.classifier_weights or self._classifier_epoch != self.current_epoch:
                  try:
                      self._build_zero_shot_classifier(dataset="cifar10")
                  except Exception as e:
                       pl.utilities.rank_zero_warn(f"Failed to build CIFAR10 classifier on epoch start: {e}")
             if "cifar100" not in self.classifier_weights or self._classifier_epoch != self.current_epoch:
                 try:
                     self._build_zero_shot_classifier(dataset="cifar100")
                 except Exception as e:
                      pl.utilities.rank_zero_warn(f"Failed to build CIFAR100 classifier on epoch start: {e}")
            # Reset epoch marker at the START of validation epoch
             self._classifier_epoch = self.current_epoch
