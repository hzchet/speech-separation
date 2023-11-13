import logging
from collections import defaultdict

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SISDR
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ

from src.base import BaseTrainer
from src.utils import inf_loop, MetricTracker


logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            train_metrics,
            valid_metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, train_metrics, valid_metrics, optimizer, config, device, lr_scheduler)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics_tracker = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.train_metrics], writer=self.writer
        )
        self.evaluation_metrics_tracker = MetricTracker(
            "loss", *[m.name for m in self.valid_metrics], writer=self.writer
        )
        
        self.num_accumulation_iters = self.config["trainer"].get("num_accumulation_iters", 1)
        self.schedule_lr_per_epoch = self.config["trainer"].get("scheduler_lr_per_epoch", False)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["mixed_wave", "target_wave", "reference_wave", "speaker_id"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.criterion.train()
        self.train_metrics_tracker.reset()
        self.writer.mode = 'train'
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics_tracker,
                    batch_idx=batch_idx
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics_tracker.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                if self.lr_scheduler is not None:
                    self.writer.add_scalar(
                        "learning rate", self._get_lr()
                    )
                self._log_predictions(**batch)
                self._log_scalars(self.train_metrics_tracker)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics_tracker.result()
                self.train_metrics_tracker.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics
        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        if self.lr_scheduler is not None and self.schedule_lr_per_epoch:
            self._step_lr(log["test_snr0_SI-SDR"])
        
        return log
    
    def _step_lr(self, metric=None):
        assert self.lr_scheduler is not None
        
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            assert metric is not None
            self.lr_scheduler.step(metric)
        else:
            self.lr_scheduler.step()
    
    def _get_lr(self):
        if self.lr_scheduler is None or isinstance(
            self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return self.optimizer.param_groups[0]['lr']
        
        return self.lr_scheduler.get_last_lr()[0]
        
    def process_batch(self, batch, is_train: bool, metrics: MetricTracker, batch_idx: int):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        batch["loss"] = self.criterion(**batch) / self.num_accumulation_iters
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            if (batch_idx + 1) % self.num_accumulation_iters == 0 or (batch_idx + 1) == self.len_epoch:
                self.optimizer.step()
            if self.lr_scheduler is not None and not self.schedule_lr_per_epoch:
                self._step_lr()

        metrics.update("loss", batch["loss"].item())
        if is_train:
            for met in self.train_metrics:
                metrics.update(met.name, met(**batch))
        else:
            for met in self.valid_metrics:
                metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        if part != 'valid':
            self.criterion.eval()
        else:
            self.criterion.train()
        self.writer.mode = part
        self.evaluation_metrics_tracker.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics_tracker,
                    batch_idx=batch_idx
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics_tracker)
            self._log_predictions(**batch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics_tracker.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            target_wave,
            reference_wave,
            mixed_wave,
            s1,
            speaker_id,
            clf_logits,
            examples_to_log: int = 5,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return
        size = target_wave.size(0)
        random_index = torch.randint(0, size, (examples_to_log,))
        sample_rate = self.config["preprocessing"].get("sr")
        si_sdr = SISDR().to(target_wave.device)
        pesq = PESQ(fs=sample_rate, mode='wb').to(target_wave.device)
        targets = target_wave[random_index, :, :]
        references = reference_wave[random_index, :, :]
        mixes = mixed_wave[random_index, :, :]
        preds = s1[random_index, :, :]
        speakers = speaker_id[random_index]
        pred_speakers = clf_logits[random_index, :].argmax(dim=1)
        
        cols = defaultdict(list)
        for mix, target, reference, pred, speaker, pred_speaker in zip(
                mixes, targets, references, preds, speakers, pred_speakers
            ):
            cols['mixed_audio'].append(self.writer.create_audio_entry(mix, sample_rate))
            cols['target_audio'].append(self.writer.create_audio_entry(target, sample_rate))
            cols['reference_audio'].append(self.writer.create_audio_entry(reference, sample_rate))
            cols['predicted_audio'].append(self.writer.create_audio_entry(pred, sample_rate))
            cols['si-sdr'].append(si_sdr(pred, target).detach().cpu())
            cols['pesq'].append(pesq(pred, target).detach().cpu())
            cols['target_speaker_id'].append(speaker.detach().cpu())
            cols['predicted_speaker_id'].append(pred_speaker.detach().cpu())
            
        self.writer.add_table("predictions", pd.DataFrame.from_dict(cols))
    
    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
