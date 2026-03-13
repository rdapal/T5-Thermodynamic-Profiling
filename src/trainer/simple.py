from typing import Any, Dict, Optional, Tuple
import src.trainer.base as base
import src.config as config
import src.trainer.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import time
import tdqm.auto


class SimpleTrainer(base.Trainer):
    """Trainer for a simple iteration.

    This trainer implements a simple iteration step for a single device.

    Parameters
    ----------
    loader
        A PyTorch dataloader that will be used to obtain the data at each step.
    model
        The model to train.
    optimizer
        The PyTorch optimizer used to update the models weights.
    lr_scheduler
        A learning rate scheduler configured to work with the provided
        optimizer.
    device
        The device on which the input batches will be moved.
    stats
        An object to gather statistics during training.

    Attributes
    ----------
    loader : torch.utils.data.DataLoader
        The object used to load data during training.
    model : torch.nn.Module
        The model to train as provided to the constructor.
    optimizer : torch.optim.Optimizer
        The optimizer used during training as provided to the constructor.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler
        The learning rate scheduler used during training as provided to the
        constructor.
    device : torch.device
        The device used to move the input batches as provided to the
        constructor.
    stats : src.trainer.stats.TrainerStats
        The `TrainerStats` object used to gather statistics.
    """

    def __init__(self,
                 loader: data.DataLoader,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 lr_scheduler: optim.lr_scheduler.LRScheduler,
                 device: torch.device,
                 stats: stats.TrainerStats,
                 conf: Optional[config.Config] = None):
        super().__init__(model, loader, device, stats)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.conf = conf

    def checkpoint_dict(self, i: int) -> Dict[str, Any]:
        super_dict = super().checkpoint_dict(i)
        super_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        super_dict["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        return super_dict

    def forward(self, i: int, batch: Any, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        # NOTE: zero_grad() was moved to step() so it is not
        # misattributed as forward-pass time.
        outputs = self.model(**batch, **model_kwargs)
        return outputs.loss

    def backward(self, i: int, loss: torch.Tensor) -> None:
        loss.backward()

    def optimizer_step(self, i: int) -> None:
        self.optimizer.step()
        self.lr_scheduler.step()

    def step(self, i: int, batch: Any,
             model_kwargs: Optional[Dict[str, Any]]) -> Tuple[torch.Tensor, Optional[str]]:
        """Training step with explicit data transfer timing and correct
        zero_grad placement.

        Changes from base class:
        - zero_grad() called between data transfer and forward (not inside
          forward) to avoid ~0.01ms cost into step overhead
        - Data transfer (CPU to GPU) is explicitly measured as a distinct phase.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Phase 1: Data transfer (CPU → GPU)
        self.stats.start_data_transfer()
        batch = self.process_batch(i, batch)
        self.stats.stop_data_transfer()

        # Zero gradients — logically an optimizer concern, placed here
        # so it is NOT timed as part of any phase. With set_to_none=True
        # (PyTorch default), this is ~0.01ms.
        self.optimizer.zero_grad(set_to_none=True)

        # Phase 2: Forward pass
        self.stats.start_forward()
        loss = self.forward(i, batch, model_kwargs)
        self.stats.stop_forward()

        # Phase 3: Backward pass
        self.stats.start_backward()
        self.backward(i, loss)
        self.stats.stop_backward()

        # Phase 4: Optimizer step
        self.stats.start_optimizer_step()
        self.optimizer_step(i)
        self.stats.stop_optimizer_step()

        return loss, None

    def train(self, model_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Overridden training loop to enforce a 5 minute execution limit 
        as required by the reporting guidelines for our paper!
        """
        if model_kwargs is None:
            model_kwargs = {}

        progress_bar = tqdm.auto.tqdm(range(len(self.loader)), desc="loss: N/A")
        self.stats.start_train()
        
        # --- 5-Minute Boundary Setup ---
        start_time_sec = time.perf_counter()
        time_limit_sec = 5 * 60  # 300 seconds (5 minutes)

        for i, batch in enumerate(self.loader):
            # Check elapsed time before starting the next step
            elapsed_time = time.perf_counter() - start_time_sec
            if elapsed_time >= time_limit_sec:
                print(f"\n[INFO] Reached 5-minute execution limit at step {i} ({elapsed_time:.1f}s). Terminating loop gracefully.")
                break

            self.stats.start_step()
            loss, descr = self.step(i, batch, model_kwargs)
            self.stats.stop_step()

            # Checkpointing
            if self.enable_checkpointing and self.should_save_checkpoint(i):
                self.stats.start_save_checkpoint()
                self.save_checkpoint(i)
                self.stats.stop_save_checkpoint()

            # Logging
            self.stats.log_loss(loss)
            self.stats.log_step()

            # Progress Bar
            if descr is not None:
                progress_bar.clear()
                print(descr)
            progress_bar.update(1)

        self.stats.stop_train()
        progress_bar.close()
        self.stats.log_stats()