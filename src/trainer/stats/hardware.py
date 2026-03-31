"""
Hardware Monitoring TrainerStats v.3

Collects per-phase (data_transfer, forward, backward, optimizer):
  - Timing (ms) via time.perf_counter_ns().
  - Runs Checkpoint profiling time
  - NOTE: torch.cuda.synchronize() is applied conditionally by the SimpleTrainer
    based on the `--trainer_configs.simple.profile_phase` flag to minimize CUDA overhead in our results.

Collects hardware metrics (Energy, Memory, Utilization, Temp):
  - Polled strictly every >= 500ms to avoid NVML API blocking overhead.
  - Energy is inferred by reading the NVML energy counter 
    (nvmlDeviceGetTotalEnergyConsumption) across the steps that occurred within the 500ms window.
  - CO2 emissions (mg) = energy_kWh * carbon_intensity


Known Limitations
-----------------
1. GPU utilization from nvmlDeviceGetUtilizationRates is averaged over
   the driver's sampling window (~1s to 1/6s), so readings are rolling averages
2. The NVML energy counter updates every ~100ms. By throttling our polling to 500ms,
   we ensure we capture at least 5 hardware updates per software poll.
3. DataLoader iteration time (batch fetching from CPU memory) occurs
   before start_step() in the training loop and is NOT captured.
4. CPU utilization from psutil.cpu_percent(interval=None) reflects
   usage since the previous call, not a true instantaneous snapshot.

Reference
---------
Interface defined in src/trainer/stats/base.py
"""

import logging
import os
import csv
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict, fields
from typing import List, Optional

import torch

import src.config as config
import src.trainer.stats.base as base

logger = logging.getLogger(__name__)

trainer_stats_name = "hardware"

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.debug("pynvml not available — GPU hardware metrics disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil not available — CPU utilization disabled")

# ============================================================
# Number of warmup steps to exclude from steady-state statistics
# Using 6 for conservative power/energy analysis
# ============================================================
WARMUP_STEPS = 6


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    """Factory function called by the auto-discovery system
    when --trainer_stats hardware is specified
    """
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided — defaulting to cuda:0")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_train_steps = kwargs.get("num_train_steps", 100)

    # Defaults
    output_dir = "./hardware_stats"
    run_id = None
    carbon_intensity = 30.0  # Quebec (Hydro-Québec) default

    hw_conf = getattr(conf.trainer_stats_configs, "hardware", None)
    if hw_conf:
        output_dir = getattr(hw_conf, "output_dir", output_dir)
        run_id = getattr(hw_conf, "run_id", run_id)
        carbon_intensity = getattr(hw_conf, "carbon_intensity", carbon_intensity)

    return HardwareTrainerStats(
        device=device,
        output_dir=output_dir,
        run_id=run_id,
        num_train_steps=num_train_steps,
        carbon_intensity=carbon_intensity,
    )


# ============================================================
# StepRecord — one row per training step in the output CSV
# ============================================================

@dataclass
class StepRecord:
    """Trimmed and validated metrics for MilaBench reporting"""
    step_num: int

    # ---- Timing (milliseconds) ----
    step_time_ms: float
    data_transfer_time_ms: float
    forward_time_ms: float
    backward_time_ms: float
    optimizer_time_ms: float
    checkpoint_time_ms: float

    # ---- GPU Memory via PyTorch (MB) ----
    # Polled every 500ms, cached between polls
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    gpu_memory_peak_mb: float

    # ---- Energy (Joules) ----
    # Inferred from 500ms counter deltas amortized over steps
    energy_step_j: float

    # ---- CO2 Emissions ----
    co2_step_mg: float

    # ---- Environment ----
    gpu_temperature_c: float
    gpu_utilization: float
    gpu_memory_utilization: float
    cpu_utilization: float

    # ---- Timestamp ----
    timestamp: str


# ============================================================
# Main class
# ============================================================

class HardwareTrainerStats(base.TrainerStats):
    """TrainerStats implementation compliant with the 500ms polling rule
    """

    def __init__(
        self,
        device: torch.device,
        output_dir: str = "./hardware_stats",
        run_id: Optional[str] = None,
        num_train_steps: int = 100,
        carbon_intensity: float = 30.0,
    ):
        super().__init__()
        self.device = device
        self.output_dir = output_dir
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.num_train_steps = num_train_steps
        self.carbon_intensity = carbon_intensity

        os.makedirs(output_dir, exist_ok=True)

        # ---- NVML initialisation ----
        self.nvml_handle = None
        self.gpu_name = "unknown"
        self.gpu_power_limit_w = 0.0
        if PYNVML_AVAILABLE and device.type == "cuda":
            try:
                pynvml.nvmlInit()
                idx = device.index if device.index is not None else 0
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name = pynvml.nvmlDeviceGetName(self.nvml_handle)
                self.gpu_name = name.decode("utf-8") if isinstance(name, bytes) else name
                try:
                    self.gpu_power_limit_w = (
                        pynvml.nvmlDeviceGetPowerManagementLimit(self.nvml_handle) / 1000.0
                    )
                except pynvml.NVMLError:
                    logger.warning("Could not query GPU power management limit")
                logger.info(
                    f"NVML ready: {self.gpu_name}, "
                    f"power limit = {self.gpu_power_limit_w:.0f} W"
                )
            except Exception as exc:
                logger.warning(f"Could not initialise NVML: {exc}")

        # ---- Storage ----
        self.step_records: List[StepRecord] = []
        self.current_step = 0

        # ---- Step-level accumulators ----
        self._step_start_ns: int = 0
        self._dt_start_ns: int = 0
        self._fwd_start_ns: int = 0
        self._bwd_start_ns: int = 0
        self._opt_start_ns: int = 0

        self._dt_time_ns: int = 0
        self._fwd_time_ns: int = 0
        self._bwd_time_ns: int = 0
        self._opt_time_ns: int = 0

        # ---- Checkpoint State ----
        self._ckpt_start_ns: int = 0
        self._ckpt_time_ns: int = 0

        # ---- 500ms Throttling State ----
        self._last_poll_ns: int = 0
        self._last_energy_mj: float = 0.0
        self._steps_since_poll: int = 0
        
        # ---- Cached hardware states ----
        self._cached_energy_step_j: float = 0.0
        self._cached_util_gpu: float = 0.0
        self._cached_util_mem: float = 0.0
        self._cached_temp: float = 0.0
        self._cached_cpu_util: float = 0.0
        self._cached_mem_alloc: float = 0.0
        self._cached_mem_res: float = 0.0
        self._cached_mem_peak: float = 0.0

        logger.info(
            f"HardwareTrainerStats v3 initialised — "
            f"output_dir={output_dir}, run_id={self.run_id}, "
            f"gpu={self.gpu_name}, "
            f"carbon_intensity={carbon_intensity} gCO2eq/kWh"
        )

    # ==========================================================
    # Helpers
    # ==========================================================

    @staticmethod
    def _time_ns() -> int:
        return time.perf_counter_ns()

    def _get_total_energy_mj(self) -> float:
        """Reads the hardware energy counter directly (millijoules)"""
        if self.nvml_handle is not None:
            try:
                return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(self.nvml_handle))
            except pynvml.NVMLError:
                pass
        return 0.0

    def _sync_cuda(self) -> None:
        """Ensure all pending GPU kernels have completed"""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _poll_hardware(self):
        """Polls NVML and PyTorch memory - executed only every >= 500ms"""
        if self.device.type == "cuda":
            self._cached_mem_alloc = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            self._cached_mem_res = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
            self._cached_mem_peak = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats(self.device)

        if self.nvml_handle is not None:
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                self._cached_util_gpu = float(u.gpu)
                self._cached_util_mem = float(u.memory)
                self._cached_temp = float(pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU))
            except pynvml.NVMLError:
                pass

        if PSUTIL_AVAILABLE:
            try:
                self._cached_cpu_util = psutil.cpu_percent(interval=None)
            except Exception:
                pass

    # ==========================================================
    # TrainerStats interface — training lifecycle
    # ==========================================================

    def start_train(self) -> None:
        logger.info(f"Training start — hardware v3 (500ms throttling)")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None) 
            
        self._last_poll_ns = self._time_ns()
        self._last_energy_mj = self._get_total_energy_mj()

    def stop_train(self) -> None:
        logger.info("Training complete — saving hardware+energy stats...")
        self._save_results()
        if self.nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # ==========================================================
    # TrainerStats interface — step lifecycle
    # ==========================================================

    def start_step(self) -> None:
        self._step_start_ns = self._time_ns()
        self._dt_time_ns = 0
        self._fwd_time_ns = 0
        self._bwd_time_ns = 0
        self._opt_time_ns = 0
        self._ckpt_time_ns = 0

    def stop_step(self) -> None:
        step_end_ns = self._time_ns()
        step_time_ns = step_end_ns - self._step_start_ns
        self._steps_since_poll += 1

        # ---- 500ms Hardware Polling Throttle ----
        if (step_end_ns - self._last_poll_ns) >= 500_000_000:
            current_energy_mj = self._get_total_energy_mj()
            delta_j = (current_energy_mj - self._last_energy_mj) / 1000.0
            
            # Amortize energy across the steps that occurred in this window
            self._cached_energy_step_j = delta_j / max(1, self._steps_since_poll)
            
            self._poll_hardware()
            
            self._last_poll_ns = step_end_ns
            self._last_energy_mj = current_energy_mj
            self._steps_since_poll = 0

        co2_mg = (self._cached_energy_step_j * self.carbon_intensity) / 3600.0

        record = StepRecord(
            step_num=self.current_step,
            step_time_ms=step_time_ns / 1e6,
            data_transfer_time_ms=self._dt_time_ns / 1e6,
            forward_time_ms=self._fwd_time_ns / 1e6,
            backward_time_ms=self._bwd_time_ns / 1e6,
            optimizer_time_ms=self._opt_time_ns / 1e6,
            checkpoint_time_ms=self._ckpt_time_ns / 1e6,
            gpu_memory_allocated_mb=self._cached_mem_alloc,
            gpu_memory_reserved_mb=self._cached_mem_res,
            gpu_memory_peak_mb=self._cached_mem_peak,
            energy_step_j=self._cached_energy_step_j,
            co2_step_mg=co2_mg,
            gpu_temperature_c=self._cached_temp,
            gpu_utilization=self._cached_util_gpu,
            gpu_memory_utilization=self._cached_util_mem,
            cpu_utilization=self._cached_cpu_util,
            timestamp=datetime.now().isoformat(),
        )
        self.step_records.append(record)
        self.current_step += 1

    # ==========================================================
    # TrainerStats interface — Phase Hooks
    # NOTE: torch.cuda.synchronize() is handled by simple.py based on phase flags
    # ==========================================================

    def start_data_transfer(self) -> None:
        self._sync_cuda()
        self._dt_start_ns = self._time_ns()

    def stop_data_transfer(self) -> None:
        self._sync_cuda()
        self._dt_time_ns = self._time_ns() - self._dt_start_ns

    def start_forward(self) -> None:
        self._sync_cuda()
        self._fwd_start_ns = self._time_ns()

    def stop_forward(self) -> None:
        self._sync_cuda()
        self._fwd_time_ns = self._time_ns() - self._fwd_start_ns

    def start_backward(self) -> None:
        self._sync_cuda()
        self._bwd_start_ns = self._time_ns()

    def stop_backward(self) -> None:
        self._sync_cuda()
        self._bwd_time_ns = self._time_ns() - self._bwd_start_ns

    def start_optimizer_step(self) -> None:
        self._sync_cuda()
        self._opt_start_ns = self._time_ns()

    def stop_optimizer_step(self) -> None:
        self._sync_cuda()
        self._opt_time_ns = self._time_ns() - self._opt_start_ns
    
    def start_save_checkpoint(self) -> None:
        self._sync_cuda()
        self._ckpt_start_ns = self._time_ns()

    def stop_save_checkpoint(self) -> None:
        self._sync_cuda()
        self._ckpt_time_ns = self._time_ns() - self._ckpt_start_ns

    # ==========================================================
    # Loss / Logging
    # ==========================================================

    def log_loss(self, loss: torch.Tensor) -> None:
        pass

    def log_step(self) -> None:
        """Log the most recent step to console."""
        if not self.step_records:
            return
        rec = self.step_records[-1]
        print(
            f"[Step {rec.step_num:4d}] "
            f"total={rec.step_time_ms:.1f}ms "
            f"E={rec.energy_step_j * 1000:.1f}mJ | "
            f"T={rec.gpu_temperature_c:.0f}°C | "
            f"mem={rec.gpu_memory_allocated_mb:.0f}MB | "
            f"util={rec.gpu_utilization:.0f}%"
        )

    def log_stats(self) -> None:
        """Print summary statistics at end of training"""
        if len(self.step_records) < WARMUP_STEPS + 1:
            logger.warning(
                f"Only {len(self.step_records)} steps recorded, "
                f"need > {WARMUP_STEPS} for steady-state stats"
            )
            return

        steady = self.step_records[WARMUP_STEPS:]
        all_recs = self.step_records

        print("\n" + "=" * 65)
        print("HARDWARE + ENERGY SUMMARY  (v3)")
        print("=" * 65)
        print(f"Total steps: {len(all_recs)} ({WARMUP_STEPS} warmup, {len(steady)} steady-state)")

        # ---- Timing ----
        print(f"\n--- Timing (steady state, ms) ---")
        for label, attr in [
            ("Data xfer", "data_transfer_time_ms"),
            ("Forward",   "forward_time_ms"),
            ("Backward",  "backward_time_ms"),
            ("Optimizer", "optimizer_time_ms"),
            ("Checkpoint","checkpoint_time_ms"),
            ("Total step","step_time_ms"),
        ]:
            vals = [getattr(r, attr) for r in steady]
            avg = sum(vals) / len(vals)
            std = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
            print(f"  {label:12s}: {avg:8.2f} ± {std:.2f} ms")

        # ---- Energy ----
        print(f"\n--- Energy ---")
        total_e = sum(r.energy_step_j for r in all_recs)
        avg_e_mj = sum(r.energy_step_j for r in steady) / len(steady) * 1000
        print(f"  Total training:    {total_e:.1f} J ({total_e / 3600:.4f} Wh)")
        print(f"  Avg steady step:   {avg_e_mj:.1f} mJ")

        print("=" * 65 + "\n")

    # ==========================================================
    # Persistence
    # ==========================================================

    def _save_results(self) -> None:
        """Write CSV + JSON metadata"""
        if not self.step_records:
            logger.warning("No step records to save")
            return

        # ---- CSV ----
        csv_path = os.path.join(
            self.output_dir, f"hardware_stats_{self.run_id}.csv"
        )
        field_names = [f.name for f in fields(StepRecord)]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for rec in self.step_records:
                writer.writerow(asdict(rec))
        logger.info(f"CSV saved: {csv_path} ({len(self.step_records)} rows)")

        # ---- JSON metadata ----
        steady = self.step_records[WARMUP_STEPS:]
        total_energy = sum(r.energy_step_j for r in self.step_records)
        total_co2 = sum(r.co2_step_mg for r in self.step_records)

        metadata = {
            "version": 3,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "gpu": {
                "name": self.gpu_name,
                "power_limit_w": self.gpu_power_limit_w,
            },
            "config": {
                "num_steps": len(self.step_records),
                "warmup_steps": WARMUP_STEPS,
                "steady_state_steps": len(steady),
                "carbon_intensity_gco2eq_per_kwh": self.carbon_intensity,
            },
            "totals": {
                "energy_j": round(total_energy, 3),
                "energy_wh": round(total_energy / 3600, 6),
                "co2_mg": round(total_co2, 4),
                "co2_g": round(total_co2 / 1000, 7),
            },
            "steady_state_averages": {
                "step_time_ms": round(
                    sum(r.step_time_ms for r in steady) / len(steady), 2
                ) if steady else 0,
                "energy_step_mj": round(
                    sum(r.energy_step_j for r in steady) / len(steady) * 1000, 1
                ) if steady else 0,
                "gpu_utilization_pct": round(
                    sum(r.gpu_utilization for r in steady) / len(steady), 1
                ) if steady else 0,
                "gpu_temperature_c": round(
                    sum(r.gpu_temperature_c for r in steady) / len(steady), 1
                ) if steady else 0,
            },
            "known_limitations": [
                "Hardware metrics polled every >= 500ms to reduce overhead",
                "Energy inferred from NVML total counter and amortized across steps in window",
                "Phase synchronization controlled by SimpleTrainer to isolate overhead",
            ],
        }

        json_path = os.path.join(
            self.output_dir, f"metadata_{self.run_id}.json"
        )
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved: {json_path}")
