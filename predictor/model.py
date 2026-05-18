"""
predictor/model.py — Нейросетевая компонента: прямой многошаговый квантильный GRU.

Архитектура:
  Вход : последовательность w_input шагов, каждый шаг = 7 признаков
         [r̃_t, sin h, cos h, sin d, cos d, sin m, cos m]
  Тело : 2 слоя GRU(hidden_dim) с межслойным dropout
  Голова: Linear(hidden_dim → 64) + ReLU + Dropout + Linear(64 → n_quantiles*horizon_h)
  Выход: тензор формы (batch, horizon_h, n_quantiles).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pinball loss для многошагового квантильного выхода
# ─────────────────────────────────────────────────────────────────────────────

class QuantileLoss(nn.Module):
    """
    Многошаговая pinball-функция потерь.

    preds  : (batch, horizon_h, n_q)
    target : (batch, horizon_h)
    """

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.register_buffer(
            "q", torch.tensor(list(quantiles), dtype=torch.float32).view(1, 1, -1)
        )

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target: (B, h) → (B, h, 1)
        e = target.unsqueeze(-1) - preds                      # (B, h, n_q)
        loss = torch.maximum(self.q * e, (self.q - 1.0) * e)   # (B, h, n_q)
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Архитектура: прямой многошаговый квантильный GRU
# ─────────────────────────────────────────────────────────────────────────────

class GRUQuantileNet(nn.Module):
    """
    Прямой многошаговый квантильный GRU с residual-выходом.

    Архитектурные особенности:
      • два слоя GRU с межслойным dropout;
      • голова из LayerNorm + Linear+GELU+Dropout + Linear, выдающая h*n_q значений;
      • residual-связь: к каждому квантилю добавляется первая компонента входа
        последнего шага (как правило — нормализованное cpu_t). Это даёт модели
        «бесплатный» AR(1)-приор и фокусирует обучение на отклонениях от него.

    Параметры:
      d_step      : число признаков на временной шаг
      hidden_dim  : размерность скрытого состояния GRU
      n_layers    : число рекуррентных слоёв
      dropout     : dropout между слоями GRU и в голове
      n_quantiles : сколько квантилей предсказывать
      horizon_h   : горизонт прогноза (число шагов вперёд)
      residual_idx: индекс признака во входном векторе, который используется
                    как остаточный приор (по умолчанию 0).

    Параметр d_in оставлен для обратной совместимости.
    """

    def __init__(
        self,
        d_in: int = 8,
        d_step: int = 8,
        hidden_dim: int = 96,
        n_layers: int = 2,
        dropout: float = 0.20,
        n_quantiles: int = 3,
        horizon_h: int = 3,
        residual_idx: int = 0,
    ):
        super().__init__()
        self.d_step = d_step
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_quantiles = n_quantiles
        self.horizon_h = horizon_h
        self.residual_idx = int(residual_idx)

        self.gru = nn.GRU(
            input_size=d_step,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, n_quantiles * horizon_h),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, d_step)
        возвращает : (batch, horizon_h, n_quantiles)

        Residual-prior: к каждому квантилю каждого шага горизонта добавляется
        последнее значение приоритетного канала (residual_idx). Это бесплатно
        даёт сети AR(1)-приор; голова GRU обучается предсказывать только дельту.
        """
        out, _ = self.gru(x)                                 # (B, seq, hidden)
        last = out[:, -1, :]                                  # (B, hidden)
        delta = self.head(last)                               # (B, n_q*h)
        delta = delta.view(-1, self.horizon_h, self.n_quantiles)
        anchor = x[:, -1, self.residual_idx].view(-1, 1, 1)  # (B, 1, 1)
        return delta + anchor


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset для прямого многошагового обучения
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Скользящее окно для прямого многошагового обучения.

    Каждый пример (X, y):
      X : (w_input, d_step) — признаки на каждом шаге окна
      y : (horizon_h,)      — следующие horizon_h нормализованных значений целевой переменной

    По умолчанию (при `extra_channel=None`) используется одна канал-целевая
    переменная + 6 тригонометрических признаков → d_step = 7.
    Если `extra_channel` передан, он добавляется как второй сигнал → d_step = 8.
    Это позволяет одновременно подавать модель и raw cpu_norm, и residual_norm.
    """

    def __init__(
        self,
        target: np.ndarray,
        timestamps: np.ndarray,
        phi: Optional[np.ndarray] = None,  # сохранён для совместимости, не используется
        w_input: int = 60,
        horizon_h: int = 3,
        extra_channel: Optional[np.ndarray] = None,
    ):
        self.target = np.asarray(target, dtype=np.float32)
        self.timestamps = np.asarray(timestamps, dtype=np.int64)
        self.w = int(w_input)
        self.h = int(horizon_h)
        self.n = max(0, len(self.target) - self.w - self.h + 1)
        if extra_channel is not None:
            extra = np.asarray(extra_channel, dtype=np.float32)
            if len(extra) != len(self.target):
                raise ValueError(
                    f"extra_channel length {len(extra)} != target length {len(self.target)}"
                )
            self.extra = extra
        else:
            self.extra = None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        w, h = self.w, self.h
        x_window = self.target[idx: idx + w]
        ts_window = self.timestamps[idx: idx + w]
        y_future = self.target[idx + w: idx + w + h]

        # Тригонометрические временные признаки
        hours = (ts_window % 86400) / 3600.0
        dows = ((ts_window // 86400) % 7).astype(np.float32)
        minutes = (ts_window % 3600) / 60.0

        sin_h = np.sin(2 * np.pi * hours / 24.0)
        cos_h = np.cos(2 * np.pi * hours / 24.0)
        sin_d = np.sin(2 * np.pi * dows / 7.0)
        cos_d = np.cos(2 * np.pi * dows / 7.0)
        sin_m = np.sin(2 * np.pi * minutes / 60.0)
        cos_m = np.cos(2 * np.pi * minutes / 60.0)

        if self.extra is not None:
            extra_window = self.extra[idx: idx + w]
            X = np.stack(
                [x_window, extra_window, sin_h, cos_h, sin_d, cos_d, sin_m, cos_m], axis=1
            ).astype(np.float32)
        else:
            X = np.stack(
                [x_window, sin_h, cos_h, sin_d, cos_d, sin_m, cos_m], axis=1
            ).astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(y_future.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Тренер
# ─────────────────────────────────────────────────────────────────────────────

class GRUTrainer:
    """
    Обучение GRUQuantileNet (прямой многошаговый режим).

    Особенности:
      • CUDA по умолчанию, если доступна;
      • ранняя остановка при patience > 0;
      • восстановление лучших весов на CPU;
      • cosine-warm-restart планировщик скорости обучения.
    """

    def __init__(
        self,
        model: nn.Module,
        quantiles: List[float] = (0.025, 0.5, 0.975),
        lr: float = 1e-3,
        lr_decay: float = 0.99,
        grad_clip: float = 1.0,
        max_epochs: int = 60,
        patience: int = 8,
        batch_size: int = 64,
        device: Optional[str] = None,
        weight_decay: float = 1e-4,
    ):
        self.model = model
        self.quantiles = list(quantiles)
        self.lr = float(lr)
        self.lr_decay = float(lr_decay)
        self.grad_clip = float(grad_clip)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.criterion = QuantileLoss(self.quantiles).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss: float = float("inf")
        self.best_state: Optional[dict] = None
        self.epochs_no_improve: int = 0
        self.stopped_epoch: int = 0

    # ─────────────────────────────────────────────────────────────────────
    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epoch_callback=None,
    ) -> None:
        """Обучает модель на train_dataset, валидируется на val_dataset."""
        if len(train_dataset) == 0:
            logger.warning("Empty train dataset; skipping fit.")
            return

        pin = self.device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=pin,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(self.batch_size, 128),
            shuffle=False,
            pin_memory=pin,
            num_workers=0,
        )

        for epoch in range(1, self.max_epochs + 1):
            # — Тренировка —
            self.model.train()
            running = 0.0
            n_batches = 0
            for X, y in train_loader:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                preds = self.model(X)                  # (B, h, n_q)
                loss = self.criterion(preds, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                running += float(loss.item())
                n_batches += 1
            train_loss = running / max(n_batches, 1)
            self.train_losses.append(train_loss)

            # — Валидация —
            val_loss = self._evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.scheduler.step()

            improved = val_loss + 1e-6 < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if epoch_callback is not None:
                epoch_callback(epoch, train_loss, val_loss, self.best_val_loss)

            logger.info(
                "epoch %3d  train=%.5f  val=%.5f  best=%.5f  no_improve=%d",
                epoch, train_loss, val_loss, self.best_val_loss, self.epochs_no_improve,
            )

            if self.patience > 0 and self.epochs_no_improve >= self.patience:
                self.stopped_epoch = epoch
                logger.info("Early stopping at epoch %d.", epoch)
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.model.to(self.device)

    # ─────────────────────────────────────────────────────────────────────
    def _evaluate(self, loader: DataLoader) -> float:
        if len(loader.dataset) == 0:
            return float("inf")
        self.model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                preds = self.model(X)
                total += float(self.criterion(preds, y).item())
                n += 1
        return total / max(n, 1)

    # ─────────────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Прогноз для одной (или нескольких) последовательностей.

        X может иметь форму:
          (w, d_step)            → одна последовательность
          (1, w, d_step)         → одна последовательность
          (B, w, d_step)         → батч

        Возвращает numpy-массив формы (horizon_h, n_quantiles) для одиночного входа
        или (B, horizon_h, n_quantiles) для батча.
        """
        self.model.eval()
        with torch.no_grad():
            t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            if t.ndim == 2:
                t = t.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            out = self.model(t).cpu().numpy()        # (B, h, n_q)
            return out[0] if squeeze else out
