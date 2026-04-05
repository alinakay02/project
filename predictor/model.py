"""
predictor/model.py — Нейросетевая компонента GRU (параграф 3.2.2)

Классы:
  GRUQuantileNet — архитектура: 2 слоя GRU (64), dropout 0.10, 3 квантильных выхода
  GRUTrainer     — обучение: квантильная функция потерь (формула 3.13), Adam, ранняя остановка
"""

import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Квантильная функция потерь (формула 3.13)
# ══════════════════════════════════════════════════════════════════════════════

class QuantileLoss(nn.Module):
    """
    Pinball loss (квантильная функция потерь, формула 3.13):
    L_q(y, ŷ) = q * max(y - ŷ, 0) + (1-q) * max(ŷ - y, 0)
    Суммарная потеря — среднее по трём квантилям.
    """
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles  # [0.025, 0.5, 0.975]

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds  : (batch, n_quantiles)
        targets: (batch,)
        """
        total = torch.zeros(1, device=preds.device)
        for i, q in enumerate(self.quantiles):
            e = targets - preds[:, i]
            total += torch.mean(torch.max(q * e, (q - 1) * e))
        return total / len(self.quantiles)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Архитектура нейронной сети (таблица 4.2)
# ══════════════════════════════════════════════════════════════════════════════

class GRUQuantileNet(nn.Module):
    """
    Архитектура (таблица 4.2):
      Входной слой: d_in = 69
        - 60 значений нормализованного остатка R̃_t
        - 6 тригонометрических временных признаков (sin/cos час, день, минута)
        - 3 признака состава классов φ_t
      GRU слой 1: 64 нейрона
      Dropout: p = 0.10
      GRU слой 2: 64 нейрона
      Полносвязный выход: 3 квантиля (0.025, 0.5, 0.975)
    """
    def __init__(
        self,
        d_in: int = 69,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.10,
        n_quantiles: int = 3,
    ):
        super().__init__()
        self.d_in = d_in
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Два рекуррентных слоя GRU (с dropout между ними)
        self.gru1 = nn.GRU(
            input_size=d_in, hidden_size=hidden_dim,
            num_layers=1, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.gru2 = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=1, batch_first=True
        )
        # Выходной полносвязный слой → 3 квантиля
        self.fc = nn.Linear(hidden_dim, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len=60, d_in=69) — если d_in=1, то seq задаётся иначе
        Для нашей задачи: x имеет форму (batch, 1, d_in=69) — один шаг
        """
        out1, _ = self.gru1(x)           # (batch, seq, hidden)
        out1 = self.dropout(out1)
        out2, _ = self.gru2(out1)        # (batch, seq, hidden)
        last = out2[:, -1, :]            # берём последний временной шаг
        return self.fc(last)             # (batch, 3)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dataset для временного ряда
# ══════════════════════════════════════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    """
    Скользящее окно длиной w_input для обучения GRU.
    Каждый пример: (X, y), где
      X — окно из w_input временных шагов с признаками (d_in=69)
      y — следующее значение нормализованного остатка
    """
    def __init__(
        self,
        residual_norm: np.ndarray,   # R̃_t (нормализованный остаток)
        timestamps: np.ndarray,      # unix timestamps для временных признаков
        phi: np.ndarray,             # φ_t (3, n) — состав классов
        w_input: int = 60,
    ):
        self.residual = residual_norm.astype(np.float32)
        self.timestamps = timestamps
        self.phi = phi.astype(np.float32)  # (3, n)
        self.w = w_input
        self.n = len(residual_norm) - w_input

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window_r = self.residual[idx: idx + self.w]   # (w,)
        window_ts = self.timestamps[idx: idx + self.w]
        window_phi = self.phi[:, idx: idx + self.w].T  # (w, 3)

        # Тригонометрические временные признаки (6 значений на шаг)
        hours = (window_ts % 86400) / 3600.0       # час суток [0,24)
        dows = ((window_ts // 86400) % 7).astype(float)  # день недели [0,7)
        minutes = ((window_ts % 3600) / 60.0)       # минута часа [0,60)

        sin_h = np.sin(2 * np.pi * hours / 24).reshape(-1, 1)
        cos_h = np.cos(2 * np.pi * hours / 24).reshape(-1, 1)
        sin_d = np.sin(2 * np.pi * dows / 7).reshape(-1, 1)
        cos_d = np.cos(2 * np.pi * dows / 7).reshape(-1, 1)
        sin_m = np.sin(2 * np.pi * minutes / 60).reshape(-1, 1)
        cos_m = np.cos(2 * np.pi * minutes / 60).reshape(-1, 1)

        # Собираем вектор признаков: (w, 60+6+3 = 69)
        # Но архитектура принимает ОДИН шаг как весь вектор → (1, 69)
        # Здесь используем скользящее окно как одну «точку» с 60 историческими + 6 + 3
        residual_block = window_r.reshape(1, -1)        # (1, 60)
        time_feats = np.concatenate(
            [sin_h[-1:], cos_h[-1:], sin_d[-1:], cos_d[-1:], sin_m[-1:], cos_m[-1:]],
            axis=1
        )  # (1, 6) — временные признаки ПОСЛЕДНЕЙ точки окна
        phi_block = window_phi[-1:, :]  # (1, 3)

        X = np.concatenate([residual_block, time_feats, phi_block], axis=1)  # (1, 69)
        y = self.residual[idx + self.w]

        return torch.FloatTensor(X), torch.FloatTensor([y])


# ══════════════════════════════════════════════════════════════════════════════
# 4. Тренер модели
# ══════════════════════════════════════════════════════════════════════════════

class GRUTrainer:
    """
    Инкапсулирует процедуру обучения GRUQuantileNet (параграф 4.1).

    Гиперпараметры (таблица 4.2):
      lr=0.001, lr_decay=0.99, grad_clip=1.0,
      max_epochs=100, patience=10
    """
    def __init__(
        self,
        model: GRUQuantileNet,
        quantiles: List[float] = (0.025, 0.5, 0.975),
        lr: float = 0.001,
        lr_decay: float = 0.99,
        grad_clip: float = 1.0,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model = model
        self.quantiles = list(quantiles)
        self.lr = lr
        self.lr_decay = lr_decay
        self.grad_clip = grad_clip
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.criterion = QuantileLoss(self.quantiles)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=lr_decay
        )

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss: float = float("inf")
        self.best_state: Optional[dict] = None
        self.epochs_no_improve: int = 0
        self.stopped_epoch: int = 0

    def fit(
        self,
        train_dataset: TimeSeriesDataset,
        val_dataset: TimeSeriesDataset,
    ) -> None:
        """
        Обучение с ранней остановкой (patience=10 эпох).
        Восстанавливает лучшие веса по val_loss.
        """
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        for epoch in range(1, self.max_epochs + 1):
            # ── Обучение ──────────────────────────────────────────────────
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.squeeze(1).to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                # Обрезка нормы градиента (grad_clip=1.0)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss / len(train_loader)
            self.train_losses.append(train_loss)

            # ── Валидация ────────────────────────────────────────────────
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.squeeze(1).to(self.device)
                    preds = self.model(X_batch)
                    val_loss += self.criterion(preds, y_batch).item()
            val_loss /= max(len(val_loader), 1)
            self.val_losses.append(val_loss)

            # ── Ранняя остановка ──────────────────────────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.scheduler.step()

            logger.info(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | no_improve={self.epochs_no_improve}"
            )

            if self.epochs_no_improve >= self.patience:
                self.stopped_epoch = epoch
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        # Восстановление лучшего состояния
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Прогноз для одного входного вектора.
        X: (1, 69) → возвращает (3,) — три квантиля нормализованного остатка.
        """
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self.device)
            if t.ndim == 2:
                t = t.unsqueeze(0)  # (1, 1, 69)
            out = self.model(t)     # (1, 3)
            return out.cpu().numpy().flatten()
