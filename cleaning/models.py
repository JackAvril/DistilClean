import math
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.ensemble import RandomForestClassifier
from torch import nn


class Scorer(nn.Sequential):
    def __init__(self, features, widths, dropout):
        layers = []
        for width in widths:
            layers.extend([nn.Linear(features, width), nn.ReLU(), nn.Dropout(dropout)])
            features = width
        super().__init__(*layers, nn.Linear(features, 1))

    def forward(self, values):
        return super().forward(values).squeeze(-1)


class Student:
    """Identical scorer, optimizer and weighted BCE for cell and preference tasks."""

    def __init__(self, features, settings, seed, device, *, mask_unseen=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.device = device
        self.model = Scorer(features, settings["hidden"], settings["dropout"]).to(device)
        weight_decay = settings["weight_decay"]
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("Invalid student weight decay")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=settings["learning_rate"], weight_decay=weight_decay)
        self.settings = settings
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.mean = None
        self.scale = None
        self.mask_unseen = mask_unseen or settings.get("mask_unseen", False)
        self.unscaled_from = settings.get("unscaled_from", features)
        if type(self.unscaled_from) is not int or not 0 <= self.unscaled_from <= features:
            raise ValueError("Invalid unscaled feature boundary")
        self.active = None

    def tensor(self, values):
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def standardized(self, values):
        normalized = (values - self.mean) / self.scale
        limit = self.settings["standardized_clip"]
        if limit is not None:
            normalized = normalized.clamp(-limit, limit)
        return normalized if self.active is None else normalized * self.active

    def fit(self, left, labels, weights, right=None):
        if not len(labels):
            return
        left, labels, weights = self.tensor(left), self.tensor(labels), self.tensor(weights)
        right = None if right is None else self.tensor(right)
        if self.mean is None or self.mask_unseen:
            inputs = left if right is None else torch.cat([left, right])
        if self.mask_unseen:
            # Each fit receives the cumulative training examples for that round.
            self.active = (inputs != inputs[0]).any(dim=0)
            # A categorical indicator that is always one was observed, even
            # though it has zero variance. Only never-seen (all-zero) categories
            # are ignored. Numeric constant-feature masking stays unchanged.
            self.active[self.unscaled_from:] = (inputs[:, self.unscaled_from:] != 0).any(dim=0)
        if self.mean is None:
            self.mean = inputs.mean(dim=0)
            self.scale = inputs.std(dim=0, unbiased=False).clamp_min(0.001)
            if self.active is not None:
                self.scale = torch.where(self.active, self.scale, 1.0)
            # Original preprocessing scales numeric columns, then concatenates
            # one-hot categorical indicators without rescaling those indicators.
            self.mean[self.unscaled_from:] = 0
            self.scale[self.unscaled_from:] = 1
        left = self.standardized(left)
        if right is not None:
            right = self.standardized(right)
        self.model.train()
        batch = self.settings["batch_size"]
        for _ in range(self.settings["epochs"]):
            order = self.rng.permutation(len(labels))
            for offset in range(0, len(order), batch):
                indices = torch.as_tensor(order[offset:offset + batch], device=self.device)
                logit = self.model(left[indices])
                if right is not None:
                    logit = logit - self.model(right[indices])
                loss = functional.binary_cross_entropy_with_logits(logit, labels[indices], reduction="none")
                loss = (loss * weights[indices]).sum() / weights[indices].sum().clamp_min(0.001)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.steps += 1

    def scores(self, values):
        self.model.eval()
        scores = []
        with torch.inference_mode():
            for offset in range(0, len(values), 4096):
                batch = self.tensor(values[offset:offset + 4096])
                batch = self.standardized(batch)
                scores.extend(self.model(batch).cpu().tolist())
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError("Student produced non-finite scores")
        return scores

    def save(self, path, feature_names):
        if (self.mean is None or len(feature_names) != self.mean.numel()
                or any(not isinstance(name, str) for name in feature_names)
                or len(set(feature_names)) != len(feature_names)):
            raise ValueError("Checkpoint feature names must match the fitted model input")
        payload = {"feature_names": feature_names, "state": self.model.state_dict(),
                   "mean": self.mean, "scale": self.scale, "steps": self.steps, "settings": self.settings}
        if self.active is not None:
            payload["active_features"] = self.active
        torch.save(payload, path)


def class_weights(labels, weights):
    counts = Counter(labels)
    return [weight * len(labels) / (len(counts) * counts[label]) for label, weight in zip(labels, weights)]


def verifier_fit(indices, labels, vectors, detector_scores, threshold, seed, settings):
    # Fit only on calibration cells rejected from student training.
    eligible = [i for i in indices if detector_scores[i] >= threshold]
    targets = [labels[i][0] for i in eligible]
    if len(targets) < settings["minimum_samples"] or len(set(targets)) < 2:
        return None
    features = [[*vectors[i], detector_scores[i]] for i in eligible]
    return RandomForestClassifier(
        n_estimators=settings["n_estimators"], max_depth=settings["max_depth"],
        class_weight=settings["class_weight"], random_state=seed, n_jobs=settings["n_jobs"],
    ).fit(features, targets)
