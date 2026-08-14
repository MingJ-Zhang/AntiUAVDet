"""
Fog-aware components for AoDE-DEIM.

The modules in this file are deliberately detector-centric: they do not reconstruct
an RGB clear image. They estimate unresolved detection evidence, allocate bounded
feature/query update budgets, and synthesize counterfactual stronger fog views for
consistency training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-6


def spatial_minmax(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Per-sample, per-channel min-max normalization over spatial dimensions."""
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def image_grad_mag(x: torch.Tensor) -> torch.Tensor:
    """Cheap differentiable gradient magnitude for BCHW tensors."""
    dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + _EPS)


def bounded_curve(u: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Closed-form monotone bounded response curve.

    Args:
        u: response in [0, 1].
        gamma: modulation parameter strictly inside (-1, 1).
    """
    gamma = gamma.clamp(-0.98, 0.98)
    denom = 1.0 + gamma * (2.0 * u - 1.0)
    return ((1.0 + gamma) * u / (denom + eps)).clamp(0.0, 1.0)


class ConvNormAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, groups: int = 1, act: bool = True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, padding=p, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SpatialAoDERefiner(nn.Module):
    """Spatial Age-of-Detection-Evidence refinement for high-resolution features.

    The module maintains a fading-memory state for each selected feature level. A
    contextual Trust map and a non-increasing stage envelope produce a bounded
    spatial budget. The budget constrains a closed-form curve that gates a residual
    feature update.
    """

    def __init__(
        self,
        channels: int,
        num_levels: int = 2,
        iterations: int = 2,
        memory: float = 0.70,
        prior_weight: float = 0.15,
        residual_weight: float = 0.35,
        conflict_weight: float = 0.10,
        variation_weight: float = 0.15,
        trust_floor: float = 0.05,
        budget_min: float = 0.02,
        budget_max: Sequence[float] = (0.18, 0.10),
        residual_scale: float = 0.50,
    ) -> None:
        super().__init__()
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if len(budget_max) != iterations:
            raise ValueError("budget_max length must equal iterations")
        self.channels = channels
        self.num_levels = num_levels
        self.iterations = iterations
        self.memory = float(memory)
        self.prior_weight = float(prior_weight)
        self.residual_weight = float(residual_weight)
        self.conflict_weight = float(conflict_weight)
        self.variation_weight = float(variation_weight)
        self.trust_floor = float(trust_floor)
        self.budget_min = float(budget_min)
        self.register_buffer("budget_max", torch.tensor(list(budget_max), dtype=torch.float32), persistent=True)
        self.residual_scale = float(residual_scale)

        hidden = max(32, channels // 4)
        self.objectness = nn.ModuleList([
            nn.Sequential(
                ConvNormAct(channels, hidden, 3),
                nn.Conv2d(hidden, 1, 1),
            ) for _ in range(num_levels)
        ])
        self.trust_heads = nn.ModuleList([
            nn.Sequential(
                ConvNormAct(channels + 4, hidden, 3),
                ConvNormAct(hidden, hidden, 3, groups=hidden),
                nn.Conv2d(hidden, 1, 1),
            ) for _ in range(num_levels)
        ])
        self.gamma_heads = nn.ModuleList([
            nn.Sequential(
                ConvNormAct(channels + 4, hidden, 3),
                nn.Conv2d(hidden, 1, 1),
            ) for _ in range(num_levels)
        ])
        self.update_heads = nn.ModuleList([
            nn.Sequential(
                ConvNormAct(channels, channels, 3, groups=channels),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            ) for _ in range(num_levels)
        ])

        # Start close to identity so the detector can warm up stably.
        # Zero only the curve-parameter heads. The residual branches keep their
        # default initialization so gamma receives a non-zero learning signal at
        # the identity starting point.
        for module in self.gamma_heads:
            last_conv = [m for m in module.modules() if isinstance(m, nn.Conv2d)][-1]
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    @staticmethod
    def _cross_level_conflict(objectness: List[torch.Tensor], level: int) -> torch.Tensor:
        cur = objectness[level]
        refs = []
        for j, other in enumerate(objectness):
            if j == level:
                continue
            refs.append(F.interpolate(other, size=cur.shape[-2:], mode="bilinear", align_corners=False))
        if not refs:
            return torch.zeros_like(cur)
        ref = torch.stack(refs, dim=0).mean(0)
        return (cur - ref).abs()

    def forward(self, features: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
        if len(features) < self.num_levels:
            raise ValueError(f"expected at least {self.num_levels} feature levels, got {len(features)}")

        outs = list(features)
        selected = [outs[i] for i in range(self.num_levels)]
        objectness = [torch.sigmoid(self.objectness[i](f)) for i, f in enumerate(selected)]

        ages: List[torch.Tensor] = []
        priors: List[torch.Tensor] = []
        prev_feats = [f for f in selected]
        for i, f in enumerate(selected):
            mean_abs = f.abs().mean(1, keepdim=True)
            flatness = 1.0 - spatial_minmax(image_grad_mag(mean_abs))
            low_response = 1.0 - spatial_minmax(objectness[i])
            conflict = spatial_minmax(self._cross_level_conflict(objectness, i))
            prior = spatial_minmax(0.45 * flatness + 0.35 * low_response + 0.20 * conflict)
            priors.append(prior)
            ages.append(prior)

        age_history: List[torch.Tensor] = []
        budget_history: List[torch.Tensor] = []
        trust_history: List[torch.Tensor] = []

        for k in range(self.iterations):
            new_feats: List[torch.Tensor] = []
            new_ages: List[torch.Tensor] = []
            current_objectness = [torch.sigmoid(self.objectness[i](selected[i])) for i in range(self.num_levels)]

            for i, feat in enumerate(selected):
                obj = current_objectness[i]
                residual = 1.0 - spatial_minmax(obj)
                conflict = spatial_minmax(self._cross_level_conflict(current_objectness, i))
                variation = spatial_minmax((feat - prev_feats[i]).abs().mean(1, keepdim=True))

                base = (
                    self.memory * ages[i]
                    + self.prior_weight * priors[i]
                    + self.residual_weight * residual
                    + self.conflict_weight * conflict
                )
                normalizer = self.memory + self.prior_weight + self.residual_weight + self.conflict_weight + _EPS
                age = (base / normalizer - self.variation_weight * variation * ages[i]).clamp(0.0, 1.0)

                cue = torch.cat([feat, age, priors[i], residual, conflict], dim=1)
                trust = self.trust_floor + (1.0 - self.trust_floor) * torch.sigmoid(self.trust_heads[i](cue))
                priority = age * trust
                bmax = self.budget_max[k].to(dtype=feat.dtype, device=feat.device)
                budget = self.budget_min + (bmax - self.budget_min) * priority
                gamma = budget * torch.tanh(self.gamma_heads[i](cue))

                gate = obj.clamp(0.0, 1.0)
                curved = bounded_curve(gate, gamma)
                modulation = curved - gate
                update = self.update_heads[i](feat) * modulation
                updated = feat + self.residual_scale * update

                new_feats.append(updated)
                new_ages.append(age)
                prev_feats[i] = feat

                age_history.append(age)
                budget_history.append(budget)
                trust_history.append(trust)

            selected = new_feats
            ages = new_ages

        for i in range(self.num_levels):
            outs[i] = selected[i]

        # Histories are ordered by iteration then level. Final lists are convenient for losses/visualization.
        state = {
            "age_history": age_history,
            "budget_history": budget_history,
            "trust_history": trust_history,
            "age_final": ages,
            "budget_final": budget_history[-self.num_levels:],
            "trust_final": trust_history[-self.num_levels:],
            "prior": priors,
        }
        return outs, state


class CounterfactualFogGenerator(nn.Module):
    """Generate a stronger, spatially non-uniform fog view from any input image.

    This is intentionally one-way: the generated image is always at least as foggy
    as the input, so it is valid for already-foggy datasets and does not require a
    paired clear observation.
    """

    def __init__(
        self,
        beta_range: Tuple[float, float] = (0.035, 0.11),
        airlight_range: Tuple[float, float] = (0.68, 0.92),
        nonuniform_range: Tuple[float, float] = (0.10, 0.35),
        blur_kernel: int = 31,
    ) -> None:
        super().__init__()
        if beta_range[0] <= 0 or beta_range[1] < beta_range[0]:
            raise ValueError("invalid beta_range")
        if blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be odd")
        self.beta_range = tuple(float(v) for v in beta_range)
        self.airlight_range = tuple(float(v) for v in airlight_range)
        self.nonuniform_range = tuple(float(v) for v in nonuniform_range)
        self.blur_kernel = int(blur_kernel)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be BCHW RGB")
        b, _, h, w = images.shape
        dtype, device = images.dtype, images.device

        beta = torch.empty(b, 1, 1, 1, device=device, dtype=dtype).uniform_(*self.beta_range)
        airlight = torch.empty(b, 1, 1, 1, device=device, dtype=dtype).uniform_(*self.airlight_range)
        nonuniform = torch.empty(b, 1, 1, 1, device=device, dtype=dtype).uniform_(*self.nonuniform_range)

        # A perspective-like depth prior: upper image regions are usually farther away.
        y = torch.linspace(1.0, 0.15, h, device=device, dtype=dtype).view(1, 1, h, 1)
        depth = y.expand(b, 1, h, w)

        # Smooth random field creates spatially non-uniform fog without external depth.
        noise_h = max(2, h // 32)
        noise_w = max(2, w // 32)
        coarse = torch.rand(b, 1, noise_h, noise_w, device=device, dtype=dtype)
        field = F.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=False)
        field = spatial_minmax(field)
        depth = (depth * (1.0 + nonuniform * (field - 0.5))).clamp_min(0.05)

        transmission = torch.exp(-beta * depth)
        fogged = images * transmission + airlight * (1.0 - transmission)
        fogged = fogged.clamp(0.0, 1.0)

        strength = (1.0 - transmission).mean(dim=(-2, -1, -3))
        return fogged, strength, transmission


@torch.no_grad()
def estimate_visibility_map(
    images: torch.Tensor,
    window: int = 7,
    contrast_scale: float = 0.12,
    gradient_scale: float = 0.20,
) -> torch.Tensor:
    """Estimate local relative visibility with fixed cross-image calibration.

    Unlike per-image min-max normalization, fixed scales preserve the ordering
    between a source image and its stronger-fog counterfactual. The output is a
    training reliability prior rather than a physical transmission estimate.
    Inputs are expected in [0, 1].
    """
    if contrast_scale <= 0 or gradient_scale <= 0:
        raise ValueError('visibility calibration scales must be positive')
    luminance = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
    pad = window // 2
    padded = F.pad(luminance, (pad, pad, pad, pad), mode='reflect')
    mean = F.avg_pool2d(padded, window, stride=1, padding=0)
    mean_sq = F.avg_pool2d(padded.square(), window, stride=1, padding=0)
    contrast = torch.sqrt((mean_sq - mean.square()).clamp_min(0.0) + _EPS)
    grad = image_grad_mag(luminance)
    contrast_score = (contrast / float(contrast_scale)).clamp(0.0, 1.0)
    gradient_score = (grad / float(gradient_scale)).clamp(0.0, 1.0)
    visibility = 0.55 * contrast_score + 0.45 * gradient_score
    return visibility.clamp(0.02, 1.0)


@torch.no_grad()
def box_mean_values(maps: torch.Tensor, targets: Sequence[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
    """Mean map value inside normalized cxcywh target boxes."""
    b, _, h, w = maps.shape
    if len(targets) != b:
        raise ValueError("batch size and targets length mismatch")
    values: List[torch.Tensor] = []
    for i, target in enumerate(targets):
        boxes = target.get("boxes")
        if boxes is None or boxes.numel() == 0:
            values.append(torch.empty(0, device=maps.device, dtype=maps.dtype))
            continue
        per_box = []
        for box in boxes:
            cx, cy, bw, bh = box.tolist()
            x1 = max(0, min(w - 1, int(math.floor((cx - bw / 2.0) * w))))
            y1 = max(0, min(h - 1, int(math.floor((cy - bh / 2.0) * h))))
            x2 = max(x1 + 1, min(w, int(math.ceil((cx + bw / 2.0) * w))))
            y2 = max(y1 + 1, min(h, int(math.ceil((cy + bh / 2.0) * h))))
            per_box.append(maps[i, 0, y1:y2, x1:x2].mean())
        values.append(torch.stack(per_box))
    return values


def pool_state_in_boxes(
    state_maps: Sequence[torch.Tensor],
    targets: Sequence[Dict[str, torch.Tensor]],
) -> List[torch.Tensor]:
    """Differentiable box means over one or more spatial state maps.

    Maps from multiple levels are averaged after independent ROI pooling.
    """
    if not state_maps:
        device = targets[0]["boxes"].device
        return [torch.empty(0, device=device) for _ in targets]
    batch = state_maps[0].shape[0]
    outputs: List[List[torch.Tensor]] = [[] for _ in range(batch)]
    for fmap in state_maps:
        _, _, h, w = fmap.shape
        for bi, target in enumerate(targets):
            boxes = target.get("boxes")
            if boxes is None or boxes.numel() == 0:
                continue
            vals = []
            for box in boxes:
                cx, cy, bw, bh = box
                x1 = int(torch.floor((cx - bw / 2) * w).clamp(0, w - 1).item())
                y1 = int(torch.floor((cy - bh / 2) * h).clamp(0, h - 1).item())
                x2 = int(torch.ceil((cx + bw / 2) * w).clamp(x1 + 1, w).item())
                y2 = int(torch.ceil((cy + bh / 2) * h).clamp(y1 + 1, h).item())
                vals.append(fmap[bi, 0, y1:y2, x1:x2].mean())
            outputs[bi].append(torch.stack(vals))
    result: List[torch.Tensor] = []
    for bi, target in enumerate(targets):
        n = len(target.get("boxes", []))
        if n == 0:
            result.append(torch.empty(0, device=state_maps[0].device, dtype=state_maps[0].dtype))
        elif outputs[bi]:
            result.append(torch.stack(outputs[bi], dim=0).mean(0))
        else:
            result.append(torch.zeros(n, device=state_maps[0].device, dtype=state_maps[0].dtype))
    return result
