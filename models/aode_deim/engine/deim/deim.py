"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

AoDE-DEIM additions:
- detector-centric spatial Age-of-Detection-Evidence state propagation;
- counterfactual stronger-fog consistency training without paired clear images;
- local visibility priors for dense O2O matching and denoising.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..core import register
from .fog_modules import CounterfactualFogGenerator, box_mean_values, estimate_visibility_map


__all__ = ['DEIM']


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        use_visibility_prior: bool = False,
        use_counterfactual_fog: bool = False,
        cfc_probability: float = 0.75,
        cfc_start_epoch: int = 4,
        cfc_stop_epoch: int = 10_000,
        cfc_beta_range: Sequence[float] = (0.035, 0.11),
        cfc_airlight_range: Sequence[float] = (0.68, 0.92),
        cfc_nonuniform_range: Sequence[float] = (0.10, 0.35),
        force_decoder_fp32: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.use_visibility_prior = bool(use_visibility_prior)
        self.use_counterfactual_fog = bool(use_counterfactual_fog)
        self.cfc_probability = float(cfc_probability)
        self.cfc_start_epoch = int(cfc_start_epoch)
        self.cfc_stop_epoch = int(cfc_stop_epoch)
        self.force_decoder_fp32 = bool(force_decoder_fp32)

        self.fog_generator = CounterfactualFogGenerator(
            beta_range=tuple(cfc_beta_range),
            airlight_range=tuple(cfc_airlight_range),
            nonuniform_range=tuple(cfc_nonuniform_range),
        ) if self.use_counterfactual_fog else None

    @staticmethod
    def _targets_with_visibility(
        targets: Optional[Sequence[Dict[str, torch.Tensor]]],
        visibility: Optional[Sequence[torch.Tensor]],
    ) -> Optional[List[Dict[str, torch.Tensor]]]:
        if targets is None:
            return None
        copied = [dict(t) for t in targets]
        if visibility is not None:
            for target, vis in zip(copied, visibility):
                target['visibility'] = vis
        return copied

    @staticmethod
    def _attach_visibility_to_outputs(outputs: Dict, visibility: Optional[Sequence[torch.Tensor]]) -> None:
        if visibility is None:
            return
        outputs['target_visibility'] = visibility
        for key in ('aux_outputs', 'enc_aux_outputs', 'dn_outputs'):
            for item in outputs.get(key, []):
                item['target_visibility'] = visibility
        for key in ('pre_outputs', 'dn_pre_outputs'):
            if key in outputs:
                outputs[key]['target_visibility'] = visibility

    def _forward_once(
        self,
        images: torch.Tensor,
        targets: Optional[Sequence[Dict[str, torch.Tensor]]],
        target_visibility: Optional[Sequence[torch.Tensor]],
    ) -> Dict:
        backbone_features = self.backbone(images)
        encoder_out = self.encoder(backbone_features)
        if isinstance(encoder_out, tuple):
            encoded_features, spatial_state = encoder_out
        else:
            encoded_features, spatial_state = encoder_out, {}

        decoder_targets = self._targets_with_visibility(targets, target_visibility)
        if self.force_decoder_fp32 and images.is_cuda:
            # Keep box-distribution refinement and AoDE query updates out of
            # float16. A rare fp16 overflow here can create non-finite cxcywh
            # predictions before the float32 criterion/matcher gets control.
            # Casting only the decoder inputs preserves AMP memory savings in
            # the backbone and encoder, which is required by the high-resolution
            # single-GPU CFC profile.
            with torch.autocast(device_type='cuda', enabled=False):
                decoder_features = [feature.float() for feature in encoded_features]
                outputs = self.decoder(decoder_features, decoder_targets)
        else:
            outputs = self.decoder(encoded_features, decoder_targets)
        if self.training and spatial_state:
            outputs['spatial_state'] = spatial_state
        self._attach_visibility_to_outputs(outputs, target_visibility)
        return outputs

    @staticmethod
    def _deterministic_cfc_gate(probability: float, epoch: int, global_step: Optional[int]) -> bool:
        """Deterministic Bernoulli gate shared by all ranks.

        The decision depends only on epoch/global_step, so DDP workers always execute
        the same branch. It is also reproducible across resumed runs.
        """
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        step = int(global_step) if global_step is not None else int(epoch)
        # A small integer hash mapped to [0, 1). No process-local RNG is used.
        value = (step * 1103515245 + int(epoch) * 12345 + 1013904223) & 0x7FFFFFFF
        return (value / float(0x80000000)) < probability

    def forward(self, x: torch.Tensor, targets=None, epoch: int = 0, global_step: Optional[int] = None):
        target_visibility = None
        if self.training and targets is not None and self.use_visibility_prior:
            visibility_map = estimate_visibility_map(x)
            target_visibility = box_mean_values(visibility_map, targets)

        outputs = self._forward_once(x, targets, target_visibility)

        use_cfc = (
            self.training
            and targets is not None
            and self.fog_generator is not None
            and self.cfc_start_epoch <= int(epoch) < self.cfc_stop_epoch
            and self._deterministic_cfc_gate(self.cfc_probability, int(epoch), global_step)
        )
        if use_cfc:
            fogged, fog_strength, _ = self.fog_generator(x)
            cfc_visibility = None
            if self.use_visibility_prior:
                cfc_visibility = box_mean_values(estimate_visibility_map(fogged), targets)
            cfc_outputs = self._forward_once(fogged, targets, cfc_visibility)
            cfc_outputs['fog_strength'] = fog_strength
            outputs['cfc_outputs'] = cfc_outputs

        return outputs

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
