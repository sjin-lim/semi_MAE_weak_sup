# SegFormer-style lightweight MLP decoder for DINOv3 backbone.
#
# SegFormer (Xie et al., NeurIPS 2021) decoder 구조를 차용:
#   1. 각 레이어 feature → Linear proj → 공통 embed_dim
#   2. 모두 같은 해상도로 upsample 후 concat
#   3. MLP fusion → class logits
#
# LinearHead 대비 장점:
#   - 멀티스케일 feature 를 MLP 로 mixing → 더 풍부한 semantic 캡처
#   - LayerNorm 기반이라 small batch / few-shot 에서 안정적
#   - 파라미터 수: ~2M (ViT-L, 4-layer, embed_dim=256)
#
# 사용법:
#   build_segmentation_decoder(backbone, "FOUR_EVEN_INTERVALS", "segformer", ...)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """Linear → LayerNorm → GELU → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, H*W, C)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(self.act(self.norm(self.fc(x))))
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x


class SegFormerHead(nn.Module):
    """SegFormer-style All-MLP decoder.

    Multi-scale backbone features 를 받아서:
      1) 각 스케일을 Linear proj → 공통 embed_dim
      2) 가장 큰 feature map 해상도로 bilinear upsample
      3) Concat → MLP fusion → class logits

    Args:
        in_channels:  각 backbone layer 의 channel 수 리스트.
                      ex) ViT-L FOUR_EVEN_INTERVALS → [1024, 1024, 1024, 1024]
        embed_dim:    각 layer projection 후 차원. 256 권장.
        n_output_channels: 출력 클래스 수.
        dropout:      MLP dropout 비율.
    """

    def __init__(
        self,
        in_channels: list[int],
        n_output_channels: int,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # per-layer projection: channel → embed_dim
        self.linear_layers = nn.ModuleList([
            MLPBlock(ch, embed_dim, dropout=dropout)
            for ch in in_channels
        ])

        # fusion MLP: concat(4 × embed_dim) → embed_dim → num_classes
        fused_dim = embed_dim * len(in_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim, n_output_channels, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _fuse(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Project + upsample + concat + fuse."""
        target_size = inputs[0].shape[2:]  # 가장 큰 feature map 기준

        projected = []
        for feat, proj in zip(inputs, self.linear_layers):
            p = proj(feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            projected.append(p)

        fused = torch.cat(projected, dim=1)  # (B, embed_dim * n_layers, H, W)
        return self.fusion(fused)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Forward — training 시 사용.

        Args:
            inputs: backbone intermediate features [(B, C, H, W), ...]
        Returns:
            logits: (B, num_classes, H_feat, W_feat)
        """
        return self._fuse(inputs)

    def predict(self, inputs: list[torch.Tensor], rescale_to=(512, 512)) -> torch.Tensor:
        """Predict — evaluation 시 사용. 출력을 원본 해상도로 upsample.

        Args:
            inputs:     backbone intermediate features.
            rescale_to: 최종 출력 해상도 (H, W).
        Returns:
            logits: (B, num_classes, H_out, W_out)
        """
        logits = self._fuse(inputs)
        return F.interpolate(logits, size=rescale_to, mode="bilinear", align_corners=False)
