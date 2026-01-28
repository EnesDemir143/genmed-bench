"""
PipMix: Patch-in-Patch Mix Augmentation for Domain Generalization

Görüntüyü patch'lere ayırarak ve farklı domain'lerden gelen patch'leri
karıştırarak domain-robust özellikler öğrenmeyi teşvik eder.

Özellikler:
1. Patch-level mixing: Görüntünün belirli bölgelerinde mixing
2. Cross-domain patch exchange: Farklı domain'lerden patch değişimi
3. Multi-scale support: Farklı patch boyutlarında çalışabilme
"""

import random
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PipMix(nn.Module):
    """
    Patch-in-Patch Mix augmentation.
    
    Görüntüyü patch'lere ayırarak ve farklı domain görüntülerinin
    patch'leri ile karıştırarak domain-robust özellikler öğrenmeyi teşvik eder.
    
    Args:
        patch_size: Her patch'in boyutu (tuple veya int)
        mix_prob: Mixing olasılığı
        alpha: Mixup için beta dağılım parametresi
        num_patches_to_mix: Karıştırılacak patch sayısı (None ise random)
        
    Example:
        >>> pipmix = PipMix(patch_size=32, mix_prob=0.5)
        >>> augmented_img, lam = pipmix(anchor_img, cross_domain_img)
    """
    
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 32,
        mix_prob: float = 0.5,
        alpha: float = 0.4,
        num_patches_to_mix: Optional[int] = None
    ):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.mix_prob = mix_prob
        self.alpha = alpha
        self.num_patches_to_mix = num_patches_to_mix
    
    def _sample_lambda(self) -> float:
        """Beta dağılımından lambda değeri örnekle."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        return lam
    
    def _extract_patches(
        self,
        img: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Görüntüyü patch'lere ayır.
        
        Args:
            img: Input görüntü [C, H, W]
            
        Returns:
            Tuple of (patches, num_patches_h, num_patches_w)
            patches shape: [num_patches, C, patch_h, patch_w]
        """
        c, h, w = img.shape
        ph, pw = self.patch_size
        
        # Patch sayısını hesapla
        nh = h // ph
        nw = w // pw
        
        # Görüntüyü kırp (tam patch sayısına uyacak şekilde)
        img_cropped = img[:, :nh*ph, :nw*pw]
        
        # Unfold ile patch'lere ayır
        patches = img_cropped.unfold(1, ph, ph).unfold(2, pw, pw)
        # Shape: [C, nh, nw, ph, pw]
        
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        # Shape: [nh, nw, C, ph, pw]
        
        patches = patches.view(-1, c, ph, pw)
        # Shape: [nh*nw, C, ph, pw]
        
        return patches, nh, nw
    
    def _reconstruct_from_patches(
        self,
        patches: torch.Tensor,
        num_h: int,
        num_w: int,
        original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Patch'lerden görüntüyü yeniden oluştur.
        
        Args:
            patches: Patches [num_patches, C, patch_h, patch_w]
            num_h: Dikey patch sayısı
            num_w: Yatay patch sayısı
            original_size: Orijinal görüntü boyutu (H, W)
            
        Returns:
            Reconstructed image [C, H, W]
        """
        n, c, ph, pw = patches.shape
        
        # Reshape to grid
        patches = patches.view(num_h, num_w, c, ph, pw)
        
        # Combine patches
        patches = patches.permute(2, 0, 3, 1, 4).contiguous()
        # Shape: [C, nh, ph, nw, pw]
        
        img = patches.view(c, num_h * ph, num_w * pw)
        
        # Pad to original size if needed
        h, w = original_size
        if img.shape[1] < h or img.shape[2] < w:
            pad_h = h - img.shape[1]
            pad_w = w - img.shape[2]
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        
        return img[:, :h, :w]
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        PipMix augmentation uygula.
        
        Anchor görüntünün bazı patch'lerini cross-domain görüntünün
        patch'leri ile değiştirir.
        
        Args:
            img1: Anchor görüntü [C, H, W]
            img2: Cross-domain görüntü [C, H, W]
            
        Returns:
            Tuple of:
                - mixed_img: Karıştırılmış görüntü
                - lam: Anchor görüntünün oranı (1 - değiştirilen patch oranı)
        """
        # Probability check
        if random.random() > self.mix_prob:
            return img1, 1.0
        
        c, h, w = img1.shape
        
        # Her iki görüntüyü de patch'lere ayır
        patches1, nh, nw = self._extract_patches(img1)
        patches2, _, _ = self._extract_patches(img2)
        
        total_patches = nh * nw
        
        # Karıştırılacak patch sayısını belirle
        if self.num_patches_to_mix is None:
            num_to_mix = random.randint(1, max(1, total_patches // 2))
        else:
            num_to_mix = min(self.num_patches_to_mix, total_patches)
        
        # Rastgele patch indeksleri seç
        selected_indices = random.sample(range(total_patches), num_to_mix)
        
        # Lambda örnekle (patch içi mixing için)
        lam = self._sample_lambda()
        
        # Seçilen patch'leri karıştır
        mixed_patches = patches1.clone()
        for idx in selected_indices:
            # Patch-level mixup
            mixed_patches[idx] = lam * patches1[idx] + (1 - lam) * patches2[idx]
        
        # Görüntüyü yeniden oluştur
        result = self._reconstruct_from_patches(mixed_patches, nh, nw, (h, w))
        
        # Gerçek lambda: anchor patch'lerin ağırlığı
        # (karıştırılmayan + karıştırılan * lam) / toplam
        actual_lam = ((total_patches - num_to_mix) + num_to_mix * lam) / total_patches
        
        return result, actual_lam


class PipMixBatch(nn.Module):
    """
    Batch-level PipMix uygulaması.
    
    Bir batch içindeki görüntülerin patch'lerini cross-domain
    görüntülerle karıştırır.
    
    Args:
        patch_size: Patch boyutu
        mix_prob: Her örnek için mixing olasılığı
        alpha: Beta dağılım parametresi
        num_patches_to_mix: Karıştırılacak patch sayısı
        
    Example:
        >>> pipmix_batch = PipMixBatch(patch_size=32)
        >>> mixed_images, mixed_labels, lam_values = pipmix_batch(
        ...     images, labels, domains
        ... )
    """
    
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 32,
        mix_prob: float = 0.5,
        alpha: float = 0.4,
        num_patches_to_mix: Optional[int] = None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mix_prob = mix_prob
        self.alpha = alpha
        self.num_patches_to_mix = num_patches_to_mix
        
        self.pipmix = PipMix(
            patch_size=patch_size,
            mix_prob=1.0,  # Probability handled at batch level
            alpha=alpha,
            num_patches_to_mix=num_patches_to_mix
        )
    
    def _get_cross_domain_partner(
        self,
        idx: int,
        domains: torch.Tensor
    ) -> int:
        """Farklı domain'den partner bul."""
        current_domain = domains[idx].item()
        batch_size = domains.size(0)
        
        # Farklı domain'deki örnekleri bul
        cross_domain_mask = domains != current_domain
        cross_domain_indices = torch.where(cross_domain_mask)[0]
        
        if len(cross_domain_indices) > 0:
            rand_idx = torch.randint(len(cross_domain_indices), (1,))
            return cross_domain_indices[rand_idx].item()
        else:
            # Aynı domain'den farklı bir örnek seç
            other_indices = [i for i in range(batch_size) if i != idx]
            if other_indices:
                return random.choice(other_indices)
            return idx
    
    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        domains: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch'e PipMix uygula.
        
        Args:
            images: Batch görüntüleri [B, C, H, W]
            labels: Batch label'ları [B] veya [B, num_classes]
            domains: Domain ID'leri [B]
            
        Returns:
            Tuple of:
                - mixed_images: Karıştırılmış görüntüler [B, C, H, W]
                - mixed_labels: Karıştırılmış label'lar
                - lam_values: Her örnek için lambda değerleri [B]
        """
        batch_size = images.shape[0]
        device = images.device
        
        mixed_images = []
        lam_values = []
        partner_indices = []
        
        for i in range(batch_size):
            # Probability check
            if random.random() > self.mix_prob:
                mixed_images.append(images[i])
                lam_values.append(1.0)
                partner_indices.append(i)
                continue
            
            # Cross-domain partner bul
            partner_idx = self._get_cross_domain_partner(i, domains)
            partner_indices.append(partner_idx)
            
            # PipMix uygula
            mixed_img, lam = self.pipmix(images[i], images[partner_idx])
            mixed_images.append(mixed_img)
            lam_values.append(lam)
        
        # Stack results
        mixed_images = torch.stack(mixed_images)
        lam_values = torch.tensor(lam_values, device=device)
        partner_indices = torch.tensor(partner_indices, device=device)
        
        # Label mixing
        partner_labels = labels[partner_indices]
        
        if labels.dim() == 1:
            # Soft label oluştur: [B, 2] - [original_weight, partner_weight]
            mixed_labels = torch.stack([lam_values, 1 - lam_values], dim=1)
            # Ayrıca original ve partner class index'leri için
            label_info = {
                'soft_labels': mixed_labels,
                'original_labels': labels,
                'partner_labels': partner_labels,
                'lam': lam_values
            }
            return mixed_images, label_info, lam_values
        else:
            # One-hot veya soft label durumu
            lam_expanded = lam_values.view(-1, 1)
            mixed_labels = lam_expanded * labels + (1 - lam_expanded) * partner_labels
            return mixed_images, mixed_labels, lam_values


# Convenience function
def pipmix_transform(
    img1: torch.Tensor,
    img2: torch.Tensor,
    patch_size: int = 32,
    alpha: float = 0.4
) -> Tuple[torch.Tensor, float]:
    """
    Basit PipMix transform fonksiyonu.
    
    Args:
        img1: Anchor görüntü [C, H, W]
        img2: Cross-domain görüntü [C, H, W]
        patch_size: Patch boyutu
        alpha: Mixing alpha
        
    Returns:
        Tuple of (mixed_image, lambda)
    """
    pipmix = PipMix(
        patch_size=patch_size,
        mix_prob=1.0,
        alpha=alpha
    )
    return pipmix(img1, img2)
