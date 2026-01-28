"""
PipMix: Patch-in-Patch Mix Augmentation for Domain Generalization

PipMix, PixMix ve patch-based mixing tekniklerini birleştiren bir
augmentation stratejisidir. Domain generalization için tasarlanmıştır.

Özellikler:
1. Fractal/Structure mixing: Yapısal karmaşıklık ekleme
2. Patch-level mixing: Görüntünün belirli bölgelerinde mixing
3. Multi-scale augmentation: Farklı ölçeklerde augmentation
4. Progressive mixing: Kademeli karıştırma

Referanslar:
- PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures (CVPR 2022)
- Patch-based data augmentation techniques
"""

import random
from typing import Tuple, Optional, List, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms


class PipMix(nn.Module):
    """
    Patch-in-Patch Mix augmentation.
    
    Görüntüyü patch'lere ayırarak ve farklı domain/fractal pattern'ları
    ile karıştırarak domain-robust özellikler öğrenmeyi teşvik eder.
    
    Args:
        patch_size: Her patch'in boyutu (tuple veya int)
        mix_prob: Mixing olasılığı
        alpha: Mixup için beta dağılım parametresi
        num_patches_to_mix: Karıştırılacak patch sayısı (None ise random)
        use_fractal: Fractal pattern'lar kullanılsın mı
        use_noise: Noise pattern'lar kullanılsın mı
        
    Example:
        >>> pipmix = PipMix(patch_size=32, mix_prob=0.5)
        >>> augmented_img = pipmix(image, mixing_image)
    """
    
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 32,
        mix_prob: float = 0.5,
        alpha: float = 0.4,
        num_patches_to_mix: Optional[int] = None,
        use_fractal: bool = True,
        use_noise: bool = True
    ):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.mix_prob = mix_prob
        self.alpha = alpha
        self.num_patches_to_mix = num_patches_to_mix
        self.use_fractal = use_fractal
        self.use_noise = use_noise
        
        # Mixing operations
        self.mixing_ops = [
            self._additive_mix,
            self._multiplicative_mix,
            self._soft_light_mix,
        ]
    
    def _generate_fractal(
        self, 
        height: int, 
        width: int,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Diamond-Square algoritması ile fractal pattern oluştur.
        
        Args:
            height: Görüntü yüksekliği
            width: Görüntü genişliği
            device: Tensor device
            
        Returns:
            Fractal pattern [1, H, W]
        """
        # Basit Perlin-like noise
        size = max(height, width)
        scale = random.choice([2, 4, 8, 16])
        
        # Low frequency noise oluştur
        low_freq = torch.randn(1, size // scale, size // scale, device=device)
        
        # Upscale
        fractal = F.interpolate(
            low_freq.unsqueeze(0),
            size=(height, width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Normalize to [0, 1]
        fractal = (fractal - fractal.min()) / (fractal.max() - fractal.min() + 1e-8)
        
        return fractal
    
    def _generate_noise_pattern(
        self,
        height: int,
        width: int,
        noise_type: str = 'gaussian',
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Çeşitli noise pattern'lar oluştur.
        
        Args:
            height: Görüntü yüksekliği
            width: Görüntü genişliği
            noise_type: Noise tipi ('gaussian', 'uniform', 'salt_pepper', 'perlin')
            device: Tensor device
            
        Returns:
            Noise pattern [1, H, W]
        """
        if noise_type == 'gaussian':
            noise = torch.randn(1, height, width, device=device) * 0.3 + 0.5
        elif noise_type == 'uniform':
            noise = torch.rand(1, height, width, device=device)
        elif noise_type == 'salt_pepper':
            noise = torch.rand(1, height, width, device=device)
            noise = (noise > 0.9).float() + (noise < 0.1).float() * 0.5
        elif noise_type == 'perlin':
            noise = self._generate_fractal(height, width, device)
        else:
            noise = torch.rand(1, height, width, device=device)
        
        return noise.clamp(0, 1)
    
    def _additive_mix(
        self,
        img: torch.Tensor,
        pattern: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """Additive mixing: img + lam * pattern"""
        return (img + lam * pattern).clamp(0, 1)
    
    def _multiplicative_mix(
        self,
        img: torch.Tensor,
        pattern: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """Multiplicative mixing: img * (1 + lam * (pattern - 0.5))"""
        factor = 1 + lam * (pattern - 0.5)
        return (img * factor).clamp(0, 1)
    
    def _soft_light_mix(
        self,
        img: torch.Tensor,
        pattern: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """Soft light blending mode"""
        # Soft light formula
        result = torch.where(
            pattern <= 0.5,
            img - (1 - 2 * pattern) * img * (1 - img),
            img + (2 * pattern - 1) * (self._d(img) - img)
        )
        # Blend with original
        return (lam * result + (1 - lam) * img).clamp(0, 1)
    
    def _d(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function for soft light"""
        return torch.where(
            x <= 0.25,
            ((16 * x - 12) * x + 4) * x,
            torch.sqrt(x)
        )
    
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
    
    def _mix_selected_patches(
        self,
        patches: torch.Tensor,
        mix_pattern: torch.Tensor,
        num_patches: int
    ) -> torch.Tensor:
        """
        Seçilen patch'leri pattern ile karıştır.
        
        Args:
            patches: Input patches [N, C, ph, pw]
            mix_pattern: Mixing pattern [C, ph, pw]
            num_patches: Karıştırılacak patch sayısı
            
        Returns:
            Mixed patches [N, C, ph, pw]
        """
        n = patches.shape[0]
        
        # Rastgele patch seç
        selected_indices = random.sample(range(n), min(num_patches, n))
        
        # Lambda örnekle
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 0.5
        
        # Mixing operation seç
        mix_op = random.choice(self.mixing_ops)
        
        # Seçilen patch'leri karıştır
        mixed_patches = patches.clone()
        for idx in selected_indices:
            mixed_patches[idx] = mix_op(patches[idx], mix_pattern, lam)
        
        return mixed_patches
    
    def forward(
        self,
        img: torch.Tensor,
        mixing_source: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        PipMix augmentation uygula.
        
        Args:
            img: Input görüntü [C, H, W] veya [B, C, H, W]
            mixing_source: Opsiyonel karıştırma kaynağı [C, H, W]
            
        Returns:
            Augmented görüntü
        """
        # Batch dimension check
        has_batch = img.dim() == 4
        if has_batch:
            batch_size = img.shape[0]
            results = []
            for i in range(batch_size):
                results.append(self._forward_single(img[i], mixing_source))
            return torch.stack(results)
        else:
            return self._forward_single(img, mixing_source)
    
    def _forward_single(
        self,
        img: torch.Tensor,
        mixing_source: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Tek görüntü için PipMix uygula."""
        # Probability check
        if random.random() > self.mix_prob:
            return img
        
        c, h, w = img.shape
        device = img.device
        
        # Patch'lere ayır
        patches, nh, nw = self._extract_patches(img)
        
        # Karıştırılacak patch sayısı
        total_patches = nh * nw
        if self.num_patches_to_mix is None:
            num_to_mix = random.randint(1, max(1, total_patches // 2))
        else:
            num_to_mix = min(self.num_patches_to_mix, total_patches)
        
        # Karıştırma pattern'ı oluştur
        if mixing_source is not None:
            # Cross-domain mixing source kullan
            mix_pattern = F.interpolate(
                mixing_source.unsqueeze(0),
                size=self.patch_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        elif self.use_fractal and random.random() < 0.5:
            # Fractal pattern
            fractal = self._generate_fractal(self.patch_size[0], self.patch_size[1], device)
            mix_pattern = fractal.expand(c, -1, -1)
        elif self.use_noise:
            # Noise pattern
            noise_type = random.choice(['gaussian', 'uniform', 'perlin'])
            noise = self._generate_noise_pattern(
                self.patch_size[0], self.patch_size[1], noise_type, device
            )
            mix_pattern = noise.expand(c, -1, -1)
        else:
            # Skip mixing
            return img
        
        # Patch'leri karıştır
        mixed_patches = self._mix_selected_patches(patches, mix_pattern, num_to_mix)
        
        # Görüntüyü yeniden oluştur
        result = self._reconstruct_from_patches(mixed_patches, nh, nw, (h, w))
        
        return result


class PipMixBatch(nn.Module):
    """
    Batch-level PipMix uygulaması.
    
    Bir batch içindeki görüntülerin patch'lerini karıştırır.
    Cross-domain patch mixing de destekler.
    
    Args:
        patch_size: Patch boyutu
        mix_prob: Mixing olasılığı
        alpha: Beta dağılım parametresi
        cross_domain_prob: Cross-domain mixing olasılığı
        
    Example:
        >>> pipmix_batch = PipMixBatch(patch_size=32)
        >>> augmented = pipmix_batch(images, domains)
    """
    
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 32,
        mix_prob: float = 0.5,
        alpha: float = 0.4,
        cross_domain_prob: float = 0.5
    ):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.mix_prob = mix_prob
        self.alpha = alpha
        self.cross_domain_prob = cross_domain_prob
        
        self.pipmix = PipMix(
            patch_size=patch_size,
            mix_prob=1.0,  # Probability handled at batch level
            alpha=alpha,
            use_fractal=True,
            use_noise=True
        )
    
    def _get_cross_domain_partner(
        self,
        idx: int,
        domains: torch.Tensor
    ) -> Optional[int]:
        """Farklı domain'den partner bul."""
        current_domain = domains[idx].item()
        
        # Farklı domain'deki örnekleri bul
        cross_domain_mask = domains != current_domain
        cross_domain_indices = torch.where(cross_domain_mask)[0]
        
        if len(cross_domain_indices) > 0:
            rand_idx = torch.randint(len(cross_domain_indices), (1,))
            return cross_domain_indices[rand_idx].item()
        return None
    
    def forward(
        self,
        images: torch.Tensor,
        domains: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Batch'e PipMix uygula.
        
        Args:
            images: Batch görüntüleri [B, C, H, W]
            domains: Opsiyonel domain ID'leri [B]
            
        Returns:
            Augmented görüntüler [B, C, H, W]
        """
        batch_size = images.shape[0]
        results = []
        
        for i in range(batch_size):
            # Probability check
            if random.random() > self.mix_prob:
                results.append(images[i])
                continue
            
            mixing_source = None
            
            # Cross-domain mixing
            if domains is not None and random.random() < self.cross_domain_prob:
                partner_idx = self._get_cross_domain_partner(i, domains)
                if partner_idx is not None:
                    mixing_source = images[partner_idx]
            
            # PipMix uygula
            augmented = self.pipmix._forward_single(images[i], mixing_source)
            results.append(augmented)
        
        return torch.stack(results)


class PixMix(nn.Module):
    """
    PixMix augmentation (CVPR 2022).
    
    Fractal ve feature visualization pattern'ları kullanarak
    model robustness'ını artıran augmentation tekniği.
    
    Args:
        mixing_set_path: Mixing görüntülerinin path'i (opsiyonel)
        k: Augmentation zinciri uzunluğu
        beta: Mixing oranı için beta dağılım parametresi
        augmentations: Uygulanacak augmentation listesi
        
    Reference:
        Hendrycks et al., "PixMix: Dreamlike Pictures Comprehensively
        Improve Safety Measures", CVPR 2022
    """
    
    def __init__(
        self,
        k: int = 4,
        beta: float = 3.0,
        use_fractal: bool = True,
        use_noise: bool = True,
        severity: int = 3
    ):
        super().__init__()
        self.k = k
        self.beta = beta
        self.use_fractal = use_fractal
        self.use_noise = use_noise
        self.severity = severity
        
        # Augmentation operations
        self.augmentations = self._build_augmentations()
        
        # Mixing operations
        self.mixing_ops = ['add', 'multiply']
    
    def _build_augmentations(self) -> List[Callable]:
        """Augmentation listesi oluştur."""
        augmentations = [
            # Geometric
            lambda x: transforms.functional.rotate(x, random.uniform(-15, 15)),
            lambda x: transforms.functional.affine(
                x, angle=0,
                translate=(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)),
                scale=1.0, shear=0
            ),
            
            # Intensity
            lambda x: transforms.functional.adjust_brightness(
                x, 1 + random.uniform(-0.3, 0.3)
            ),
            lambda x: transforms.functional.adjust_contrast(
                x, 1 + random.uniform(-0.3, 0.3)
            ),
            
            # Blur
            lambda x: transforms.functional.gaussian_blur(
                x, kernel_size=random.choice([3, 5, 7])
            ),
        ]
        return augmentations
    
    def _generate_mixing_pic(
        self,
        height: int,
        width: int,
        channels: int,
        device: torch.device
    ) -> torch.Tensor:
        """Fractal veya noise pattern oluştur."""
        if self.use_fractal and random.random() < 0.5:
            # Multi-scale fractal
            pattern = torch.zeros(channels, height, width, device=device)
            
            for scale in [2, 4, 8, 16]:
                if scale > min(height, width):
                    continue
                    
                noise = torch.randn(channels, height // scale, width // scale, device=device)
                upscaled = F.interpolate(
                    noise.unsqueeze(0),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                pattern = pattern + upscaled / scale
            
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
            
        else:
            # Random noise
            pattern = torch.rand(channels, height, width, device=device)
            
            # Apply some structure
            if random.random() < 0.5:
                # Gaussian blur for smoother patterns
                pattern = transforms.functional.gaussian_blur(
                    pattern.unsqueeze(0),
                    kernel_size=random.choice([11, 21, 31])
                ).squeeze(0)
        
        return pattern
    
    def _mixup(
        self,
        img: torch.Tensor,
        mixing_pic: torch.Tensor,
        op: str
    ) -> torch.Tensor:
        """Görüntüyü mixing picture ile karıştır."""
        # Sample mixing weight
        m = np.random.beta(self.beta, self.beta)
        
        if op == 'add':
            mixed = (1 - m) * img + m * mixing_pic
        elif op == 'multiply':
            mixed = img * (1 + m * (mixing_pic - 0.5))
        else:
            mixed = img
        
        return mixed.clamp(0, 1)
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        PixMix uygula.
        
        Args:
            img: Input görüntü [C, H, W]
            
        Returns:
            Augmented görüntü [C, H, W]
        """
        c, h, w = img.shape
        device = img.device
        
        # Initialize output
        output = img.clone()
        
        # Augmentation chain
        for _ in range(self.k):
            # Randomly choose: augment or mix
            if random.random() < 0.5:
                # Apply random augmentation
                aug = random.choice(self.augmentations)
                try:
                    output = aug(output)
                except:
                    pass  # Skip if augmentation fails
            else:
                # Mix with fractal/noise pattern
                mixing_pic = self._generate_mixing_pic(h, w, c, device)
                op = random.choice(self.mixing_ops)
                output = self._mixup(output, mixing_pic, op)
        
        return output


class ProgressivePipMix(nn.Module):
    """
    Progressive Patch-in-Patch Mix.
    
    Eğitim süresince kademeli olarak artan karmaşıklıkta
    patch mixing uygular. Curriculum learning yaklaşımı.
    
    Args:
        patch_size_schedule: Epoch'a göre patch boyutu schedule'ı
        mix_prob_schedule: Epoch'a göre mixing olasılığı schedule'ı
        
    Example:
        >>> progressive = ProgressivePipMix()
        >>> progressive.set_epoch(10)
        >>> augmented = progressive(image)
    """
    
    def __init__(
        self,
        initial_patch_size: int = 64,
        final_patch_size: int = 16,
        initial_mix_prob: float = 0.2,
        final_mix_prob: float = 0.8,
        warmup_epochs: int = 10,
        total_epochs: int = 100
    ):
        super().__init__()
        self.initial_patch_size = initial_patch_size
        self.final_patch_size = final_patch_size
        self.initial_mix_prob = initial_mix_prob
        self.final_mix_prob = final_mix_prob
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
        self.current_epoch = 0
        self._update_pipmix()
    
    def _get_current_params(self) -> Tuple[int, float]:
        """Mevcut epoch için parametreleri hesapla."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            ratio = self.current_epoch / self.warmup_epochs
            patch_size = int(
                self.initial_patch_size - 
                ratio * (self.initial_patch_size - self.final_patch_size)
            )
            mix_prob = (
                self.initial_mix_prob + 
                ratio * (self.final_mix_prob - self.initial_mix_prob)
            )
        else:
            # Full augmentation
            patch_size = self.final_patch_size
            mix_prob = self.final_mix_prob
        
        return patch_size, mix_prob
    
    def _update_pipmix(self):
        """PipMix instance'ını güncelle."""
        patch_size, mix_prob = self._get_current_params()
        self.pipmix = PipMix(
            patch_size=patch_size,
            mix_prob=mix_prob,
            alpha=0.4,
            use_fractal=True,
            use_noise=True
        )
    
    def set_epoch(self, epoch: int):
        """Epoch'u ayarla ve parametreleri güncelle."""
        self.current_epoch = epoch
        self._update_pipmix()
    
    def forward(
        self,
        img: torch.Tensor,
        mixing_source: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """PipMix uygula."""
        return self.pipmix(img, mixing_source)


# Convenience functions
def pipmix_transform(
    img: torch.Tensor,
    patch_size: int = 32,
    alpha: float = 0.4
) -> torch.Tensor:
    """
    Basit PipMix transform fonksiyonu.
    
    Args:
        img: Input görüntü [C, H, W]
        patch_size: Patch boyutu
        alpha: Mixing alpha
        
    Returns:
        Augmented görüntü
    """
    pipmix = PipMix(
        patch_size=patch_size,
        mix_prob=1.0,
        alpha=alpha,
        use_fractal=True,
        use_noise=True
    )
    return pipmix(img)


def pixmix_transform(
    img: torch.Tensor,
    k: int = 4,
    beta: float = 3.0
) -> torch.Tensor:
    """
    Basit PixMix transform fonksiyonu.
    
    Args:
        img: Input görüntü [C, H, W]
        k: Augmentation zinciri uzunluğu
        beta: Mixing beta
        
    Returns:
        Augmented görüntü
    """
    pixmix = PixMix(k=k, beta=beta)
    return pixmix(img)
