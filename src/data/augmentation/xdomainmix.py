"""
XDomainMix: Cross-Domain Mixup Augmentation for Domain Generalization

XDomainMix, farklı domain'lerden gelen görüntüleri karıştırarak domain-invariant
özellik öğrenmeyi teşvik eden bir augmentation tekniğidir.

Teknik Detaylar:
- Kaynak domain'den bir görüntü alınır (anchor)
- Farklı bir domain'den rastgele bir görüntü seçilir (cross-domain sample)
- İki görüntü belirli bir oranda (lambda) karıştırılır
- Label'lar da aynı oranda karıştırılır (soft labels)

Referans:
- Domain Generalization araştırmasında yaygın kullanılan MixUp ve CutMix
  tekniklerinin cross-domain varyasyonu
"""

import random
from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn
from PIL import Image
import numpy as np


class XDomainMix(nn.Module):
    """
    Cross-Domain Mixup augmentation.
    
    Farklı domain'lerden gelen görüntüleri mixup yöntemiyle karıştırır.
    Bu, modelin domain-agnostic özellikler öğrenmesine yardımcı olur.
    
    Args:
        alpha: Beta dağılımının parametresi (mixup oranı için)
        prob: Augmentation uygulama olasılığı
        mix_strategy: Karıştırma stratejisi ('mixup', 'cutmix', 'hybrid')
    
    Example:
        >>> xdomain_mix = XDomainMix(alpha=0.2, prob=0.5)
        >>> mixed_img, mixed_label, lam = xdomain_mix(
        ...     img1, label1, domain1,
        ...     img2, label2, domain2
        ... )
    """
    
    def __init__(
        self,
        alpha: float = 0.2,
        prob: float = 0.5,
        mix_strategy: str = 'mixup'
    ):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
        self.mix_strategy = mix_strategy
        
        if mix_strategy not in ['mixup', 'cutmix', 'hybrid']:
            raise ValueError(f"Invalid mix_strategy: {mix_strategy}. "
                           f"Choose from 'mixup', 'cutmix', 'hybrid'")
    
    def _sample_lambda(self) -> float:
        """Beta dağılımından lambda değeri örnekle."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        return lam
    
    def _mixup(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """
        Standart mixup: iki görüntüyü lineer olarak karıştır.
        
        Args:
            img1: Anchor görüntü [C, H, W]
            img2: Cross-domain görüntü [C, H, W]
            lam: Karıştırma oranı
            
        Returns:
            Karıştırılmış görüntü [C, H, W]
        """
        return lam * img1 + (1 - lam) * img2
    
    def _cutmix(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        lam: float
    ) -> Tuple[torch.Tensor, float]:
        """
        CutMix: bir görüntüden patch keserek diğerine yapıştır.
        
        Args:
            img1: Anchor görüntü [C, H, W]
            img2: Cross-domain görüntü [C, H, W]
            lam: Patch boyutu oranı
            
        Returns:
            Tuple of (karıştırılmış görüntü, güncellenmiş lambda)
        """
        _, h, w = img1.shape
        
        # Patch boyutunu hesapla
        cut_ratio = np.sqrt(1. - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Patch merkezi
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Görüntüleri karıştır
        mixed = img1.clone()
        mixed[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]
        
        # Gerçek lambda'yı hesapla (patch alanına göre)
        actual_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return mixed, actual_lam
    
    def forward(
        self,
        img1: torch.Tensor,
        label1: torch.Tensor,
        domain1: int,
        img2: torch.Tensor,
        label2: torch.Tensor,
        domain2: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        XDomainMix uygula.
        
        Args:
            img1: Anchor görüntü [C, H, W]
            label1: Anchor label (one-hot veya class index)
            domain1: Anchor domain ID
            img2: Cross-domain görüntü [C, H, W]
            label2: Cross-domain label
            domain2: Cross-domain ID
            
        Returns:
            Tuple of:
                - mixed_img: Karıştırılmış görüntü
                - mixed_label: Karıştırılmış label
                - lam: Kullanılan mixup oranı
        """
        # Aynı domain'den geliyorsa karıştırma yapma
        if domain1 == domain2:
            return img1, label1, 1.0
        
        # Probability check
        if random.random() > self.prob:
            return img1, label1, 1.0
        
        # Lambda örnekle
        lam = self._sample_lambda()
        
        # Strateji seç
        if self.mix_strategy == 'mixup':
            mixed_img = self._mixup(img1, img2, lam)
        elif self.mix_strategy == 'cutmix':
            mixed_img, lam = self._cutmix(img1, img2, lam)
        else:  # hybrid
            if random.random() < 0.5:
                mixed_img = self._mixup(img1, img2, lam)
            else:
                mixed_img, lam = self._cutmix(img1, img2, lam)
        
        # Label'ları karıştır
        if label1.dim() == 0:
            # Class index durumu - one-hot'a çevir
            num_classes = max(label1.item(), label2.item()) + 1
            label1_onehot = torch.zeros(num_classes)
            label2_onehot = torch.zeros(num_classes)
            label1_onehot[label1] = 1
            label2_onehot[label2] = 1
            mixed_label = lam * label1_onehot + (1 - lam) * label2_onehot
        else:
            # Zaten one-hot veya soft label
            mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_img, mixed_label, lam


class XDomainMixBatch(nn.Module):
    """
    Batch-level XDomainMix uygulaması.
    
    Bir batch içindeki farklı domain'lerden gelen örnekleri karıştırır.
    Training loop içinde kullanılmak üzere tasarlanmıştır.
    
    Args:
        alpha: Beta dağılımının parametresi
        prob: Her örnek için augmentation uygulama olasılığı
        mix_strategy: Karıştırma stratejisi
        same_domain_fallback: Aynı domain'de karıştırma yapılıp yapılmayacağı
        
    Example:
        >>> xdomain_batch = XDomainMixBatch(alpha=0.2)
        >>> mixed_images, mixed_labels, lam = xdomain_batch(
        ...     images=batch['image'],      # [B, C, H, W]
        ...     labels=batch['label'],      # [B] or [B, num_classes]
        ...     domains=batch['domain']     # [B]
        ... )
    """
    
    def __init__(
        self,
        alpha: float = 0.2,
        prob: float = 0.5,
        mix_strategy: str = 'mixup',
        same_domain_fallback: bool = False
    ):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
        self.mix_strategy = mix_strategy
        self.same_domain_fallback = same_domain_fallback
        self.xdomain_mix = XDomainMix(alpha, prob=1.0, mix_strategy=mix_strategy)
    
    def _get_cross_domain_indices(
        self,
        domains: torch.Tensor
    ) -> torch.Tensor:
        """
        Her örnek için farklı domain'den bir eşleşme bul.
        
        Args:
            domains: Domain ID'leri [B]
            
        Returns:
            Eşleşme indeksleri [B]
        """
        batch_size = domains.size(0)
        indices = torch.zeros(batch_size, dtype=torch.long, device=domains.device)
        
        for i in range(batch_size):
            current_domain = domains[i].item()
            
            # Farklı domain'deki örnekleri bul
            cross_domain_mask = domains != current_domain
            cross_domain_indices = torch.where(cross_domain_mask)[0]
            
            if len(cross_domain_indices) > 0:
                # Farklı domain'den rastgele seç
                rand_idx = torch.randint(len(cross_domain_indices), (1,))
                indices[i] = cross_domain_indices[rand_idx]
            elif self.same_domain_fallback:
                # Aynı domain'den farklı bir örnek seç
                other_indices = torch.where(torch.arange(batch_size, device=domains.device) != i)[0]
                rand_idx = torch.randint(len(other_indices), (1,))
                indices[i] = other_indices[rand_idx]
            else:
                # Kendini döndür (karıştırma yapılmayacak)
                indices[i] = i
        
        return indices
    
    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        domains: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch'e XDomainMix uygula.
        
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
        batch_size = images.size(0)
        device = images.device
        
        # Cross-domain eşleşmelerini bul
        partner_indices = self._get_cross_domain_indices(domains)
        
        # Her örnek için lambda örnekle
        if self.alpha > 0:
            lam_values = torch.from_numpy(
                np.random.beta(self.alpha, self.alpha, batch_size)
            ).float().to(device)
        else:
            lam_values = torch.ones(batch_size, device=device)
        
        # Probability mask
        prob_mask = torch.rand(batch_size, device=device) < self.prob
        
        # Aynı domain mask (karıştırma yapılmayacak)
        same_domain_mask = partner_indices == torch.arange(batch_size, device=device)
        
        # Karıştırma yapılmayacak örnekler için lambda=1
        lam_values[~prob_mask | same_domain_mask] = 1.0
        
        # Partner görüntü ve label'ları al
        partner_images = images[partner_indices]
        partner_labels = labels[partner_indices]
        
        # Mixup uygula
        lam_expanded = lam_values.view(-1, 1, 1, 1)
        mixed_images = lam_expanded * images + (1 - lam_expanded) * partner_images
        
        # Label mixup
        if labels.dim() == 1:
            # Soft label oluştur
            mixed_labels = torch.stack([
                lam_values,  # Original class confidence
                1 - lam_values  # Partner class confidence
            ], dim=1)
            # Original ve partner class index'lerini de döndür
            original_labels = labels
            partner_labels_ret = labels[partner_indices]
        else:
            lam_expanded_labels = lam_values.view(-1, 1)
            mixed_labels = lam_expanded_labels * labels + (1 - lam_expanded_labels) * partner_labels
            original_labels = None
            partner_labels_ret = None
        
        return mixed_images, mixed_labels, lam_values


class XDomainMixDataset(torch.utils.data.Dataset):
    """
    XDomainMix için multi-domain dataset wrapper.
    
    Birden fazla domain dataset'ini birleştirir ve cross-domain
    çiftleri oluşturur.
    
    Args:
        datasets: Domain dataset'lerinin dict'i {domain_name: dataset}
        transform: Uygulanacak transform
        return_domain: Domain bilgisinin döndürülüp döndürülmeyeceği
        
    Example:
        >>> datasets = {
        ...     'nih': NIHChestXrayDataset(...),
        ...     'covidx': COVIDxDataset(...),
        ...     'vinbigdata': VinBigDataDataset(...)
        ... }
        >>> xdomain_dataset = XDomainMixDataset(datasets)
        >>> sample = xdomain_dataset[0]
        >>> # sample = {'image': tensor, 'label': tensor, 'domain': int, 
        >>> #           'domain_name': str, 'cross_domain_image': tensor, ...}
    """
    
    def __init__(
        self,
        datasets: Dict[str, torch.utils.data.Dataset],
        transform: Optional[nn.Module] = None,
        return_domain: bool = True
    ):
        super().__init__()
        self.datasets = datasets
        self.transform = transform
        self.return_domain = return_domain
        
        # Domain mapping
        self.domain_names = list(datasets.keys())
        self.domain_to_id = {name: idx for idx, name in enumerate(self.domain_names)}
        
        # Cumulative lengths (hangi index'in hangi dataset'e ait olduğunu bulmak için)
        self.cumulative_lengths = []
        total = 0
        for name in self.domain_names:
            total += len(datasets[name])
            self.cumulative_lengths.append(total)
        
        self._length = total
    
    def __len__(self) -> int:
        return self._length
    
    def _get_domain_and_local_idx(self, idx: int) -> Tuple[int, int]:
        """Global index'ten domain ve local index çıkar."""
        for domain_idx, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                local_idx = idx if domain_idx == 0 else idx - self.cumulative_lengths[domain_idx - 1]
                return domain_idx, local_idx
        raise IndexError(f"Index {idx} out of range")
    
    def _get_sample_from_domain(
        self,
        domain_idx: int,
        local_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Belirtilen domain'den sample al."""
        domain_name = self.domain_names[domain_idx]
        sample = self.datasets[domain_name][local_idx]
        
        # Dataset'in döndürdüğü formatı handle et
        if isinstance(sample, tuple):
            image, label = sample[0], sample[1]
        elif isinstance(sample, dict):
            image = sample.get('image', sample.get('img'))
            label = sample.get('label', sample.get('target'))
        else:
            image = sample
            label = torch.tensor(0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _get_cross_domain_sample(
        self,
        exclude_domain: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Belirtilen domain dışından rastgele sample al."""
        # Farklı domain seç
        available_domains = [i for i in range(len(self.domain_names)) if i != exclude_domain]
        
        if not available_domains:
            # Sadece bir domain varsa aynı domain'den farklı örnek al
            target_domain = exclude_domain
        else:
            target_domain = random.choice(available_domains)
        
        # O domain'den rastgele index seç
        domain_len = len(self.datasets[self.domain_names[target_domain]])
        local_idx = random.randint(0, domain_len - 1)
        
        image, label = self._get_sample_from_domain(target_domain, local_idx)
        
        return image, label, target_domain
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Sample döndür.
        
        Returns:
            Dict with keys:
                - 'image': Anchor görüntü
                - 'label': Anchor label
                - 'domain': Domain ID
                - 'domain_name': Domain adı
                - 'cross_image': Cross-domain görüntü
                - 'cross_label': Cross-domain label
                - 'cross_domain': Cross-domain ID
        """
        domain_idx, local_idx = self._get_domain_and_local_idx(idx)
        
        # Anchor sample
        image, label = self._get_sample_from_domain(domain_idx, local_idx)
        
        # Cross-domain sample
        cross_image, cross_label, cross_domain_idx = self._get_cross_domain_sample(domain_idx)
        
        result = {
            'image': image,
            'label': label,
            'domain': torch.tensor(domain_idx),
            'domain_name': self.domain_names[domain_idx],
            'cross_image': cross_image,
            'cross_label': cross_label,
            'cross_domain': torch.tensor(cross_domain_idx),
            'cross_domain_name': self.domain_names[cross_domain_idx]
        }
        
        return result


# Convenience function
def xdomain_mixup(
    img1: torch.Tensor,
    img2: torch.Tensor,
    label1: torch.Tensor,
    label2: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Basit XDomainMix fonksiyonu.
    
    Args:
        img1: Birinci görüntü
        img2: İkinci görüntü (farklı domain'den)
        label1: Birinci label
        label2: İkinci label
        alpha: Beta dağılım parametresi
        
    Returns:
        Tuple of (mixed_image, mixed_label, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    mixed_img = lam * img1 + (1 - lam) * img2
    mixed_label = lam * label1 + (1 - lam) * label2
    
    return mixed_img, mixed_label, lam
