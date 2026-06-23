"""
Barlow-JEPA : structure JEPA (encodeur + PRÉDICTEUR) + objectif BARLOW TWINS.

Idée (hybride demandé) :
  - on garde le PRÉDICTEUR de JEPA (le côté prédictif : prédire l'embedding de
    la vue propre depuis la vue masquée) ;
  - mais l'objectif n'est PLUS smooth-L1 contre une cible EMA ; c'est la perte
    BARLOW TWINS sur la cross-corrélation entre l'embedding prédit (p) et
    l'embedding de la vue propre (z) :

        C = (p̂ᵀ ẑ) / B          (p̂, ẑ normalisés sur le batch, dim par dim)
        L = Σ_i (1 − C_ii)²      ← invariance (les 2 vues coïncident)
          + λ Σ_{i≠j} C_ij²      ← réduction de redondance (dims décorrélées)

Conséquence : PAS d'EMA (la réduction de redondance empêche le collapse toute
seule), pas de smooth-L1, pas de terme de variance ad hoc. Plus simple, plus
stable que l'EMA (qui était instable chez nous).

Deux vues : PROPRE + MASQUÉE (les « deux sets de neurones » = 2 passages dans
l'encodeur partagé). Embedding niveau-séquence (mean-pool) -> projection -> BT.

Interface identique à TextJEPA -> branchable dans cl_jepa_demo / cl_jepa_probe.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from cl_jepa_text import Encoder      # encodeur partagé (tok+pos+blocs+ln), routage inchangé

def barlow_loss(p, z, lam):
    """Perte Barlow Twins entre deux lots d'embeddings (B, D)."""
    B = p.size(0)
    p = (p - p.mean(0)) / (p.std(0) + 1e-5)
    z = (z - z.mean(0)) / (z.std(0) + 1e-5)
    C = (p.T @ z) / B                                   # (D,D) cross-corrélation
    diag = torch.diagonal(C)
    on = ((diag - 1.0) ** 2).sum()                      # invariance
    off = (C ** 2).sum() - (diag ** 2).sum()            # redondance (hors-diagonale)
    return on + lam * off

class BarlowJEPA(nn.Module):
    def __init__(self, a, V):
        super().__init__()
        d = a.d_model
        self.enc = Encoder(a, V)                         # partagé entre les 2 vues
        self.mask_token = nn.Parameter(torch.zeros(d))
        pd = getattr(a, "proj_dim", 256)
        self.projector = nn.Sequential(nn.Linear(d, pd), nn.BatchNorm1d(pd), nn.GELU(),
                                       nn.Linear(pd, pd))
        self.predictor = nn.Sequential(nn.Linear(pd, pd), nn.GELU(), nn.Linear(pd, pd))
        self.lam = getattr(a, "bt_lambda", 5e-3)
        self.stopgrad = getattr(a, "bt_stopgrad", False)
        self.target_mode = "barlow"                     # marqueur (pas d'EMA)
    def _pool(self, ids, mask=None):
        mt = self.mask_token if mask is not None else None
        h = self.enc(ids, None, mask, mt)               # (B,T,d) ; rc=None car dense
        return h.mean(1)                                # embedding séquence (B,d)
    def forward(self, ids, mask):
        p = self.predictor(self.projector(self._pool(ids, mask)))   # vue MASQUÉE -> prédite
        z = self.projector(self._pool(ids, None))                   # vue PROPRE
        if self.stopgrad:
            z = z.detach()
        loss = barlow_loss(p, z, self.lam)
        with torch.no_grad():
            std = z.std(0).mean().item()                # moniteur de collapse (~1 = sain)
        return loss, loss.item(), std
    def ema_update(self):
        pass                                            # pas d'EMA
    @torch.no_grad()
    def features(self, ids):
        return self.enc(ids, None)                      # (B,T,d) ; le probe mean-pool
    @torch.no_grad()
    def activations(self, ids, layer):
        self.enc.blocks[layer].ffn.capture = True
        self.enc(ids, None)
        self.enc.blocks[layer].ffn.capture = False
        h = self.enc.blocks[layer].ffn._cap; self.enc.blocks[layer].ffn._cap = None
        return h
