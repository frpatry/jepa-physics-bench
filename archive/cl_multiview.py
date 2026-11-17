"""
Multi-view Barlow-JEPA : 1 cible (vue propre) + N vues corrompues différemment.

Architecture (« 4 NN », généralisable à N+1) :
  - NN1 (cible)   : pile Transformer sur la vue PROPRE -> embedding z
  - NN2..NN(N+1)  : N piles Transformer, chacune sur une vue CORROMPUE
  - predictor     : combinaison des N vues -> prédit z (NN1)
  - perte         : Barlow Twins(prédit, z)   [pas d'EMA ; redondance = anti-collapse]

FORMES DE BROUILLAGE (par vue, via --corruptions) :
  mask / span / subst / noise / drop  (assignées cycliquement aux vues)

EMBEDDING : partagé par défaut ; --separate_embed => chaque NN a SA table
(token+pos+mask-token) -> enlève la surface de partage (l'embedding commun que
tous les domaines réécrivent) -> teste si ça réduit l'oubli. Coût : (N+1)× la
table d'embedding.

Interface compatible cl_jepa_demo (forward/features/ema_update).
"""
import torch, torch.nn as nn, torch.nn.functional as F
from cl_jepa_text import Block, make_mask
from cl_barlow import barlow_loss

def span_mask(B, T, ratio, dev):
    L = max(1, int(ratio * T))
    start = torch.randint(0, max(1, T - L + 1), (B, 1), device=dev)
    idx = torch.arange(T, device=dev).unsqueeze(0)
    return (idx >= start) & (idx < start + L)

class Stack(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.blocks = nn.ModuleList([Block(a) for _ in range(a.n_layer)])
        self.ln = nn.LayerNorm(a.d_model)
    def forward(self, emb):
        x = emb
        for b in self.blocks:
            x = b(x, None)
        return self.ln(x)

class MultiViewBarlow(nn.Module):
    def __init__(self, a, V):
        super().__init__()
        d = a.d_model
        self.n_views = getattr(a, "n_views", 3)
        self.mask_ratio = a.mask_ratio
        self.V = V
        self.separate = getattr(a, "separate_embed", False)
        ne = (self.n_views + 1) if self.separate else 1     # nb de tables d'embedding
        self.tok = nn.ModuleList([nn.Embedding(V, d) for _ in range(ne)])
        self.pos = nn.ModuleList([nn.Embedding(a.seq, d) for _ in range(ne)])
        self.mask_token = nn.Parameter(torch.zeros(ne, d))
        self.target_stack = Stack(a)                        # NN1 (cible) -> idx embedding 0
        self.ctx_stacks = nn.ModuleList([Stack(a) for _ in range(self.n_views)])  # -> idx 1..N
        pd = getattr(a, "proj_dim", 256)
        self.projector = nn.Sequential(nn.Linear(d, pd), nn.BatchNorm1d(pd), nn.GELU(),
                                       nn.Linear(pd, pd))
        self.predictor = nn.Sequential(nn.Linear(pd * self.n_views, pd), nn.GELU(),
                                       nn.Linear(pd, pd))
        self.lam = getattr(a, "bt_lambda", 5e-3)
        cs = getattr(a, "corruptions", ["mask"]) or ["mask"]
        self.corruptions = [cs[i % len(cs)] for i in range(self.n_views)]
        self.target_mode = "multiview"
    def _ei(self, idx):                                     # index de table (0 si partagé)
        return idx if self.separate else 0
    def _corrupt(self, ids, ctype, ratio, idx):
        ei = self._ei(idx)
        tok, pos, mtok = self.tok[ei], self.pos[ei], self.mask_token[ei]
        e = tok(ids); B, T, d = e.shape
        if ctype == "clean":
            out = e
        elif ctype == "mask":
            m = make_mask(B, T, ratio, None).to(e.device)
            out = torch.where(m.unsqueeze(-1), mtok, e)
        elif ctype == "span":
            m = span_mask(B, T, ratio, e.device)
            out = torch.where(m.unsqueeze(-1), mtok, e)
        elif ctype == "subst":
            m = make_mask(B, T, ratio, None).to(e.device)
            rand = tok(torch.randint(0, self.V, (B, T), device=e.device))
            out = torch.where(m.unsqueeze(-1), rand, e)
        elif ctype == "noise":
            out = e + ratio * torch.randn_like(e) * e.detach().std()
        elif ctype == "drop":
            out = F.dropout(e, p=ratio, training=True)
        else:
            raise ValueError(f"corruption inconnue: {ctype}")
        return out + pos(torch.arange(T, device=e.device))
    def _pool(self, stack, ids, ctype, ratio, idx):
        return stack(self._corrupt(ids, ctype, ratio, idx)).mean(1)
    def forward(self, ids, _mask_ignored):
        z = self.projector(self._pool(self.target_stack, ids, "clean", 0.0, 0))
        parts = [self.projector(self._pool(s, ids, self.corruptions[i], self.mask_ratio, i + 1))
                 for i, s in enumerate(self.ctx_stacks)]
        p = self.predictor(torch.cat(parts, dim=-1))
        loss = barlow_loss(p, z, self.lam)
        with torch.no_grad():
            std = z.std(0).mean().item()
        return loss, loss.item(), std
    def ema_update(self):
        pass
    @torch.no_grad()
    def features(self, ids):
        return self.target_stack(self._corrupt(ids, "clean", 0.0, 0))         # NN1
    @torch.no_grad()
    def component_features(self, ids):
        nn1 = self.target_stack(self._corrupt(ids, "clean", 0.0, 0)).mean(1)
        ctxs = [s(self._corrupt(ids, "clean", 0.0, i + 1)).mean(1)
                for i, s in enumerate(self.ctx_stacks)]
        pred = self.predictor(torch.cat([self.projector(c) for c in ctxs], dim=-1))
        comps = {"NN1": nn1}
        for i, c in enumerate(ctxs):
            comps[f"NN{i + 2}"] = c
        comps["predictor"] = pred
        comps["concat"] = torch.cat([nn1] + ctxs + [pred], dim=-1)
        return comps
    @torch.no_grad()
    def activations(self, ids, layer):
        self.target_stack.blocks[layer].ffn.capture = True
        self.target_stack(self._corrupt(ids, "clean", 0.0, 0))
        self.target_stack.blocks[layer].ffn.capture = False
        h = self.target_stack.blocks[layer].ffn._cap
        self.target_stack.blocks[layer].ffn._cap = None
        return h
