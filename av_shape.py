"""
PHYSIQUE DE FORME (Push-T dans le monde du bébé ; décision user 2026-10-21) — numpy seulement.

Avant : tous les objets se poussaient comme des DISQUES (contact au rayon rc, la rotation n'était qu'un bruit au hasard)
-> l'orientation d'un T était incontrôlable. Ici le contact main <-> objet se calcule sur la VRAIE forme (distance signée,
la même que le dessin de av_world5.shape_alpha) et l'impulsion s'applique AU POINT DE CONTACT : pousser une branche du T
le fait tourner, pousser le centre d'un disque le fait glisser sans tourner (pour un disque, rien ne change).
Même code pour le monde vécu (av_world0, --shape_phys) et le monde des tâches (av_plan0.Push0).
"""
import math
import numpy as np

I_K = (0.5, 0.43, 0.2, 0.3)          # moment d'inertie / (m s²) : disque, carré, triangle, T (ordres de grandeur)
OM_MAX = 0.5                          # vitesse de rotation max (rad / image) : stabilité


def shape_sdf(kind, s, th, cx, cy, x, y):
    """distance signée (POSITIVE À L'INTÉRIEUR) à la forme de « rayon » s, orientée th, centrée (cx, cy) — même géométrie
    que av_world5.shape_alpha (avant le passage en pixels)."""
    dx, dy = x - cx, y - cy; c, si = math.cos(th), math.sin(th)
    u, v = c * dx + si * dy, -si * dx + c * dy
    if kind == 0: return s - np.sqrt(dx * dx + dy * dy)
    if kind == 1: return 0.8 * s - np.maximum(abs(u), abs(v))
    if kind == 2: return 0.55 * s - np.max(np.stack([u * math.cos(a) + v * math.sin(a) for a in (math.pi / 2, 7 * math.pi / 6, 11 * math.pi / 6)]), 0)
    bar = np.minimum(s - abs(u), 0.25 * s - abs(v - 0.62 * s))
    leg = np.minimum(0.25 * s - abs(u), 0.62 * s - abs(v + 0.25 * s))
    return np.maximum(bar, leg)


def inertia(kind, m, s):
    return I_K[int(kind)] * m * s * s


def contact(kind, s, th, P, V, om, m, Hc, vh, R, e=1.0, m_hand=1.0):
    """main (disque centre Hc, rayon R, vitesse vh) contre la forme (centre P, vitesse V, rotation om, masse m).
    -> None si pas de contact, sinon dict(nv = normale main -> objet, ov = enfoncement, J = impulsion (≥ 0),
    V, om = nouvelles vitesses de l'objet, c = point de contact). La loi est celle des disques (J = -(1+e) s / (1/m_main + 1/m_eff))
    avec la masse effective au point de contact (1/m + (r×n)²/I) : pour un disque, r×n = 0 -> exactement l'ancienne loi."""
    d = float(shape_sdf(kind, s, th, P[0], P[1], Hc[0], Hc[1]))
    if d <= -R: return None
    h = 1e-3
    g = np.array([float(shape_sdf(kind, s, th, P[0], P[1], Hc[0] + h, Hc[1])) - float(shape_sdf(kind, s, th, P[0], P[1], Hc[0] - h, Hc[1])),
                  float(shape_sdf(kind, s, th, P[0], P[1], Hc[0], Hc[1] + h)) - float(shape_sdf(kind, s, th, P[0], P[1], Hc[0], Hc[1] - h))])
    ng = float(np.linalg.norm(g))
    if ng < 1e-9:                                                     # centre de la main pile sur un pli : direction centre -> centre
        g = np.asarray(P, float) - np.asarray(Hc, float); ng = float(np.linalg.norm(g)) + 1e-9
    nv = (g / ng).astype(np.float32)                                  # vers l'INTÉRIEUR = de la main vers l'objet
    c = (np.asarray(Hc, float) - nv * d).astype(np.float32)          # point de la surface en face de la main
    r = c - np.asarray(P, float); rxn = float(r[0] * nv[1] - r[1] * nv[0])
    I = inertia(kind, m, s)
    vc = np.asarray(V, float) + om * np.array([-r[1], r[0]])          # vitesse du point de contact de l'objet
    s_ = float((vc - np.asarray(vh, float)) @ nv); J = 0.0; V2, om2 = np.asarray(V, np.float32).copy(), float(om)
    if s_ < 0:
        J = -(1 + e) * s_ / (1 / m_hand + 1 / m + rxn * rxn / I)
        V2 = (V2 + J / m * nv).astype(np.float32); om2 = float(np.clip(om + J * rxn / I, -OM_MAX, OM_MAX))
    return dict(nv=nv, ov=float(d + R), J=J, V=V2, om=om2, c=c)
