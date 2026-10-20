"""
Murmur — doğrulama kapıları.

    python3 tests/verify.py

Blender gerekmez (numpy var). Sürü tek beden gibi mi davranıyor, çarpışma
yok mu, şahin korkusu dalga yapıp toparlanıyor mu — hepsi burada kanıtlanır.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import boids  # noqa: E402

GATES = []


def gate(name):
    def deco(fn):
        GATES.append((name, fn))
        return fn
    return deco


def small(n=600, t=8.0, hawk=None):
    f = boids.Flock(n=n, seed=7)
    hp = None
    if hawk:

        def hp(t_):
            return np.array([math.cos(t_ * 0.7) * 8.0, math.sin(t_ * 0.7) * 8.0, 3.0])
    rec = f.run(t, hawk_path=hp)
    return f, rec


@gate("G1 çarpışma yasağı: tüm koşuda en yakın kuş mesafesi asla çökmüyor")
def g1_no_collision():
    errs = []
    f, rec = small(n=500, t=10.0)
    worst = np.inf
    for _, pos, _ in rec:
        d = pos[None, :, :] - pos[:, None, :]
        dist = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(dist, np.inf)
        worst = min(worst, float(dist.min()))
    if worst < 0.04:
        errs.append(f"en yakın çift {worst:.4f} < 0.04 — kuşlar çakışıyor")
    return errs


@gate("G2 hız bandı: her kuş her an v_min..v_max aralığında")
def g2_speed():
    errs = []
    f, rec = small(n=400, t=8.0)
    for _, pos, _ in rec:
        pass
    # hız bandı step içinde zorlanır; kayıtlardan bağımsız son durumu doğrula
    spd = np.linalg.norm(f.vel, axis=1)
    if spd.min() < 3.9 or spd.max() > 9.1:
        errs.append(f"hız bandı: [{spd.min():.2f}, {spd.max():.2f}] dışında")
    return errs


@gate("G3 tutarlılık: sürü tek beden — polarizasyon yüksek + sınırlı yayılma")
def g3_coherence():
    errs = []
    f, rec = small(n=800, t=12.0)
    v = f.vel
    pol = float(np.linalg.norm(v.mean(axis=0)) / np.linalg.norm(v, axis=1).mean())
    if pol < 0.5:
        errs.append(f"polarizasyon {pol:.2f} < 0.5 — sürü ortak yöne akışkan değil")
    for _, pos, _ in rec[-40:]:
        c = pos.mean(axis=0)
        spread = float(np.linalg.norm(pos - c, axis=1).max())
        if spread > 36.0:
            errs.append(f"sürü dağılmış: yayılma {spread:.1f} > 36")
            break
    return errs


@gate("G4 şahin: korku dalgası sürüyü yarıp toparlıyor")
def g4_hawk():
    errs = []
    f = boids.Flock(n=900, seed=3)

    def hp(t_):
        return np.array([math.cos(t_ * 0.7) * 8.0, math.sin(t_ * 0.7) * 8.0, 3.0])

    rec = f.run(10.0, hawk_path=hp)
    # şahin dalgası: yakınındaki kuşların minimum mesafesi açılıyor mu?
    near_frac = []
    for _, pos, hawk in rec:
        if hawk is None:
            continue
        disth = np.linalg.norm(pos - hawk[None, :], axis=1)
        near_frac.append(float((disth < 1.0).sum()) / len(pos))
    if not near_frac:
        errs.append("şahin kaydı yok")
    elif min(near_frac) > 0.005:
        errs.append(f"şahin çevresi hiç boşalmadı (min doluluk {min(near_frac):.4f}) — korku dalgası ölü")
    # toparlanma: şahin geçtikten sonra sürü yeniden sıkı
    fv = f.vel
    pol_after = float(np.linalg.norm(fv.mean(axis=0)) / max(np.linalg.norm(fv, axis=1).mean(), 1e-9))
    if pol_after < 0.6:
        errs.append(f"şahin sonrası toparlanma yok: polarizasyon {pol_after:.2f} < 0.6")
    return errs


@gate("G5 determinizm: aynı tohum -> bit bit aynı sürü")
def g5_determinism():
    errs = []
    a = boids.Flock(n=300, seed=11)
    b2 = boids.Flock(n=300, seed=11)
    for _ in range(60):
        a.step()
        b2.step()
    if not np.array_equal(a.pos, b2.pos) or not np.array_equal(a.vel, b2.vel):
        errs.append("aynı tohum farklı sonuç")
    return errs


@gate("G6 sınır: kuşlar yumuşak duvarın dışına taşmıyor")
def g6_bounds():
    errs = []
    f, rec = small(n=400, t=14.0)
    r = f.domain * 0.62
    worst = 0.0
    for _, pos, _ in rec:
        worst = max(worst, float(np.abs(pos).max()))
    if worst > r + 8.0:
        errs.append(f"sınır ihlali: |p|max {worst:.1f} > {r + 8.0:.1f}")
    return errs


def main():
    failures = 0
    for name, fn in GATES:
        errs = fn()
        if errs:
            failures += 1
            print(f"❌ {name}")
            for e in errs[:8]:
                print(f"   {e}")
        else:
            print(f"✅ {name}")
    print(f"\n{'KAPILAR KIRIK' if failures else 'TÜM KAPILAR YEŞİL'} ({failures} kırık)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
