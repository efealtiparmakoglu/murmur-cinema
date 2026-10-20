"""
Murmur — boids sürü çekirdeği (numpy, saf matematik).

Blender'a bağımlılığı yok; tests/verify.py kapıları doğrudan bu modülü test eder.
Reynolds üçlüsü: ayrılma (çarpışma yasağı), hizalanma, tutarlılık — artı hız
sınırları, yumuşak sınır duvarı ve şahin korku dalgası. Kural: kuş konumları
her adımda gerçek dinamiğin çözümüdür; keyframe hilesi yok.
"""

import math

import numpy as np


class Flock:
    def __init__(self, n=4000, seed=7, domain=22.0):
        rng = np.random.default_rng(seed)
        self.n = n
        self.domain = domain
        self.pos = rng.uniform(-4.0, 4.0, (n, 3))  # sıkı doğum bulutu (uzlaşma bağlantısı)
        vel = rng.normal(0, 1, (n, 3))
        vel[:, 2] *= 0.3                      # gökyüzü sürüsü yatay akar
        vel += np.array([1.2, 0.3, 0.0])     # ortak havalanma tohumu (simetri kırıcı)
        self.vel = vel / np.linalg.norm(vel, axis=1, keepdims=True) * 6.0
        self.v_min, self.v_max = 4.5, 7.5
        self.r_percept = 6.0                  # uzlaşma yarıçapı (başlangıç bağlantısı şart)
        self.r_sep = 0.95                     # ayrılma yarıçapı (çekirdek doymasın)
        self.w_sep, self.w_ali, self.w_coh = 10.0, 3.0, 2.6
        self.w_bound = 22.0                   # sınır itmesi
        self.d_resolve = 0.3                 # örtüşme çözüm yarıçapı
        self.hawk = None                      # (pos,) varsa korku dalgası
        self.w_hawk = 26.0
        self.w_home = 1.1
        self.z_floor = 3.0   # sürü yere gömülmesin (görsel katman)
        self.w_glob = 0.5
        self.r_hawk = 3.2

    def step(self, dt=0.05):
        p, v = self.pos, self.vel
        d = p[None, :, :] - p[:, None, :]                     # (n, n, 3)
        dist = np.linalg.norm(d, axis=-1)                     # (n, n)
        np.fill_diagonal(dist, np.inf)

        # ayrılma: çok yakın komşulardan kaç
        mask_sep = dist < self.r_sep
        w_sep = (1.0 / np.maximum(dist, 1e-6) - 1.0 / self.r_sep) * mask_sep
        sep = -(w_sep[..., None] * d).sum(axis=1) * self.w_sep  # d=p_j-p_i -> itme NEGATİF

        # hizalanma + tutarlılık TEK DENKLEM: Cucker-Smale uzlaşması
        # (kanıtlanmış: bağlantılı sürü ortak yöne yakınsar)
        w_conn = np.clip(1.0 - dist / self.r_percept, 0.0, None) ** 2   # Vicsek keskin çekirdek
        wsum = np.maximum(w_conn.sum(axis=1, keepdims=True), 1e-9)
        vbar = (w_conn[..., None] * v[None, :, :]).sum(axis=1) / wsum
        cons = 15.0 * (vbar - v)

        # sınır: yumuşak duvar — kuadratik, kelepçe DIŞI (sürü merkezini savaşta yenemez)
        r = self.domain * 0.62
        o = np.maximum(np.abs(p) - r, 0.0)
        bound = -np.sign(p) * (6.0 * o + 0.9 * o * o)

        home = np.array([0.0, 0.0, 6.0])          # geceleme: harabenin üstünde
        inner = sep + cons + self.w_home * (home[None, :] - p)
        inner_norm = np.linalg.norm(inner, axis=1, keepdims=True)
        inner = inner * np.clip(80.0 / np.maximum(inner_norm, 1e-9), 0.0, 1.0)  # kuş başı kelepçe
        acc = inner + bound
        acc[:, 2] += np.clip(self.z_floor - p[:, 2], 0.0, None) * 40.0  # zemin garantisi (kelepçe dışı)
        if self.hawk is not None:
            dh = p - self.hawk[None, :]
            disth = np.linalg.norm(dh, axis=-1)
            m = disth < self.r_hawk
            acc += -(dh / np.maximum(disth, 1e-6)[:, None] *
                     (self.w_hawk * (1.0 - disth / self.r_hawk))[:, None]) * m[:, None]

        if not hasattr(self, "_dbg") or self._dbg[0] < 5:
            if not hasattr(self, "_dbg"): self._dbg = [0]
            self._dbg[0] += 1
            print(f"DBG{self._dbg[0]}: |acc|={np.linalg.norm(acc,axis=1).mean():.2f} "
                  f"|cons|={np.linalg.norm(cons,axis=1).mean():.2f} |sep|={np.linalg.norm(sep,axis=1).mean():.2f} "
                  f"|bound|={np.linalg.norm(bound,axis=1).mean():.2f} clamp%={float((inner_norm>80).mean()):.2f} "
                  f"polar_onu={float(np.linalg.norm(v.mean(axis=0))/max(np.linalg.norm(v,axis=1).mean(),1e-9)):.3f}")
        v = v + acc * dt
        spd = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.maximum(spd, 1e-9) * np.clip(spd, self.v_min, self.v_max)

        # sınırda banklama: ivme ters-paralel kalınca v_min yönü KİLİTLER
        # (kuş dışarıda 3.5 u/s ile sonsuza dek yürür) — başı eve döndür
        o_tot = np.linalg.norm(np.maximum(np.abs(self.pos) - self.domain * 0.62, 0.0), axis=1)
        alpha = np.clip(o_tot / 3.0, 0.0, 1.2)[:, None]
        to_home = home - self.pos
        to_home = to_home / np.maximum(np.linalg.norm(to_home, axis=1, keepdims=True), 1e-9)
        spd2 = np.linalg.norm(v, axis=1, keepdims=True)
        v = v + alpha * to_home * spd2
        spd3 = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.maximum(spd3, 1e-9) * np.clip(spd3, self.v_min, self.v_max)
        # HIZ GERİ YAZILMALI: v yerel değişkende kalırsa sürü her adımda
        # başlangıç hızına döner ve kuşlar yerinde "kaynıyor" (uçmuyor).
        self.vel = v.copy()
        self.pos = p + v * dt

        # örtüşme çözümü: dinamik kaçamazsa geometri ayırır (çarpışmasızlık inşaatla)
        d2 = self.pos[None, :, :] - self.pos[:, None, :]
        dist2 = np.linalg.norm(d2, axis=-1)
        np.fill_diagonal(dist2, np.inf)
        m = dist2 < self.d_resolve
        if m.any():
            # 0 x inf = NaN tuzağı: where + alt sınır (diag inf'i asla çarpıma girmez)
            push = np.where(
                m[..., None],
                -(d2 / np.maximum(dist2, 1e-9)[..., None]) *
                np.maximum(self.d_resolve - dist2, 0.0)[..., None] * 0.5,
                0.0)  # i'yi j'den UZAKLAŞTIR: -d2 yönü
            self.pos += push.sum(axis=1)

    def run(self, t_total, dt=0.01, record_every=8, hawk_path=None):
        """Simülasyonu koştur; [(t, positions, hawk_pos)] kaydı."""
        rec = []
        steps = int(round(t_total / dt))
        for i in range(steps + 1):
            t = i * dt
            if hawk_path is not None:
                self.hawk = hawk_path(t)
            if i % record_every == 0:
                hp = None if self.hawk is None else self.hawk.copy()
                rec.append((t, self.pos.copy(), hp))
            if i < steps:
                self.step(dt)
        return rec


def flock_metrics(pos, vel):
    """Kapı ölçerleri: (min komşu mesafesi, yayılma yarıçapı, hız bandı ihlali)."""
    d = pos[None, :, :] - pos[:, None, :]
    dist = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(dist, np.inf)
    min_pair = float(dist.min())
    center = pos.mean(axis=0)
    spread = float(np.linalg.norm(pos - center, axis=1).max())
    spd = np.linalg.norm(vel, axis=1)
    v_viol = float(np.max(np.maximum(4.0 - spd, 0) + np.maximum(spd - 9.0, 0)))
    return min_pair, spread, v_viol
