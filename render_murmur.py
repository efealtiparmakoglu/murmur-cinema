"""
Murmur: gün batımında harabe üstünde sürü — 4.000 kuş, gerçek boids dinamiği.

    blender --background --python render_murmur.py -- --preview   # 4 kare 640x360
    blender --background --python render_murmur.py                # 49 kare 960x540
    blender --background --python render_murmur.py -- --olc       # sadece kompozisyon olcumu

FİZİK → GÖRÜNTÜ SÖZLEŞMESİ
---------------------------
boids.py saf numpy çekirdeğidir ve HİÇBİR ŞEY DEĞİŞTİRİLMEZ. Ekranda
görünen her konum, o karede çözülmüş konumdur; Blender animasyonu bu kaydı
LINEAR anahtarlarla taşır (easing, sinüs, prosedürel salınım YOKTUR).
Kamera da keyframe hilesi değildir: her karede kuş bulutunun canlı ağırlık
merkezine bakan açı AÇIKÇA hesaplanır (TRACK_TO constraint yok — 5.x'te
constraint + matrix_world güncellemesi kırıktır).

ÖLÇEK TUZAĞI (bu projede iki kez ısırıldı)
--------------------------------------------
Blender'da ölçek NESNE üzerindedir, mesh verisinde değildir. Prototip
"proto.scale = (...)" ile ölçeklenip prototip NESNESİ silinince ölçek
kaybolur: kuşlar 1 birimlik dev koniler olur. Burada ölçek doğrudan MESH
VERİSİNE yazılır (from_pydata ile vertexler birim uzunlukta değil, gerçek
boyutta üretilir) — ayrıca hiçbir yerde proto.scale kullanılmaz.
"""

import math
import os
import sys
import time

import numpy as np
from mathutils import Vector

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import bpy  # noqa: E402

import boids  # noqa: E402

# ------------------------------------------------------------------ sabitler
N_BIRDS = 4000
PREVIEW_BIRDS = 1200
CLIP = 48                 # 49 kare (1..49)
FPS = 12
SIM_T = 10.0
ANIM_T0 = SIM_T - CLIP / FPS
DOMAIN = 7.0              # boids alanı: sürü 22 birim değil ~11 birim çapında
BIRD_LEN = 0.24           # kuş boyu (birim) — hedef ~0.2
BIRD_SPAN = 0.15
HAWK_LEN = 0.66           # şahin kuşlardan ~3x büyük
HAWK_SPAN = 0.62

LENS = 48.0
SENSOR_W = 36.0
HORIZON_V = 0.33          # ufuk ALT 1/3'te (v=alt'tan) -> gökyüzü kareye hakim
CAM_Y = -58.0             # sürü merkezine ~58 birim (sürü kadrajı daha doldurur)
CAM_Z = 0.8
CAM_DRIFT = 7.0           # yatay kayma (birim)
CAM_RISE = 0.55           # hafif yukarı çıkış
SUN_ELEV = 1.8            # güneş yüksekliği (derece)
SUN_OFFSET = 16.0         # kamera eksenine göre sağdaki sapma (derece)
DOME_R = 30000.0
GROUND_SIZE = 4000.0

PREVIEW_FRAMES = [1, 16, 32, 49]
SIM_VER = 2                # sahin yolu/ayar degisince bu surum artirilir


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return {"preview": "--preview" in a, "olc": "--olc" in a}


# ------------------------------------------------------------------ eylemler
def _yeni_aksiyon(ad):
    """Blender 5.x eylem API'si: action -> layer -> strip -> channelbag -> fcurve."""
    act = bpy.data.actions.new(ad)
    lay = act.layers.new("Katman")
    strip = lay.strips.new(type="KEYFRAME")
    return act, strip


def _fcurve(strip, slot, yol, indeks, kareler, degerler):
    bag = strip.channelbag(slot, ensure=True)
    fc = bag.fcurves.new(yol, index=indeks)
    n = len(kareler)
    fc.keyframe_points.add(n)
    co = np.empty(n * 2, dtype=np.float32)
    co[0::2] = np.asarray(kareler, dtype=np.float32)
    co[1::2] = np.asarray(degerler, dtype=np.float32)
    fc.keyframe_points.foreach_set("co", co)
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"      # easing yasak
    fc.update()
    return fc


def _hepsi_lineer(ob):
    ad = ob.animation_data
    if not (ad and ad.action):
        return
    for lay in ad.action.layers:
        for st in lay.strips:
            for bag in st.channelbags:
                for fc in bag.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "LINEAR"


# ------------------------------------------------------------------ materyal
def _materyal(ad, taban, rough, spec=0.5, emission=None, emis_str=0.0):
    m = bpy.data.materials.new(ad)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*taban, 1.0)
    b.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = spec
    elif "Specular" in b.inputs:
        b.inputs["Specular"].default_value = spec
    if emission is not None:
        b.inputs["Emission Color"].default_value = (*emission, 1.0)
        b.inputs["Emission Strength"].default_value = emis_str
    return m


# ------------------------------------------------------------------ gök kubbesi
def gokyuzu(sc, kamera_azim):
    """Gün batımı göğü: fiziksel gökyüzü düğümü YOK, ölçülmüş prosedurel kubbe.

    Dünya shader'ında yön vektörü karışıktı; bu yüzden gökyüzü, kameranın
    çevreleyen 30 km'lik bir ışık kubbesi olarak modellenir: kubbenin ekvatoru
    tam olarak ufuk çizgisidir (kamera merkezden ~60 birim uzakta, parallax
    0.1 dereceden az). Renk, kubbenin nesne koordinatından hesaplanır:
      dikey eksen -> ufukta altın, yukarı doğru koyu lacivert-morarden
      gunes yonu   -> sicak hale + kucuk parlak disk
    """
    # kubbe
    bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=DOME_R,
                                        location=(0, 0, 0))
    kubbe = bpy.context.active_object
    kubbe.name = "GokKubbesi"
    bpy.ops.object.shade_smooth()
    m = bpy.data.materials.new("Gok")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    cik = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0

    tc = nt.nodes.new("ShaderNodeTexCoord")
    norm = nt.nodes.new("ShaderNodeVectorMath")
    norm.operation = "NORMALIZE"
    nt.links.new(tc.outputs["Object"], norm.inputs[0])
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(norm.outputs["Vector"], sep.inputs["Vector"])

    # dikey gecis: z = sin(yukseklik)
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.0
    mr.inputs["From Max"].default_value = 0.30     # ~17.5 derece
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "EASE"
    cr.elements[0].position = 0.0
    cr.elements[0].color = (1.0, 0.26, 0.045, 1)    # ufuk: derin turuncu
    cr.elements[1].position = 1.0
    cr.elements[1].color = (0.012, 0.010, 0.045, 1)  # zenit: lacivert-morarden
    for pos, col in ((0.07, (1.0, 0.50, 0.11)),      # altin bant
                     (0.15, (0.88, 0.27, 0.075)),    # turuncu
                     (0.26, (0.33, 0.075, 0.060)),    # koyu kizil
                     (0.40, (0.082, 0.030, 0.062)),   # mor
                     (0.58, (0.020, 0.017, 0.048))): # lacivert
        e = cr.elements.new(pos)
        e.color = (*col, 1)
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])

    # gunes yonu (dunya koordinati): azimut kamera eksenine gore
    az = math.radians(kamera_azim + SUN_OFFSET)
    el = math.radians(SUN_ELEV)
    sun_dir = (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))
    sdx = nt.nodes.new("ShaderNodeCombineXYZ")
    sdx.inputs["X"].default_value = sun_dir[0]
    sdx.inputs["Y"].default_value = sun_dir[1]
    sdx.inputs["Z"].default_value = sun_dir[2]
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    nt.links.new(norm.outputs["Vector"], dot.inputs[0])
    nt.links.new(sdx.outputs["Vector"], dot.inputs[1])
    cl = nt.nodes.new("ShaderNodeMath")
    cl.operation = "MAXIMUM"
    cl.inputs[1].default_value = 0.0
    nt.links.new(dot.outputs["Value"], cl.inputs[0])
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 62.0
    nt.links.new(cl.outputs["Value"], pw.inputs[0])
    gl = nt.nodes.new("ShaderNodeMath")
    gl.operation = "MULTIPLY"
    gl.inputs[1].default_value = 1.15
    nt.links.new(pw.outputs["Value"], gl.inputs[0])

    # gunes diski: kucuk ve parlak
    dk = nt.nodes.new("ShaderNodeMath")
    dk.operation = "GREATER_THAN"
    dk.inputs[1].default_value = math.cos(math.radians(0.42))
    nt.links.new(dot.outputs["Value"], dk.inputs[0])
    dv = nt.nodes.new("ShaderNodeMath")
    dv.operation = "MULTIPLY"
    dv.inputs[1].default_value = 12.0
    nt.links.new(dk.outputs["Value"], dv.inputs[0])

    # sicak hale + disk: skaler -> renk vektoru, sonra toplama
    def _skaler_vektor(node_out, renk, katsayi):
        c = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(node_out, c.inputs["X"])
        nt.links.new(node_out, c.inputs["Y"])
        nt.links.new(node_out, c.inputs["Z"])
        mul = nt.nodes.new("ShaderNodeVectorMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = (renk[0] * katsayi, renk[1] * katsayi, renk[2] * katsayi)
        nt.links.new(c.outputs["Vector"], mul.inputs[0])
        return mul.outputs["Vector"]

    v_hale = _skaler_vektor(gl.outputs["Value"], (1.0, 0.52, 0.22), 1.0)
    v_disk = _skaler_vektor(dv.outputs["Value"], (1.0, 0.46, 0.16), 1.0)
    topla1 = nt.nodes.new("ShaderNodeVectorMath")
    topla1.operation = "ADD"
    nt.links.new(ramp.outputs["Color"], topla1.inputs[0])
    nt.links.new(v_hale, topla1.inputs[1])
    topla2 = nt.nodes.new("ShaderNodeVectorMath")
    topla2.operation = "ADD"
    nt.links.new(topla1.outputs["Vector"], topla2.inputs[0])
    nt.links.new(v_disk, topla2.inputs[1])
    nt.links.new(topla2.outputs["Vector"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], cik.inputs["Surface"])
    kubbe.data.materials.append(m)

    # Dunya arka planini kapat (kubbe her yeri kaplar)
    w = bpy.data.worlds.new("Dunya")
    sc.world = w
    w.use_nodes = True
    bgn = w.node_tree.nodes.get("Background")
    if bgn:
        bgn.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bgn.inputs["Strength"].default_value = 0.0

    # gunes lambasi: gokyuzundeki gunesle ayni yonde
    sun = bpy.data.lights.new("Gun", "SUN")
    sun.energy = 11.0
    sun.angle = math.radians(1.8)
    sun.color = (1.0, 0.44, 0.15)
    so = bpy.data.objects.new("Gun", sun)
    sc.collection.objects.link(so)
    so.rotation_euler = Vector(sun_dir).to_track_quat("Z", "Y").to_euler()

    # soguk gokyuzu dolgusu: alacakaranlikta yerin TAM siyah kalmamasi icin
    # (dusey gunes zemine neredeyse hic vurmaz) yumusak, dusuk yogunluklu isik
    dolgu = bpy.data.lights.new("Dolgu", "SUN")
    dolgu.energy = 1.25
    dolgu.angle = math.radians(30.0)
    dolgu.color = (0.40, 0.52, 0.82)
    do = bpy.data.objects.new("Dolgu", dolgu)
    sc.collection.objects.link(do)
    do.rotation_euler = Vector((0.16, 0.10, -1.0)).to_track_quat("Z", "Y").to_euler()
    return kubbe, sun_dir


# ------------------------------------------------------------------ zemin
def _gurultu(x, y, seed):
    """Deterministik, kati katmanli deger gurultusu (numpy, hizli)."""
    out = np.zeros_like(x, dtype=np.float64)
    for k, (fx, fy, amp, ph) in enumerate(((0.010, 0.013, 1.0, 0.0),
                                           (0.027, 0.021, 0.55, 1.7),
                                           (0.061, 0.048, 0.28, 3.1),
                                           (0.130, 0.104, 0.13, 5.2))):
        out += amp * np.sin(x * fx * 2 * math.pi + ph) * np.cos(y * fy * 2 * math.pi + ph * 1.3)
    return out


def zemin(sc, coll):
    """Dalgali koyu toprak (300 birimlik duz duzlem DEGIL) + uzak siratlar."""
    m_ground = _materyal("KA_Toprak", (0.20, 0.155, 0.115), 0.95, spec=0.45)
    nt = m_ground.node_tree
    b = nt.nodes["Principled BSDF"]
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 9.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.6
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.55
    bump.inputs["Distance"].default_value = 0.35
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    # leke leke parlaklik: alacakaranlikta yatay gunesin toz üzerindeki
    # yansimasi -> ufka dogru sicak bir isik yolu
    n3 = nt.nodes.new("ShaderNodeTexNoise")
    n3.inputs["Scale"].default_value = 0.35
    n3.inputs["Detail"].default_value = 3.0
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["From Min"].default_value = 0.35
    rr.inputs["From Max"].default_value = 0.65
    rr.inputs["To Min"].default_value = 0.52
    rr.inputs["To Max"].default_value = 0.95
    nt.links.new(n3.outputs["Fac"], rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], b.inputs["Roughness"])
    # genis olcekli renk varyasyonu (kum/ot)
    n2 = nt.nodes.new("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 0.02
    n2.inputs["Detail"].default_value = 4.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.115, 0.088, 0.068, 1)
    ramp.color_ramp.elements[1].color = (0.30, 0.225, 0.160, 1)
    nt.links.new(n2.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])

    # --- zemin agi: uzaklara dogru kacinca hucre buyur (telefoto icin onemli)
    N = 260
    u = np.linspace(-1.0, 1.0, N)
    s = np.sign(u) * np.abs(u) ** 3.0
    xs = s * (GROUND_SIZE * 0.5)
    ys = s * (GROUND_SIZE * 0.5) + CAM_Y
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    r = np.hypot(X - 0.0, Y - CAM_Y)
    amp = 4.2 * np.clip((r - 12.0) / 90.0, 0.0, 1.0)      # kameranin onunde duz
    Z = _gurultu(X, Y, 11) * amp
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).tolist()
    faces = []
    for i in range(N - 1):
        for j in range(N - 1):
            a = i * N + j
            faces.append((a, a + N, a + N + 1))
            faces.append((a, a + N + 1, a + 1))
    me = bpy.data.meshes.new("ZeminMesh")
    me.from_pydata(verts, [], faces)
    me.update()
    zemin_ob = bpy.data.objects.new("Toprak", me)
    coll.objects.link(zemin_ob)
    zemin_ob.data.materials.append(m_ground)

    # --- uzak siratlar: ufuk cizgisini duz bant olmaktan cikarir
    m_ridge = _materyal("KA_Sirt", (0.030, 0.026, 0.030), 1.0, spec=0.0)
    for idx, (yy, yuk, uz) in enumerate(((260.0, 11.0, 1500.0),
                                        (520.0, 20.0, 2200.0),
                                        (1150.0, 34.0, 3200.0))):
        M = 120
        x = np.linspace(-uz / 2, uz / 2, M)
        h = (0.55 + 0.45 * np.sin(x * 0.0043 + idx * 2.1) * np.cos(x * 0.0017 + idx)) \
            * yuk * (0.6 + 0.4 * np.sin(x * 0.0009 + idx * 1.3))
        vv = []
        for k in range(M):
            vv.append((x[k], yy, 0.0))
        for k in range(M):
            vv.append((x[k], yy, float(h[k])))
        ff = []
        for k in range(M - 1):
            ff.append((k, k + 1, M + k + 1))
            ff.append((k, M + k + 1, M + k))
        m2 = bpy.data.meshes.new(f"Sirt{idx}Mesh")
        m2.from_pydata(vv, [], ff)
        m2.update()
        ob = bpy.data.objects.new(f"Sirt_{idx}", m2)
        coll.objects.link(ob)
        ob.data.materials.append(m_ridge)
    return zemin_ob


# ------------------------------------------------------------------ harabe
def _dal_parca(vs, fs, bas, yon, uzunluk, yaricap, egim=0.0):
    """Tek dal: 4 kenarli koni, ucu incelir."""
    d = np.array(yon, dtype=float)
    d /= max(np.linalg.norm(d), 1e-9)
    e1 = np.cross(d, [0, 0, 1.0])
    if np.linalg.norm(e1) < 1e-4:
        e1 = np.cross(d, [1.0, 0, 0])
    e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(d, e1)
    taban = len(vs)
    for k, (t, r) in enumerate(((0.0, yaricap), (0.62, yaricap * 0.62), (1.0, yaricap * 0.18))):
        merkez = np.array(bas) + d * (uzunluk * t) + np.array([0, 0, -egim * uzunluk * t * t])
        for j in range(4):
            a = j * math.pi / 2
            p = merkez + e1 * (math.cos(a) * r) + e2 * (math.sin(a) * r)
            vs.append(tuple(p))
    for k in range(2):
        for j in range(4):
            a = taban + k * 4 + j
            b = taban + k * 4 + (j + 1) % 4
            fs.append((a, b, b + 4))
            fs.append((a, b + 4, a + 4))
    fs.append((taban + 8, taban + 9, taban + 10))
    fs.append((taban + 8, taban + 10, taban + 11))


def calilik(sc, coll, zemin_ob):
    """Orta plan siluetleri: olu agaclar + kayalar -> bos on plan dolmasin."""
    rng = np.random.default_rng(23)
    m_wood = _materyal("KA_Agaç", (0.030, 0.022, 0.018), 0.95, spec=0.12)
    m_rock = _materyal("KA_Kaya", (0.085, 0.070, 0.060), 0.92, spec=0.25)

    # zemin yuksekligi (zemin_agina gore hizli kaba arama)
    def z_at(x, y):
        r = math.hypot(x, y - CAM_Y)
        amp = 2.6 * min(1.0, max(0.0, (r - 25.0) / 120.0))
        return float(_gurultu(np.array([[x]]), np.array([[y]]), 11)[0, 0] * amp)

    agac_mes = []
    for _ in range(7):
        vs, fs = [], []
        _dal_parca(vs, fs, (0, 0, 0), (0.12, 0.05, 1.0), 2.2, 0.22, 0.10)
        _dal_parca(vs, fs, (0.05, 0.02, 1.3), (0.45, 0.18, 0.88), 1.1, 0.11, 0.18)
        _dal_parca(vs, fs, (0.02, 0.0, 1.5), (-0.40, 0.22, 0.89), 0.9, 0.09, 0.22)
        _dal_parca(vs, fs, (0.08, 0.03, 0.8), (0.22, -0.55, 0.80), 0.8, 0.09, 0.15)
        me = bpy.data.meshes.new("AgacMesh")
        me.from_pydata(vs, [], fs)
        me.update()
        agac_mes.append(me)

    kaya_mes = []
    for _ in range(5):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.5)
        ob = bpy.context.active_object
        me = ob.data
        for v in me.vertices:
            v.co *= rng.uniform(0.7, 1.3)
        me.update()
        kaya_mes.append(me)
        bpy.data.objects.remove(ob)

    # agaclar: orta plan bandi
    for i in range(38):
        side = -1 if i % 2 == 0 else 1
        x = side * rng.uniform(13.0, 70.0)
        y = rng.uniform(12.0, 95.0)
        s = rng.uniform(1.4, 3.6)
        ob = bpy.data.objects.new(f"Agac_{i:02d}", agac_mes[rng.integers(0, len(agac_mes))])
        coll.objects.link(ob)
        ob.scale = (s * rng.uniform(0.8, 1.2), s, s * rng.uniform(0.9, 1.4))
        ob.rotation_euler = (rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06), rng.uniform(0, 6.28))
        ob.location = (x, y, z_at(x, y) - 0.12 * s)
        ob.data.materials.clear()
        ob.data.materials.append(m_wood)
    # kayalar: on plan (basi buyuk olanlar kadrajin alt kenarina girer)
    for i in range(46):
        x = rng.uniform(-55, 55)
        y = rng.uniform(CAM_Y + 5, CAM_Y + 70)
        s = rng.uniform(0.22, 1.3)
        ob = bpy.data.objects.new(f"Kaya_{i:02d}", kaya_mes[rng.integers(0, len(kaya_mes))])
        coll.objects.link(ob)
        ob.scale = (s * rng.uniform(0.9, 1.5), s * rng.uniform(0.9, 1.4), s * rng.uniform(0.8, 1.5))
        ob.rotation_euler = (rng.uniform(0, 0.5), rng.uniform(0, 0.5), rng.uniform(0, 6.28))
        ob.location = (x, y, z_at(x, y) - 0.22 * s)
        ob.data.materials.clear()
        ob.data.materials.append(m_rock)


def harabe(sc, coll):
    """Koyu harabe siluetleri (y 13-26 bandi) + orta koridor temiz.

    Siluetler: ince kuleler, kırık duvarlar, egik stump'lar. Kenarlar
    pahlanir (bevel) ki alacakaranligin kenar isigi yakalasin.
    """
    rng = np.random.default_rng(5)
    m_ruin = _materyal("KA_Harabe", (0.085, 0.066, 0.052), 0.90, spec=0.32)
    nt = m_ruin.node_tree
    b = nt.nodes["Principled BSDF"]
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 4.5
    noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.038, 0.030, 0.026, 1)
    ramp.color_ramp.elements[1].color = (0.175, 0.130, 0.098, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.4
    bump.inputs["Distance"].default_value = 0.1
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    # leke leke parlaklik: alacakaranlikta yatay gunesin toz üzerindeki
    # yansimasi -> ufka dogru sicak bir isik yolu
    n3 = nt.nodes.new("ShaderNodeTexNoise")
    n3.inputs["Scale"].default_value = 0.35
    n3.inputs["Detail"].default_value = 3.0
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["From Min"].default_value = 0.35
    rr.inputs["From Max"].default_value = 0.65
    rr.inputs["To Min"].default_value = 0.52
    rr.inputs["To Max"].default_value = 0.95
    nt.links.new(n3.outputs["Fac"], rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], b.inputs["Roughness"])

    # (tur, w, d, h) — 0: kule, 1: kırık duvar, 2: egik stump
    plan = [(0, 1.6, 1.6, 13.0), (1, 5.0, 1.2, 3.2), (0, 2.2, 2.0, 8.5),
            (2, 1.4, 1.4, 5.0), (0, 1.3, 1.5, 15.5), (1, 3.4, 1.0, 2.2),
            (0, 2.6, 2.4, 6.5), (1, 6.0, 1.4, 4.0), (0, 1.8, 1.7, 11.0),
            (2, 1.1, 1.1, 7.0), (0, 2.0, 1.9, 9.5), (1, 4.2, 1.3, 2.6),
            (0, 1.5, 1.4, 12.5), (0, 2.4, 2.2, 7.8), (1, 5.5, 1.1, 3.6),
            (0, 1.7, 1.6, 10.2), (2, 1.2, 1.2, 4.2), (0, 2.1, 2.0, 6.0)]
    for i, (tur, w, d, h) in enumerate(plan):
        side = -1 if i % 2 == 0 else 1
        x = side * rng.uniform(13.5, 27.0)
        y = rng.uniform(13.0, 26.0)
        lean = rng.uniform(-0.12, 0.12) if tur == 2 else rng.uniform(-0.05, 0.05)
        # YEREL koordinatlar: obje kendi merkezinde doner (dunya merkezi degil!)
        vs = []
        egik = rng.uniform(-0.30, 0.30)
        for (zz, sc_, lx) in ((0.0, 1.0, 0.0), (h, rng.uniform(0.45, 0.9), lean * h)):
            for (dx, dy) in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)):
                vs.append((dx * sc_ + lx + rng.uniform(-0.30, 0.30),
                           dy * sc_ + rng.uniform(-0.30, 0.30),
                           zz - 0.8 + rng.uniform(-0.30, 0.30) + egik * dx))
        fs = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
              (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
              (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
        me = bpy.data.meshes.new(f"Kule{i}Mesh")
        me.from_pydata(vs, [], fs)
        me.update()
        ob = bpy.data.objects.new(f"Kule_{i:02d}", me)
        coll.objects.link(ob)
        ob.location = (x, y, 0.0)
        ob.data.materials.append(m_ruin)
        ob.rotation_euler = (0, 0, rng.uniform(0, math.pi))
        bv = ob.modifiers.new("Pah", "BEVEL")
        bv.width = 0.07
        bv.segments = 1
        bv.limit_method = "ANGLE"


# ------------------------------------------------------------------ kus mesh
def _delta_mesh(ad, boy, span, kalinlik, kuyruk=0.95, sweep=0.12):
    """Ucgen delta kus: 6 vertex, 8 yuz. OLCEK MESH VERISINDE (birim degil!)."""
    h = boy * 0.5
    vs = [
        (h * 1.00, 0.0, 0.0),                    # 0 burun
        (-h * kuyruk, 0.0, 0.0),                 # 1 kuyruk
        (-h * sweep, span * 0.5, 0.0015),        # 2 sol kanat ucu
        (-h * sweep, -span * 0.5, 0.0015),       # 3 sag kanat ucu
        (-h * 0.10, 0.0, kalinlik * 0.75),       # 4 sirt
        (-h * 0.10, 0.0, -kalinlik * 0.45),      # 5 karin
    ]
    fs = [(0, 2, 4), (0, 4, 3), (1, 4, 2), (1, 3, 4),
          (0, 4, 2), (0, 3, 4), (1, 2, 4), (1, 4, 3)]
    me = bpy.data.meshes.new(ad)
    me.from_pydata(vs, [], fs)
    me.update()
    me.validate()
    return me


# ------------------------------------------------------------------ kamera
def _vfo(olcek_y=540, olcek_x=960):
    return 2.0 * math.atan((SENSOR_W * (olcek_y / olcek_x) * 0.5) / LENS)


def _look_euler(nokta, kamera, egim_pay):
    """look_at: kamera yonu AÇIKÇA hesaplanir (TRACK_TO constraint YOK)."""
    d = np.array(nokta) - np.array(kamera)
    beta = math.atan2(d[0], d[1])          # +Y ekseninden sag'a dogru azimut
    return (math.pi / 2 - egim_pay, 0.0, -beta)


# ------------------------------------------------------------------ ana
def main():
    opts = args()
    sc = bpy.context.scene
    t0 = time.time()
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)

    coll = bpy.data.collections.new("Suru")
    sc.collection.children.link(coll)

    n = PREVIEW_BIRDS if opts["preview"] else N_BIRDS
    f = boids.Flock(n=n, seed=7, domain=DOMAIN)

    def hawk(t):
        u = (t - ANIM_T0) / (CLIP / FPS)
        return np.array([-16.0 + 32.0 * u,
                         2.2 * math.sin(u * 3.1),
                         4.0 + 1.4 * math.sin(u * 2.2)])

    # simulasyon onbellegi: ayni tohum/alan icin tekrar hesaplamayalim
    cache = os.path.join("/tmp", f"murmur_sim_{n}_{DOMAIN:.1f}_v{SIM_VER}.npz")
    if os.path.exists(cache) and "--yeni" not in sys.argv:
        z = np.load(cache)
        rec_t, rec_p, rec_h = z["t"], z["p"], z["h"]
        print(f"[sim] onbellekten {rec_p.shape} ({time.time() - t0:.1f}s)", flush=True)
    else:
        rec = f.run(SIM_T, dt=0.01, record_every=4, hawk_path=hawk)
        rec_t = np.array([r[0] for r in rec])
        rec_p = np.stack([r[1] for r in rec])
        rec_h = np.stack([r[2] for r in rec])
        np.savez_compressed(cache, t=rec_t, p=rec_p, h=rec_h)
        print(f"[sim] kayit {rec_p.shape} ({time.time() - t0:.1f}s)", flush=True)

    # ---------------------------------------------------------- zaman ornekleme
    def orn(t):
        i = int(np.clip(np.searchsorted(rec_t, t), 1, len(rec_t) - 1))
        t0_, t1_ = rec_t[i - 1], rec_t[i]
        u = (t - t0_) / max(t1_ - t0_, 1e-9)
        return rec_p[i - 1], rec_p[i], u, rec_h[i - 1], rec_h[i]

    kare_t = ANIM_T0 + np.arange(CLIP + 1) / FPS
    orn_kare = [orn(t) for t in kare_t]

    # yumusak hiz (0.6 s pencere) — kamera titremesin.
    # Hedef: kusun bulutunun canli kutle medyani (gorunur kutlenin merkezi)
    med = np.array([np.median(p0 + (p1 - p0) * u, axis=0) for (p0, p1, u, _, _) in orn_kare])
    k = 7
    ker = np.ones(k) / k
    com_y = np.convolve(med[:, 1], ker, mode="same")
    com_x = np.convolve(med[:, 0], ker, mode="same")
    com_x[:k // 2] = com_x[k // 2]
    com_x[-(k // 2):] = com_x[-(k // 2 + 1)]
    com_y[:k // 2] = com_y[k // 2]
    com_y[-(k // 2):] = com_y[-(k // 2 + 1)]

    # ---------------------------------------------------------- kamera
    cam_data = bpy.data.cameras.new("Kamera")
    cam_data.lens = LENS
    cam_data.sensor_width = SENSOR_W
    cam_data.clip_end = 200000.0
    cam = bpy.data.objects.new("Kamera", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam

    vfo = _vfo()
    egim = math.radians((HORIZON_V - 0.5) * math.degrees(vfo))   # ufuk ust 1/3
    # kamera yatay azimut: sürü merkezine gore (ortalama)
    cam_x0 = -2.0
    cam_y0 = CAM_Y
    azimutlar = []
    for idx in range(CLIP + 1):
        cx = cam_x0 + CAM_DRIFT * idx / CLIP
        cy = cam_y0 + 0.9 * idx / CLIP
        cz = CAM_Z + CAM_RISE * idx / CLIP
        azimutlar.append(math.atan2(com_x[idx] - cx, com_y[idx] - cy))
    azimutlar = np.array(azimutlar)
    ortalama_azim = float(np.mean(azimutlar))

    act_c, strip_c = _yeni_aksiyon("KameraAksiyon")
    cam.animation_data_create()
    cam.animation_data.action = act_c
    slot_c = act_c.slots.new(id_type="OBJECT", name="Kamera")
    cam.animation_data.action_slot = slot_c

    kareler = list(range(1, CLIP + 2))
    for ch in range(3):
        vals = []
        for idx in range(CLIP + 1):
            cx = cam_x0 + CAM_DRIFT * idx / CLIP
            cy = cam_y0 + 0.9 * idx / CLIP
            cz = CAM_Z + CAM_RISE * idx / CLIP
            vals.append((cx, cy, cz)[ch])
        _fcurve(strip_c, slot_c, "location", ch, kareler, vals)
    for ch in range(3):
        vals = []
        for idx in range(CLIP + 1):
            cx = cam_x0 + CAM_DRIFT * idx / CLIP
            cy = cam_y0 + 0.9 * idx / CLIP
            cz = CAM_Z + CAM_RISE * idx / CLIP
            e = _look_euler((com_x[idx], com_y[idx], 5.0), (cx, cy, cz), egim)
            vals.append(e[ch])
        _fcurve(strip_c, slot_c, "rotation_euler", ch, kareler, np.unwrap(vals))
    _hepsi_lineer(cam)

    # ---------------------------------------------------------- gokyuzu
    gokyuzu(sc, ortalama_azim)
    zemin_ob = zemin(sc, coll)
    calilik(sc, coll, zemin_ob)
    harabe(sc, coll)

    # atmosfer: YERE YATAN INCE SIS KATMANI -> ufka dogru sicak derinlik,
    # yakinda koyu on plan. Kalin bir hacim gokyuzunu de yikardi.
    m_hava = bpy.data.materials.new("Hava")
    m_hava.use_nodes = True
    nt = m_hava.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vol = nt.nodes.new("ShaderNodeVolumeScatter")
    vol.inputs["Color"].default_value = (1.0, 0.52, 0.26, 1.0)
    vol.inputs["Density"].default_value = 0.0005
    vol.inputs["Anisotropy"].default_value = 0.45
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 700, 2))
    hava = bpy.context.active_object
    hava.name = "Hava"
    hava.scale = (2600.0, 2600.0, 9.0)
    hava.data.materials.append(m_hava)
    for c in list(hava.users_collection):
        c.objects.unlink(hava)
    coll.objects.link(hava)
    hava.visible_shadow = False

    # ---------------------------------------------------------- kuslar
    m_bird = _materyal("Kus", (0.035, 0.027, 0.022), 0.55, spec=0.55)
    mesh = _delta_mesh("KusMesh", BIRD_LEN, BIRD_SPAN, 0.026)
    mesh.materials.append(m_bird)

    n_birds = rec_p.shape[1]
    act_b, strip_b = _yeni_aksiyon("SuruAksiyon")

    # hiz (kayit turevinden, yumusatilmis) -> ucus yonu
    dt_rec = float(rec_t[1] - rec_t[0])
    vel = np.zeros_like(rec_p)
    vel[1:-1] = (rec_p[2:] - rec_p[:-2]) / (2 * dt_rec)
    vel[0] = vel[1]
    vel[-1] = vel[-2]
    w = np.array([0.25, 0.25, 0.25, 0.25])
    for ax in range(3):
        vel[..., ax] = np.convolve(vel[..., ax].ravel(), w, mode="same").reshape(vel[..., ax].shape)

    def yon(u_, v_):
        """v = hiz; kuşun +X ekseni hıza bakar, +Z dünya yukarısına yakın."""
        n = np.linalg.norm(u_, axis=-1, keepdims=True)
        u_ = u_ / np.maximum(n, 1e-9)
        up = np.array([0.0, 0.0, 1.0])
        zl = up - u_ * np.sum(u_ * up, axis=-1, keepdims=True)
        nz = np.linalg.norm(zl, axis=-1, keepdims=True)
        zl = np.where(nz < 1e-5, np.array([0.0, 0.0, 1.0]), zl / np.maximum(nz, 1e-12))
        yl = np.cross(zl, u_)
        m00, m10, m20 = u_[..., 0], u_[..., 1], u_[..., 2]
        m21, m22 = zl[..., 1], zl[..., 2]
        rx = np.arctan2(m21, m22)
        ry = np.arcsin(np.clip(-m20, -1, 1))
        rz = np.arctan2(m10, m00)
        return np.stack([rx, ry, rz], axis=-1)

    # kayit araliginda her kare icin konum + yon
    tum_p = []
    tum_e = []
    for idx in range(CLIP + 1):
        p0, p1, u, _, _ = orn_kare[idx]
        pos = p0 + (p1 - p0) * u
        v0 = vel[max(int(np.searchsorted(rec_t, kare_t[idx])) - 1, 0)]
        v1 = vel[min(int(np.searchsorted(rec_t, kare_t[idx])), len(vel) - 1)]
        vv = v0 + (v1 - v0) * u
        tum_p.append(pos)
        tum_e.append(yon(vv, None))
    tum_p = np.stack(tum_p)      # (K, N, 3)
    tum_e = np.stack(tum_e)      # (K, N, 3)
    for ax in range(3):
        tum_e[..., ax] = np.unwrap(tum_e[..., ax], axis=0)

    for i in range(n_birds):
        ob = bpy.data.objects.new(f"Kus_{i:04d}", mesh)
        coll.objects.link(ob)
        ob.animation_data_create()
        ob.animation_data.action = act_b
        slot = act_b.slots.new(id_type="OBJECT", name=f"Kus{i}")
        ob.animation_data.action_slot = slot
        for ch in range(3):
            _fcurve(strip_b, slot, "location", ch, kareler, tum_p[:, i, ch])
        for ch in range(3):
            _fcurve(strip_b, slot, "rotation_euler", ch, kareler, tum_e[:, i, ch])
    print(f"[kus] {n_birds} kus eylem yazildi ({time.time() - t0:.1f}s)", flush=True)

    # ---------------------------------------------------------- sahin
    m_hawk = _materyal("Sahin", (0.040, 0.026, 0.018), 0.42, spec=0.6)
    hmesh = _delta_mesh("SahinMesh", HAWK_LEN, HAWK_SPAN, 0.055, kuyruk=0.55, sweep=0.26)
    hmesh.materials.append(m_hawk)
    hawk_ob = bpy.data.objects.new("Sahin", hmesh)
    coll.objects.link(hawk_ob)
    act_h, strip_h = _yeni_aksiyon("SahinAksiyon")
    hawk_ob.animation_data_create()
    hawk_ob.animation_data.action = act_h
    slot_h = act_h.slots.new(id_type="OBJECT", name="Sahin")
    hawk_ob.animation_data.action_slot = slot_h
    hpos = []
    heu = []
    for idx in range(CLIP + 1):
        p0, p1, u, h0, h1 = orn_kare[idx]
        hpos.append(h0 + (h1 - h0) * u)
        # sahin hizi: yolun analitik turevi (hawk(t) fonksiyonundan)
        eps = 1e-3
        ha = hawk(kare_t[idx] - eps)
        hb = hawk(kare_t[idx] + eps)
        heu.append((hb - ha) / (2 * eps))
    heu = np.array(heu)
    hv = np.gradient(heu, 1.0 / FPS, axis=0)     # yanal ivme -> banka
    for ch in range(3):
        _fcurve(strip_h, slot_h, "location", ch, kareler, np.array(hpos)[:, ch])
    eu = []
    for idx in range(CLIP + 1):
        e = yon(np.array([heu[idx][0], heu[idx][1], heu[idx][2]]), None)
        eu.append(e)
    eu = np.array(eu)
    bank = np.clip(-hv[:, 1] * 0.55, -0.55, 0.55)
    for ch in range(3):
        vals = eu[:, ch].copy()
        if ch == 0:
            vals = vals + bank
        _fcurve(strip_h, slot_h, "rotation_euler", ch, kareler, np.unwrap(vals))
    print(f"[sahin] yol uzunlugu {np.linalg.norm(heu, axis=1).mean():.2f} birim/s", flush=True)

    # ---------------------------------------------------------- olcum
    if opts["olc"] or True:
        from bpy_extras.object_utils import world_to_camera_view
        sc.frame_set(CLIP // 2)
        bpy.context.view_layer.update()
        p = tum_p[CLIP // 2]
        uv = [world_to_camera_view(sc, cam, bpy.data.objects[f"Kus_{i:04d}"].matrix_world.translation)
              for i in range(0, n_birds, max(1, n_birds // 400))]
        uu = np.array([q.x for q in uv])
        vv = np.array([q.y for q in uv])
        uf = np.array([q.z for q in uv])
        ufk = np.array([world_to_camera_view(sc, cam, Vector((0, 100000, cam.location.z))).x])
        ufvk = np.array([world_to_camera_view(sc, cam, Vector((0, 100000, cam.location.z))).y])
        print(f"[olc] ufuk v={ufvk[0]:.3f} (hedef {HORIZON_V})")
        print(f"[olc] kus bulutu u[{uu.min():.3f},{uu.max():.3f}] v[{vv.min():.3f},{vv.max():.3f}] "
              f"yukseklik %{100 * (vv.max() - vv.min()):.1f} derinlik {uf.min():.1f}..{uf.max():.1f}")
        hp = world_to_camera_view(sc, cam, hawk_ob.matrix_world.translation)
        print(f"[olc] sahin u={hp.x:.3f} v={hp.y:.3f} derinlik {hp.z:.1f}")
        print(f"[olc] kamera {tuple(round(v,2) for v in cam.location)} azimut {np.degrees(ortalama_azim):.2f}")

    # ---------------------------------------------------------- render ayarlari
    sc.frame_start = 1
    sc.frame_end = CLIP + 1
    sc.render.fps = FPS
    sc.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU"
    except Exception as e:
        print("[cycles] GPU yok:", e)
    if opts["preview"]:
        sc.render.resolution_x, sc.render.resolution_y = 640, 360
        sc.cycles.samples = 40
    else:
        sc.render.resolution_x, sc.render.resolution_y = 960, 540
        sc.cycles.samples = 96
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.cycles.transmission_bounces = 2
    sc.cycles.volume_bounces = 1
    sc.cycles.transparent_max_bounces = 4
    sc.view_settings.view_transform = "AgX"
    for look in ("AgX - Medium High Contrast", "Medium High Contrast", "AgX - Base Contrast"):
        try:
            sc.view_settings.look = look
            print("[renk] look:", look)
            break
        except TypeError:
            continue
    sc.view_settings.exposure = 0.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.film_transparent = False

    if opts["olc"]:
        return

    out = os.path.join(REPO, "renders", "preview" if opts["preview"] else "murmur")
    os.makedirs(out, exist_ok=True)
    frames = PREVIEW_FRAMES if opts["preview"] else range(1, CLIP + 2)
    for fr in frames:
        fp = os.path.join(out, f"frame_{fr:04d}.png")
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            print(f"[kare {fr}] var, atlandi", flush=True)
            continue
        sc.frame_set(fr)
        sc.render.filepath = fp
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[kare {fr}] bitti ({time.time() - t1:.1f}s)", flush=True)
    print(f"== BİTTİ -> {out}  ({time.time() - t0:.1f}s) ==")


main()
