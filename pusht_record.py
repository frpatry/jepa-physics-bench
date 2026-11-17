"""
ENREGISTREUR DE GESTES Push-T (17 nov. 2026, demande user) — une fenêtre où l'UTILISATEUR joue Push-T à la souris ; ses gestes sont enregistrés pour être
rejoués tels quels sur le bébé gelé (av_follow_diag : « interroger l'imagination pour le bon geste, faire le geste, et ainsi de suite », sans décision).
Même physique que le test (av_plan0.Push0, monde vécu du bébé v3c : frottement 0.5, rémanence 0.5, physique de forme, T de taille normale) ; aucun modèle chargé
(torch n'est pas nécessaire : il est remplacé par un bouchon à l'import). Enregistrements : recordings/pusht_ep<N>_<date>.json (gestes + états, pour vérifier le rejeu).
  python3 pusht_record.py --port 8765        puis ouvrir http://localhost:8765
"""
import argparse, base64, importlib.abc, importlib.machinery, json, os, sys, time, types
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np


class _Any:                                     # bouchon : tout attribut / appel rend un bouchon ; utilisé comme décorateur, rend la fonction
    def __getattr__(s, k):
        if k.startswith("__"): raise AttributeError(k)
        return _Any()
    def __call__(s, *a, **k): return a[0] if len(a) == 1 and callable(a[0]) and not k and not isinstance(a[0], _Any) else _Any()
    def __getitem__(s, k): return _Any()
    def __iter__(s): return iter(())
    def __mro_entries__(s, bases): return (object,)
class _Stub(types.ModuleType):
    def __getattr__(s, k):
        if k.startswith("__"): raise AttributeError(k)
        return _Any()
class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    ROOTS = ("torch", "lejepa", "scipy", "sklearn", "matplotlib")
    def find_spec(s, name, path, target=None): return importlib.machinery.ModuleSpec(name, s, is_package=True) if name.split(".")[0] in s.ROOTS else None
    def create_module(s, spec): m = _Stub(spec.name); m.__path__ = []; return m
    def exec_module(s, m): pass
sys.meta_path.insert(0, _Finder())
import av_plan0                                  # noqa: E402 (après le bouchon)
from av_plan0 import Push0, VMAX                 # noqa: E402

WORLD = dict(damp=0.5, echo=0.5, shape_phys=True, obj_scale=1.0); LMAX = 600; H = 32
S = dict(env=None, ep=0, cov=[], saved="")


def new_env(ep):
    S["env"] = Push0(ep, task="pusht", L=LMAX + 2, **WORLD); S["ep"] = ep; S["cov"] = [S["env"].coverage()]; S["saved"] = ""


def b64(img): return base64.b64encode((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes()).decode()


def state():
    e = S["env"]; return dict(ep=S["ep"], t=int(e.t), img=b64(e.frame(e.P, e.Hp)), goal=b64(e.frame(e.g, e.Hp, e.g_ang, hand=False)), cov=S["cov"][-1], covmax=max(S["cov"]),
                              hand=[float(e.Hp[0]), float(e.Hp[1])], full=e.t >= LMAX, saved=S["saved"])


def save():
    e = S["env"]; t = int(e.t)
    if t < 1: return ""
    os.makedirs("recordings", exist_ok=True); fn = f"recordings/pusht_ep{S['ep']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(dict(ep=S["ep"], world=WORLD, n=t, cmd=e.CMD[1:t + 1].tolist(), pos=e.POS[:t + 1].tolist(), ang=[float(x) for x in e.ANG[:t + 1]], hand=e.HAND[:t + 1].tolist(), cov=S["cov"],
                   goal=[float(e.g[0]), float(e.g[1]), float(e.g_ang)]), open(fn, "w"))
    S["saved"] = fn; return fn


PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Push-T — enregistreur</title><style>
body{font-family:-apple-system,sans-serif;background:#161616;color:#e6e6e6;margin:16px;max-width:560px}
canvas{image-rendering:pixelated;border:1px solid #444;cursor:crosshair;touch-action:none;width:512px;height:512px;max-width:100%}
button{font-size:14px;padding:6px 12px;margin:6px 6px 0 0;background:#2d2d2d;color:#e6e6e6;border:1px solid #555;border-radius:6px;cursor:pointer}
#info{font-size:15px;margin:8px 0;font-variant-numeric:tabular-nums}.muted{color:#9a9a9a;font-size:13px;line-height:1.4}#saved{color:#7fd67f;font-size:13px;min-height:18px}
</style></head><body>
<div id="info">…</div>
<canvas id="c" width="512" height="512"></canvas>
<div><button id="again">Recommencer cette partie</button><button id="next">Partie suivante</button><button id="save">Enregistrer</button></div>
<div id="saved"></div>
<p class="muted">Maintiens le bouton de la souris enfoncé : la main (carré blanc) suit le pointeur, et le temps n'avance que pendant ce temps-là.
Pousse le T sur sa silhouette claire (le but). Relâche pour réfléchir. « Enregistrer » garde tes gestes ; changer de partie enregistre aussi.</p>
<script>
const c=document.getElementById('c'),x=c.getContext('2d'),off=document.createElement('canvas');off.width=off.height=32;const ox=off.getContext('2d');
let down=false,mx=.5,my=.5,busy=false,goal=null;
function img(b){const s=atob(b),d=ox.createImageData(32,32);for(let i=0;i<1024;i++){d.data[4*i]=s.charCodeAt(3*i);d.data[4*i+1]=s.charCodeAt(3*i+1);d.data[4*i+2]=s.charCodeAt(3*i+2);d.data[4*i+3]=255}return d}
function draw(s){x.imageSmoothingEnabled=false;ox.putImageData(img(s.img),0,0);x.globalAlpha=1;x.drawImage(off,0,0,512,512);
 ox.putImageData(img(s.goal),0,0);x.globalAlpha=.3;x.globalCompositeOperation='lighter';x.drawImage(off,0,0,512,512);x.globalCompositeOperation='source-over';x.globalAlpha=1;
 document.getElementById('info').textContent='Partie '+s.ep+'  |  geste '+s.t+(s.full?' (maximum atteint)':'')+'  |  couverture du but '+Math.round(s.cov*100)+' %  (max '+Math.round(s.covmax*100)+' %)';
 document.getElementById('saved').textContent=s.saved?('Enregistré : '+s.saved):''}
async function call(p,b){const r=await fetch(p,{method:'POST',body:JSON.stringify(b||{})});draw(await r.json())}
function pos(e){const r=c.getBoundingClientRect();mx=(e.clientX-r.left)/r.width;my=(e.clientY-r.top)/r.height}
c.addEventListener('pointerdown',e=>{down=true;pos(e);c.setPointerCapture(e.pointerId)});c.addEventListener('pointermove',pos);
c.addEventListener('pointerup',()=>down=false);c.addEventListener('pointercancel',()=>down=false);
setInterval(async()=>{if(!down||busy)return;busy=true;try{await call('/step',{x:mx,y:my})}finally{busy=false}},90);
document.getElementById('again').onclick=()=>call('/reset',{d:0});document.getElementById('next').onclick=()=>call('/reset',{d:1});document.getElementById('save').onclick=()=>call('/save');
call('/state');
</script></body></html>"""


class Hd(BaseHTTPRequestHandler):
    def log_message(s, *a): pass
    def _send(s, body, ct):
        s.send_response(200); s.send_header("Content-Type", ct); s.send_header("Content-Length", str(len(body))); s.end_headers(); s.wfile.write(body)
    def do_GET(s): s._send(PAGE.encode(), "text/html; charset=utf-8")
    def do_POST(s):
        n = int(s.headers.get("Content-Length", 0)); q = json.loads(s.rfile.read(n) or b"{}"); e = S["env"]
        if s.path == "/step" and e.t < LMAX:
            tg = np.array([float(q["x"]), float(q["y"])], np.float32); e.step(np.clip(tg - e.Hp, -VMAX, VMAX).astype(np.float32)); S["cov"].append(e.coverage()); S["saved"] = ""
        elif s.path == "/reset":
            if e.t >= 5 and not S["saved"]: save()
            new_env(S["ep"] + int(q.get("d", 0)))
        elif s.path == "/save": save()
        s._send(json.dumps(state()).encode(), "application/json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8765); p.add_argument("--ep", type=int, default=0); a = p.parse_args()
    av_plan0.T = 16; new_env(a.ep); print(f"enregistreur Push-T : http://localhost:{a.port}", flush=True); HTTPServer(("127.0.0.1", a.port), Hd).serve_forever()
