/* FoulObs — détection des personnes sur l'appareil (TensorFlow.js + COCO-SSD, gratuit, sans compte, sans envoi d'image).
   Produit le même objet que l'ancienne analyse : {estimate, quality, quality_note, people:[{x,y,posture,arms_up,moving,group}]} */
const FoulDetect = (() => {
  let model = null;
  async function load(onStatus) {
    if (model) return model;
    onStatus && onStatus("Chargement du modèle de détection (une seule fois, environ 15 Mo)…");
    await tf.ready();
    model = await cocoSsd.load({ base: "mobilenet_v2" });
    return model;
  }
  const area = b => b.w * b.h;
  function iou(a, b) {
    const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y), x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
    const i = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
    return { iou: i / (area(a) + area(b) - i || 1), inSmall: i / (Math.min(area(a), area(b)) || 1) };
  }
  async function detectOn(source, sx, sy, sw, sh, out) {
    const k = Math.min(1, 1024 / Math.max(sw, sh));
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(sw * k)); c.height = Math.max(1, Math.round(sh * k));
    c.getContext("2d").drawImage(source, sx, sy, sw, sh, 0, 0, c.width, c.height);
    const preds = await model.detect(c, 300, 0.33);
    preds.filter(p => p.class === "person").forEach(p => {
      const [x, y, w, h] = p.bbox;
      out.push({ x: sx + x / k, y: sy + y / k, w: w / k, h: h / k, s: p.score });
    });
  }
  /* image entière + 9 tuiles qui se chevauchent : les petites silhouettes au loin sont mieux vues */
  async function people(source, onStatus) {
    await load(onStatus);
    const W = source.naturalWidth || source.videoWidth || source.width, H = source.naturalHeight || source.videoHeight || source.height;
    const boxes = [];
    await detectOn(source, 0, 0, W, H, boxes);
    if (Math.max(W, H) >= 700) {
      const n = 2, tw = W / n * 1.3, th = H / n * 1.3;
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        onStatus && onStatus(`Analyse sur votre appareil… ${j * n + i + 1}/${n * n}`);
        const sx = Math.min(W - tw, Math.max(0, (i + .5) * W / n - tw / 2)), sy = Math.min(H - th, Math.max(0, (j + .5) * H / n - th / 2));
        await detectOn(source, sx, sy, tw, th, boxes);
      }
    }
    boxes.sort((a, b) => b.s - a.s);
    const keep = [];
    for (const b of boxes) if (keep.every(k => { const r = iou(k, b); return r.iou < .4 && r.inSmall < .7; })) keep.push(b);
    return { boxes: keep, W, H };
  }
  /* groupes : personnes proches (distance < ~0,9 fois la taille des silhouettes) */
  function groups(P) {
    const par = P.map((_, i) => i), f = i => par[i] === i ? i : (par[i] = f(par[i]));
    for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
      const a = P[i], b = P[j], d = Math.hypot(a.cx - b.cx, a.cy - b.cy);
      if (d < .9 * (a.h + b.h) / 2) par[f(i)] = f(j);
    }
    const size = {}; P.forEach((_, i) => { const r = f(i); size[r] = (size[r] || 0) + 1; });
    const id = {}; let n = 0;
    return P.map((_, i) => { const r = f(i); if (size[r] < 2) return 0; if (!id[r]) id[r] = ++n; return id[r]; });
  }
  function toResult({ boxes, W, H }) {
    const P = boxes.map(b => ({ cx: b.x + b.w / 2, cy: b.y + b.h / 2, w: b.w, h: b.h }));
    const g = groups(P);
    const people = P.map((p, i) => {
      const r = p.h / Math.max(1, p.w);
      return { x: p.cx / W, y: p.cy / H, posture: r > 1.7 ? "debout" : r < 1.05 ? "assis" : "indetermine", arms_up: false, moving: false, group: g[i] };
    });
    const hs = P.map(p => p.h / H).sort((a, b) => a - b), med = hs.length ? hs[hs.length >> 1] : 0;
    let quality = "bonne", note = "Silhouettes bien visibles.";
    if (!P.length) { quality = "faible"; note = "Aucune personne détectée : image trop sombre, trop lointaine ou sans foule."; }
    else if (med < .05) { quality = "faible"; note = "Silhouettes très petites : la foule est probablement sous-comptée."; }
    else if (med < .1 || P.length > 120) { quality = "moyenne"; note = "Foule dense ou lointaine : des personnes cachées peuvent manquer."; }
    return { estimate: people.length, quality, quality_note: note, people };
  }

  /* ---------- comptage des foules denses : CSRNet (carte de densité) ---------- */
  let sess = null;
  async function loadCrowd(onStatus) {
    if (sess) return sess;
    onStatus && onStatus("Chargement du modèle de comptage de foule (16 Mo, une seule fois)…");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    /* calcul en arrière-plan (la page ne gèle pas) et sur plusieurs cœurs quand le navigateur le permet */
    ort.env.wasm.proxy = true;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    sess = await ort.InferenceSession.create("csrnet-q8.onnx", { executionProviders: ["wasm"] });
    return sess;
  }
  async function density(source, onStatus) {
    await loadCrowd(onStatus);
    const W = source.naturalWidth || source.videoWidth || source.width, H = source.naturalHeight || source.videoHeight || source.height;
    const k = Math.min(1, 1024 / Math.max(W, H)), w = Math.max(8, Math.floor(W * k / 8) * 8), h = Math.max(8, Math.floor(H * k / 8) * 8);
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    const g = c.getContext("2d"); g.drawImage(source, 0, 0, w, h);
    const px = g.getImageData(0, 0, w, h).data, n = w * h, x = new Float32Array(3 * n);
    const M = [0.485, 0.456, 0.406], S = [0.229, 0.224, 0.225];
    for (let i = 0; i < n; i++) for (let ch = 0; ch < 3; ch++) x[ch * n + i] = (px[i * 4 + ch] / 255 - M[ch]) / S[ch];
    onStatus && onStatus("Comptage de la foule sur votre appareil (15 à 60 secondes)…");
    const out = await sess.run({ image: new ort.Tensor("float32", x, [1, 3, h, w]) });
    const d = out.density, gh = d.dims[2], gw = d.dims[3], data = d.data;
    let sum = 0; for (let i = 0; i < data.length; i++) sum += Math.max(0, data[i]);
    return { data, gw, gh, sum };
  }
  /* repères approximatifs à partir de la carte de densité (pour les zones et l'affichage) */
  function pointsFromDensity({ data, gw, gh }) {
    const pts = []; let acc = 0;
    for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
      acc += Math.max(0, data[j * gw + i]);
      while (acc >= 1) { acc -= 1; const r = ((pts.length * 0.618) % 1) - .5; pts.push({ x: (i + .5 + r * .6) / gw, y: (j + .5 - r * .6) / gh, posture: "indetermine", arms_up: false, moving: false, group: 0 }); }
    }
    return pts;
  }
  /* analyse complète : détecteur (repères précis) + CSRNet (foules denses) */
  async function analyse(source, onStatus) {
    const det = toResult(await people(source, onStatus));
    let den = null;
    try { den = await density(source, onStatus); } catch (e) { console.warn("CSRNet indisponible", e); }
    if (!den) return det;
    const est = Math.round(den.sum);
    if (est < 30 || det.estimate >= est * .7 || (est < 60 && det.estimate >= est * .5)) {   /* petit groupe : le détecteur est plus précis */
      if (est >= 30 && det.estimate >= est * .7) det.estimate = Math.max(det.estimate, est);
      return det;
    }
    const pts = pointsFromDensity(den);
    return { estimate: est, quality: est > 800 ? "moyenne" : "bonne",
      quality_note: "Foule dense : nombre estimé par carte de densité (marge d'environ 10 %). Repères approximatifs.",
      people: pts, dense: true, detected: det.estimate };
  }
  return { load, people, toResult, analyse };
})();
