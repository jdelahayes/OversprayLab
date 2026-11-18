/* OversprayLab — palette generator of the palette assistant, and the colour maths it shares with
 * the page. No DOM here: everything is a pure function of its arguments (same seed, same result).
 *
 * A proposal gives one colour per layer. A stencil reads through its values, so the lightness of
 * the layers is planned first (kept apart from each other and from the support), then hues and
 * chroma are chosen freely, in OKLCH. Many candidates are drawn, scored, and the best ones that
 * really differ from each other are kept.
 */
(function (root) {
  "use strict";
  const RAD = Math.PI / 180;
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const mix = (a, b, t) => a + (b - a) * t;
  const wrap = h => ((h % 360) + 360) % 360;
  const hueDelta = (from, to) => ((to - from + 540) % 360) - 180;   // shortest way, -180..180
  const hexRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const rgbHex = c => "#" + c.map(v => clamp(Math.round(v), 0, 255).toString(16).padStart(2, "0")).join("");

  // ---------------- CIE Lab, CIEDE2000, optimal assignment ----------------
  const srgbLab = hex => {   // sRGB hex -> CIE Lab (D65)
    const [x, y, z] = (([r, g, b]) => [[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]]
      .map(([a, c, d]) => a * r + c * g + d * b))(hexRgb(hex).map(v => (v /= 255) <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4));
    const f = t => t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116;
    const fx = f(x / .95047), fy = f(y), fz = f(z / 1.08883);
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
  };
  function deltaE00([L1, a1, b1], [L2, a2, b2]) {   // CIEDE2000 colour difference
    const rad = Math.PI / 180, C7 = c => (c ** 7 / (c ** 7 + 25 ** 7)) ** .5;
    const Cm = (Math.hypot(a1, b1) + Math.hypot(a2, b2)) / 2, G = .5 * (1 - C7(Cm));
    const A1 = a1 * (1 + G), A2 = a2 * (1 + G), C1 = Math.hypot(A1, b1), C2 = Math.hypot(A2, b2);
    const hue = (b, a) => { const h = Math.atan2(b, a) / rad; return h < 0 ? h + 360 : h; };
    const h1 = hue(b1, A1), h2 = hue(b2, A2);
    let dh = C1 * C2 === 0 ? 0 : h2 - h1;
    if (dh > 180) dh -= 360; else if (dh < -180) dh += 360;
    const dL = L2 - L1, dC = C2 - C1, dH = 2 * Math.sqrt(C1 * C2) * Math.sin(dh * rad / 2);
    const Lm = (L1 + L2) / 2, Cp = (C1 + C2) / 2;
    let hm = h1 + h2;
    if (C1 * C2 !== 0) hm = Math.abs(h1 - h2) <= 180 ? hm / 2 : (hm < 360 ? (hm + 360) / 2 : (hm - 360) / 2);
    const T = 1 - .17 * Math.cos((hm - 30) * rad) + .24 * Math.cos(2 * hm * rad) + .32 * Math.cos((3 * hm + 6) * rad) - .2 * Math.cos((4 * hm - 63) * rad);
    const SL = 1 + .015 * (Lm - 50) ** 2 / Math.sqrt(20 + (Lm - 50) ** 2), SC = 1 + .045 * Cp, SH = 1 + .015 * Cp * T;
    const RT = -2 * C7(Cp) * Math.sin(60 * Math.exp(-(((hm - 275) / 25) ** 2)) * rad);
    return Math.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2 + RT * (dC / SC) * (dH / SH));
  }
  // Minimum-cost assignment (Hungarian algorithm), cost[n][m] with n <= m: column of each row
  function assignMin(cost) {
    const n = cost.length, m = cost[0].length, u = Array(n + 1).fill(0), v = Array(m + 1).fill(0);
    const p = Array(m + 1).fill(0), way = Array(m + 1).fill(0);
    for (let i = 1; i <= n; i++) {
      p[0] = i; let j0 = 0;
      const minv = Array(m + 1).fill(Infinity), used = Array(m + 1).fill(false);
      do {
        used[j0] = true; const i0 = p[j0]; let delta = Infinity, j1 = 0;
        for (let j = 1; j <= m; j++) if (!used[j]) {
          const c = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (c < minv[j]) { minv[j] = c; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
        for (let j = 0; j <= m; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
        j0 = j1;
      } while (p[j0]);
      do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    const col = Array(n);
    for (let j = 1; j <= m; j++) if (p[j]) col[p[j] - 1] = j - 1;
    return col;
  }

  // ---------------- OKLab / OKLCH ----------------
  const toLin = v => (v /= 255) <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
  const fromLin = v => 255 * (v <= .0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - .055);
  function rgbOklab(rgb) {
    const [r, g, b] = rgb.map(toLin);
    const l = Math.cbrt(.4122214708 * r + .5363325363 * g + .0514459929 * b);
    const m = Math.cbrt(.2119034982 * r + .6806995451 * g + .1073969566 * b);
    const s = Math.cbrt(.0883024619 * r + .2817188376 * g + .6299787005 * b);
    return [.2104542553 * l + .793617785 * m - .0040720468 * s,
      1.9779984951 * l - 2.428592205 * m + .4505937099 * s,
      .0259040371 * l + .7827717662 * m - .808675766 * s];
  }
  function oklabLinear(L, a, b) {
    const l = (L + .3963377774 * a + .2158037573 * b) ** 3;
    const m = (L - .1055613458 * a - .0638541728 * b) ** 3;
    const s = (L - .0894841775 * a - 1.291485548 * b) ** 3;
    return [4.0767416621 * l - 3.3077115913 * m + .2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - .3413193965 * s,
      -.0041960863 * l - .7034186147 * m + 1.707614701 * s];
  }
  const hexOklab = hex => rgbOklab(hexRgb(hex));
  const labLch = ([L, a, b]) => [L, Math.hypot(a, b), wrap(Math.atan2(b, a) / RAD)];
  const hexLch = hex => labLch(hexOklab(hex));
  const lchLinear = (L, C, h) => oklabLinear(L, C * Math.cos(h * RAD), C * Math.sin(h * RAD));
  const inGamut = rgb => rgb.every(v => v >= -1e-4 && v <= 1 + 1e-4);
  function maxChroma(L, h) {                 // largest chroma displayable at this lightness and hue
    let lo = 0, hi = .4;
    for (let i = 0; i < 11; i++) {
      const c = (lo + hi) / 2;
      if (inGamut(lchLinear(L, c, h))) lo = c; else hi = c;
    }
    return lo;
  }
  function lchHex(L, C, h) {                 // chroma reduced to fit in sRGB, lightness and hue kept
    L = clamp(L, 0, 1);
    const c = Math.min(Math.max(0, C), maxChroma(L, h));
    return rgbHex(lchLinear(L, c, h).map(v => fromLin(clamp(v, 0, 1))));
  }
  const labHex = ([L, a, b]) => { const [l, c, h] = labLch([L, a, b]); return lchHex(l, c, h); };

  function rngFrom(seed) {                   // mulberry32
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------------- Settings ----------------
  const DEFAULTS = {
    style: "fidele", fidelity: 30, source: "hue", autoHues: true, hues: [], hueTones: [], harmony: "analogous",
    vividness: 55, lightness: 50, contrast: 60, diversity: 50,
    temperature: 0, accent: "none", distribution: "balanced", valueOrder: "image", anchors: "none",
    minContrast: 50, supportMode: "keep", paints: "",
  };
  const look = (fidelity, source, harmony, vividness, lightness, contrast, temperature, accent, distribution, anchors) =>
    ({ fidelity, source, harmony, vividness, lightness, contrast, temperature, accent, distribution, anchors, valueOrder: "image" });
  const STYLES = {
    fidele:  look(60, "exact", "free", 60, 50, 58, 0, "none", "balanced", "none"),
    pop:     look(10, "hue", "triad", 95, 55, 70, 0, "small", "equal", "dark"),
    affiche: look(15, "hue", "complementary", 68, 52, 85, 5, "small", "balanced", "both"),
    pastel:  look(15, "hue", "analogous", 28, 78, 28, 5, "none", "equal", "none"),
    vintage: look(20, "hue", "split", 36, 54, 50, 30, "none", "balanced", "light"),
    neon:    look(5, "hue", "complementary", 100, 60, 50, -10, "small", "equal", "dark"),
    naturel: look(40, "hue", "analogous", 34, 50, 55, 22, "none", "balanced", "none"),
    duotone: look(10, "hue", "complementary", 62, 52, 72, 0, "none", "equal", "none"),
    mono:    look(15, "hue", "mono", 50, 52, 75, 0, "none", "balanced", "none"),
  };
  const VIVID_NEUTRAL = .6 ** 1.1;
  const HARMONY = {
    mono: [0], analogous: [0, -32, 32], complementary: [0, 180], split: [0, 150, 210], triad: [0, 120, 240],
  };
  const harmonyHues = (h, harmony) => (HARMONY[harmony] || [0]).map(o => wrap(h + o));
  const SHARES = {                           // part of the painted surface given to each hue
    balanced: { 1: [1], 2: [.7, .3], 3: [.6, .3, .1], more: [.5, .25, .15, .1] },
    dominant: { 1: [1], 2: [.85, .15], 3: [.8, .15, .05], more: [.75, .13, .08, .04] },
  };

  // Main hues of an image (RGBA pixels), weighted by their colourfulness: [{h, weight}], strongest first
  function imageHues(data) {
    const N = 36, bins = new Float64Array(N);
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] < 128) continue;
      const [, C, h] = labLch(rgbOklab([data[i], data[i + 1], data[i + 2]]));
      if (C > .03) bins[Math.floor(h / 10) % N] += C;
    }
    const smooth = Array.from(bins, (_, i) => bins[(i + N - 1) % N] * .25 + bins[i] * .5 + bins[(i + 1) % N] * .25);
    const top = Math.max(...smooth);
    if (!(top > 0)) return [];
    const out = [];
    smooth.map((w, i) => ({ h: i * 10 + 5, weight: w / top })).sort((a, b) => b.weight - a.weight).forEach(p => {
      if (p.weight >= .15 && out.length < 5 && out.every(q => Math.abs(hueDelta(q.h, p.h)) >= 25)) out.push(p);
    });
    return out;
  }

  // ---------------- Lightness plan ----------------
  // Moves the wanted lightnesses as little as needed: at least `gap` between two of them and from
  // each fixed value (support, locked layers); the order is kept.
  function placeLightness(wanted, fixed, gap, lo, hi) {
    const bands = [];
    fixed.map(f => [Math.max(lo, f - gap), Math.min(hi, f + gap)]).filter(b => b[1] > b[0])
      .sort((a, b) => a[0] - b[0]).forEach(b => {
        const p = bands[bands.length - 1];
        if (p && b[0] <= p[1]) p[1] = Math.max(p[1], b[1]); else bands.push(b.slice());
      });
    const free = hi - lo - bands.reduce((s, b) => s + b[1] - b[0], 0);
    if (free <= 1e-6) return wanted.map(x => clamp(x, lo, hi));
    const squeeze = x => {                   // position in the free part of the range
      x = clamp(x, lo, hi);
      return x - lo - bands.reduce((s, b) => s + Math.max(0, Math.min(x, b[1]) - b[0]), 0);
    };
    const expand = t => bands.reduce((x, b) => x > b[0] ? x + b[1] - b[0] : x, lo + t);
    const n = wanted.length, order = wanted.map((_, i) => i).sort((a, b) => wanted[a] - wanted[b] || a - b);
    const g = n > 1 ? Math.min(gap, free / (n - 1)) : gap, t = order.map(i => squeeze(wanted[i]));
    for (let i = 1; i < n; i++) t[i] = Math.max(t[i], t[i - 1] + g);
    if (n) t[n - 1] = Math.min(t[n - 1], free);
    for (let i = n - 2; i >= 0; i--) t[i] = Math.min(t[i], t[i + 1] - g);
    const out = Array(n);
    order.forEach((k, i) => { out[k] = clamp(expand(Math.max(0, t[i])), lo, hi); });
    return out;
  }

  // ---------------- One candidate ----------------
  function shuffled(list, rng) {
    const a = list.slice();
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }
  function pickWeighted(list, rng) {
    let r = rng() * list.reduce((s, x) => s + x.weight, 0);
    for (const x of list) if ((r -= x.weight) <= 0) return x;
    return list[list.length - 1];
  }
  function pickHues(cx, rng) {
    const { S, div } = cx, img = cx.imageHues, own = S.autoHues ? [] : S.hues;
    const jitter = a => (rng() - .5) * 2 * a;
    let h0;
    if (own.length) h0 = own[0] + jitter(div * 18);
    else if (img.length && rng() > div * .6) h0 = pickWeighted(img, rng).h + jitter(4 + div * 25);
    else h0 = rng() * 360;
    let set;
    if (S.harmony === "free") {
      const seeds = own.length ? own.slice(1) : shuffled(img.map(x => x.h), rng).slice(0, 2);
      set = [h0, ...seeds.map(h => h + jitter(4 + div * 16))];
      const extra = own.length ? (rng() < div * .7 ? 1 : 0) : Math.floor(rng() * 2 * (.3 + div));
      for (let i = 0; i < extra; i++) set.push(rng() * 360);
    } else {
      set = (HARMONY[S.harmony] || HARMONY.analogous).map(o => h0 + o * (1 + jitter(.12 * div)));
      own.slice(1).forEach(h => {            // the other chosen hues replace the closest computed ones
        const free = set.map((x, i) => i).slice(1);
        if (!free.length || set.length < own.length) { set.push(h); return; }
        const i = free.reduce((b, k) => Math.abs(hueDelta(set[k], h)) < Math.abs(hueDelta(set[b], h)) ? k : b, free[0]);
        set[i] = h;
      });
    }
    return set.map(wrap);
  }
  // Saturation (0..1 of what is displayable) of the chosen base colour closest to this hue;
  // null when the base hue has no chosen colour (image hues, wheel only)
  function baseSaturation(cx, h) {
    const { S } = cx;
    if (S.autoHues || !S.hues.length) return null;
    const dist = x => Math.abs(hueDelta(x, h));
    const i = S.hues.reduce((b, x, k) => dist(x) < dist(S.hues[b]) ? k : b, 0), t = S.hueTones[i];
    return t ? clamp(t.C / Math.max(.01, maxChroma(t.L, S.hues[i])), 0, 1) : null;
  }
  function pickSupport(cx, hues, rng) {
    if (cx.S.supportMode !== "propose") return cx.support;
    const options = [lchHex(.965, .022, 90), lchHex(.94, .035, hues[0]), lchHex(.2, .02, hues[0]), "#ffffff"];
    const guard = cx.supportGuard;
    let valid = options;
    if (guard) {                             // the tone left bare must stay the same: same stencils
      const labs = guard.tones.map(srgbLab);
      valid = options.filter(c => {
        const l = srgbLab(c), d = labs.map(t => Math.hypot(l[0] - t[0], l[1] - t[1], l[2] - t[2]));
        return d.indexOf(Math.min(...d)) === guard.keep;
      });
      valid.push(guard.tones[guard.keep]);
    }
    return rng() < .25 ? cx.support : valid[Math.floor(rng() * valid.length)];
  }
  function planValues(cx, supportL, rng) {
    const { S, div, layers, n } = cx;
    let span = (.22 + .6 * S.contrast / 100) * (1 + (rng() - .5) * .16 * div);
    span = Math.min(span, .83);
    const mid = clamp(.62 + (S.lightness - 50) / 100 * .5 + (rng() - .5) * .07 * div, .14 + span / 2, .97 - span / 2);
    const hi = mid + span / 2, lo = mid - span / 2;
    // position 0 = lightest; half even steps, half the real distances between the image tones
    let u = layers.map(l => n > 1 ? mix(l.rank / (n - 1), l.relL, .5) : .5);
    if (S.valueOrder === "inverse") u = u.map(x => 1 - x);
    else if (S.valueOrder === "free") {
      const v = shuffled(u, rng); u = v;
    }
    const wanted = u.map(x => mix(hi, lo, x));
    const free = layers.map((l, i) => i).filter(i => !layers[i].lock);
    const by = free.slice().sort((a, b) => wanted[a] - wanted[b]), anchored = {};
    if (by.length > 1 && (S.anchors === "dark" || S.anchors === "both")) { wanted[by[0]] = .2; anchored[by[0]] = 1; }
    if (by.length > 1 && (S.anchors === "light" || S.anchors === "both")) { wanted[by[by.length - 1]] = .96; anchored[by[by.length - 1]] = 1; }
    const gap = .015 + S.minContrast / 100 * .1;
    const fixed = [supportL, ...layers.filter(l => l.lock).map(l => hexLch(l.lock)[0])];
    const placed = placeLightness(free.map(i => wanted[i]), fixed, gap, .12, .98);
    const Ls = layers.map(l => l.lock ? hexLch(l.lock)[0] : 0);
    free.forEach((i, k) => { Ls[i] = placed[k]; });
    return { Ls, anchored };
  }
  function assignHues(cx, hues, rng) {
    const { S, layers, n } = cx, m = hues.length;
    let share;
    if (S.distribution === "equal" || !SHARES[S.distribution]) share = Array(m).fill(1 / m);
    else {
      const t = SHARES[S.distribution], s = t[m] || t.more;
      share = hues.map((_, i) => s[i] !== undefined ? s[i] : .02);
    }
    const got = Array(m).fill(0), out = Array(n).fill(0);
    layers.map((l, i) => ({ i, a: l.area * (1 + (rng() - .5) * .5) })).sort((a, b) => b.a - a.a).forEach(({ i }) => {
      let best = 0, top = -Infinity;
      for (let k = 0; k < m; k++) { const lack = share[k] - got[k]; if (lack > top) { top = lack; best = k; } }
      out[i] = best; got[best] += layers[i].area;
    });
    return out;
  }
  function finish(cx, lch, support) {        // locks, image mix already done: to colours + paints
    const colors = lch.map((c, i) => cx.layers[i].lock || c.hex || lchHex(c.L, c.C, c.h));
    return { colors: cx.paints ? snapPaints(cx, colors) : colors, support };
  }
  function snapPaints(cx, colors) {
    const free = cx.layers.map((l, i) => i).filter(i => !cx.layers[i].lock);
    const locked = new Set(cx.layers.filter(l => l.lock).map(l => l.lock.toLowerCase()));
    let pool = cx.paints.filter(p => !locked.has(p));
    if (!pool.length) pool = cx.paints;
    if (!free.length) return colors;
    const out = colors.slice(), labs = pool.map(cx.lab);
    const cost = free.map(i => labs.map(p => deltaE00(cx.lab(colors[i]), p)));
    if (pool.length >= free.length) {
      assignMin(cost).forEach((j, k) => { out[free[k]] = pool[j]; });
    } else {                                 // fewer paints than layers: some layers share one
      cost.forEach((row, k) => { out[free[k]] = pool[row.indexOf(Math.min(...row))]; });
    }
    return out;
  }
  function imageMix(cx, l, c) {              // brings a colour towards the tone of the layer in the image
    const f = cx.S.fidelity / 100, T = l.lch;
    if (f <= 0) return c;
    if (cx.S.source === "exact") {
      const a = c.hex ? hexOklab(c.hex) : [c.L, c.C * Math.cos(c.h * RAD), c.C * Math.sin(c.h * RAD)];
      return { hex: f >= 1 ? l.tone : labHex(a.map((v, k) => mix(v, l.lab[k], f))) };
    }
    if (T[1] < .02) return c;                // no hue to take from a grey
    const h = wrap(c.h + hueDelta(c.h, T[2]) * f);
    return { L: c.L, C: cx.S.source === "both" ? mix(c.C, T[1], f) : c.C, h };
  }
  function candidate(cx, seed) {
    const rng = rngFrom(seed), { S, layers, n } = cx;
    const hues = pickHues(cx, rng), support = pickSupport(cx, hues, rng);
    const { Ls, anchored } = planValues(cx, hexLch(support)[0], rng);
    const roles = assignHues(cx, hues, rng);
    const k = (S.vividness / 100) ** 1.1, baseSat = hues.map(h => baseSaturation(cx, h));
    let accent = -1;
    if (S.accent !== "none" && n > 1) {
      const pool = layers.map((l, i) => i).filter(i => !layers[i].lock && !anchored[i] && Ls[i] > .3 && Ls[i] < .86);
      if (pool.length) accent = pool.reduce((b, i) => (S.accent === "small" ? layers[i].area < layers[b].area : layers[i].area > layers[b].area) ? i : b, pool[0]);
    }
    const accentHue = hues.length > 1 ? hues[hues.length - 1] : wrap(hues[0] + 180 + (rng() - .5) * 40);
    const lch = layers.map((l, i) => {
      const L = Ls[i];
      let h = i === accent ? accentHue : hues[roles[i]];
      h += hueDelta(h, 260) * (.55 - L) * .22;            // shadows slightly cooler, lights warmer
      if (S.temperature) h += hueDelta(h, S.temperature > 0 ? 55 : 255) * Math.abs(S.temperature) / 100 * .4;
      h = wrap(h);
      // a chosen base colour gives its saturation at vividness 60; the slider scales it from there
      const sat = i === accent ? null : baseSat[roles[i]], wanted = sat === null ? k : sat * k / VIVID_NEUTRAL;
      const ratio = i === accent ? Math.max(k, .95) : anchored[i] ? 0 : clamp(wanted * (1 + (rng() - .5) * .22), 0, 1);
      let C = ratio * maxChroma(L, h);
      if (anchored[i]) C = .022;
      return imageMix(cx, l, { L, C, h });
    });
    return finish(cx, lch, support);
  }
  function variation(cx, seed) {             // close to the reference proposal
    const rng = rngFrom(seed), { layers, div } = cx, amp = .5 + div, ref = cx.near;
    const turn = (rng() - .5) * (10 + 26 * amp), gain = 1 + (rng() - .5) * .35 * amp, lift = (rng() - .5) * .05 * amp;
    const base = ref.colors.map(hexLch);
    const placed = placeLightness(base.filter((_, i) => !layers[i].lock).map(c => c[0] + lift + (rng() - .5) * .03 * amp),
      [hexLch(ref.support || cx.support)[0], ...layers.filter(l => l.lock).map(l => hexLch(l.lock)[0])],
      .015 + cx.S.minContrast / 100 * .1, .12, .98);
    let p = 0;
    const lch = base.map((c, i) => layers[i].lock ? { L: c[0], C: c[1], h: c[2] }
      : { L: placed[p++], C: c[1] * gain * (1 + (rng() - .5) * .16 * amp), h: wrap(c[2] + turn + (rng() - .5) * (6 + 16 * amp)) });
    return finish(cx, lch, ref.support || cx.support);
  }

  // ---------------- Score ----------------
  function score(cx, cand) {
    const { S, layers, n } = cx, cols = cand.colors, labs = cols.map(cx.lab);
    const strict = S.minContrast / 100;
    const all = [...labs.map((lab, i) => ({ lab, i })), { lab: cx.lab(cand.support), i: -1 }].sort((a, b) => a.lab[0] - b.lab[0]);
    const pairs = [];
    for (let i = 1; i < all.length; i++) {
      const a = all[i - 1].lab, b = all[i].lab;
      pairs.push(.6 * clamp((b[0] - a[0]) / (4 + 16 * strict), 0, 1) + .4 * clamp(deltaE00(a, b) / (8 + 20 * strict), 0, 1));
    }
    let legibility = pairs.length ? .7 * Math.min(...pairs) + .3 * pairs.reduce((s, x) => s + x, 0) / pairs.length : 1;
    if (S.valueOrder !== "free" && n > 1) {   // order of the values against the one asked for
      let bad = 0, tot = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const img = layers[i].rank - layers[j].rank, now = labs[j][0] - labs[i][0];
        tot++; if ((S.valueOrder === "inverse" ? -img : img) * now < 0) bad++;
      }
      legibility *= mix(1, .6, bad / tot);
    }
    const fidelity = 1 - clamp(layers.reduce((s, l, i) => s + l.area * deltaE00(labs[i], cx.lab(l.tone)), 0) / 45, 0, 1);
    const lch = cols.map(hexLch), want = (S.vividness / 100) ** 1.1;
    const vivid = layers.reduce((s, l, i) => s + l.area * (lch[i][1] / Math.max(.01, maxChroma(lch[i][0], lch[i][2]))), 0);
    const respect = clamp(1 - Math.abs(vivid - want) * 1.4, 0, 1);
    const f = S.fidelity / 100, wF = .1 + .45 * f, sum = .45 + wF + .25;
    return { legibility, fidelity, respect, vivid, total: (.45 * legibility + wF * fidelity + .25 * respect) / sum };
  }
  const BADGE = c => [
    c.scores.legibility >= .72 && "readable", c.scores.legibility < .35 && "weak",
    c.scores.fidelity >= .68 && "close", c.scores.vivid >= .62 && "vivid",
  ].filter(Boolean);

  // ---------------- Generation ----------------
  /* opts: layers [{tone, area, lock}], support, S (settings), imageHues [{h, weight}], seed, count,
   * supportGuard {tones, keep} (tone that must stay the closest to the support), paints [hex],
   * near {colors, support} (variations of this proposal).
   * Returns [{colors, support, scores, badges}], best first (`near`: the reference comes first). */
  function generate(opts) {
    const S = { ...DEFAULTS, ...opts.S }, count = opts.count || 6, seed = (opts.seed || 1) >>> 0;
    const cache = new Map(), lab = h => { let v = cache.get(h); if (!v) cache.set(h, v = srgbLab(h)); return v; };
    const raw = opts.layers.map(l => ({ tone: l.tone.toLowerCase(), lock: l.lock ? l.lock.toLowerCase() : null, area: Math.max(0, +l.area || 0) }));
    const n = raw.length;
    if (!n) return [];
    const totalArea = raw.reduce((s, l) => s + l.area, 0);
    const lightToDark = raw.map((l, i) => ({ i, L: hexOklab(l.tone)[0] })).sort((a, b) => b.L - a.L || a.i - b.i);
    const top = lightToDark[0].L, range = top - lightToDark[n - 1].L;
    lightToDark.forEach((x, r) => {
      const l = raw[x.i];
      l.rank = r; l.relL = range > 1e-3 ? (top - x.L) / range : (n > 1 ? r / (n - 1) : .5);
      l.lab = hexOklab(l.tone); l.lch = labLch(l.lab);
      l.area = totalArea > 0 ? l.area / totalArea : 1 / n;
    });
    const paints = opts.paints && opts.paints.length ? [...new Set(opts.paints.map(p => p.toLowerCase()))] : null;
    const cx = { S, div: clamp(S.diversity, 0, 100) / 100, layers: raw, n, lab, paints, support: opts.support.toLowerCase(),
      imageHues: opts.imageHues || [], supportGuard: opts.supportGuard || null, near: opts.near || null };
    const seen = new Set(), pool = [];
    const add = c => {
      const key = c.colors.join() + c.support;
      if (seen.has(key)) return;
      seen.add(key);
      c.scores = score(cx, c); c.lab = c.colors.map(hexOklab); pool.push(c);
    };
    let first = null;
    if (cx.near) {
      first = { colors: cx.near.colors.map((c, i) => raw[i].lock || c.toLowerCase()), support: (cx.near.support || cx.support).toLowerCase() };
      add(first);
    }
    for (let i = 0; i < 140; i++) add((cx.near ? variation : candidate)(cx, (seed * 2654435761 + i * 40503) >>> 0));
    pool.sort((a, b) => b.scores.total - a.scores.total);
    // the best ones, each far enough from those already kept
    const dist = (a, b) => a.lab.reduce((s, x, i) => s + Math.hypot(x[0] - b.lab[i][0], x[1] - b.lab[i][1], x[2] - b.lab[i][2]), 0) / n * 100;
    const kept = first ? [pool.find(c => c === first)] : [];
    const minDist = (cx.near ? 1.5 + 5 * cx.div : 3 + 16 * cx.div);
    for (const relax of [1, .6, .3, 0]) {
      for (const c of pool) {
        if (kept.length >= count) break;
        if (!kept.includes(c) && kept.every(k => dist(c, k) >= minDist * relax)) kept.push(c);
      }
    }
    return kept.map(c => ({ colors: c.colors, support: c.support, scores: c.scores, badges: BADGE(c) }));
  }

  root.PalGen = { srgbLab, deltaE00, assignMin, hexLch, lchHex, maxChroma, imageHues, harmonyHues, placeLightness, generate, DEFAULTS, STYLES };
})(typeof window !== "undefined" ? window : globalThis);
