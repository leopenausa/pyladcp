/* pyladcp studio — live single-station solve UI.
 *
 * One source of truth: `S` mirrors SessionConfig; every change debounces into
 * POST /api/station/{label}/solve and the canvas redraws from the JSON profile.
 * Pins freeze a solution client-side as dashed ghosts; the Δ-strip shows
 * live − selected pin. QA panels are the ladcp-qa matplotlib figures rendered
 * server-side for the current configuration. Axis legibility is a deliberate
 * priority (bright axis ink, units everywhere; full polish pass in PR 5). */
"use strict";

const $ = id => document.getElementById(id);

const S = {                                     // mirrors SessionConfig
  station: null,
  solver: "inverse",
  botfac: 1.0, barofac: 1.0, smoofac: 0.0, sadcpfac: 3.0,
  down_only: false,
  use_nearfield: false,                         // the near-field mask toggle (default OFF)
  nearfield: null,                              // 1-based bins when typed, e.g. [3,4]
  dn_geom: null,                                // {first_m, cell_m, n_bins} from the solve
  dzbelow: null,                                // null = preset
  use_sadcp: false,                             // the constraint toggle
  sadcp_key: null,                              // selected key from sadcp_sources
  sadcp_sources: [],                            // [{key, source, folder, origin}] fixed at launch
};

let seq = 0;                                    // drop stale in-flight responses
let timer = null;
let last = null;                                // last solve payload (for redraws)
const pins = [];                                // {id,color,label,sub,z,u,v}
let pinSeq = 0;
let deltaPin = null;                            // pin id used as the Δ reference
const PIN_COLORS = ["#8d7bff", "#e36fa7", "#5fb0ff", "#ffd166", "#9ef0a0"];

function status(cls, text) {
  const el = $("status");
  el.className = "solve-state " + cls;
  $("status-text").textContent = text;
}

function editBody() {
  const nf = (S.use_nearfield && S.nearfield && S.nearfield.length) ? S.nearfield : null;
  return { down_only: S.down_only, nearfield_dn_bins: nf, dzbelow: S.dzbelow };
}

function body() {
  return JSON.stringify({
    edit: editBody(),
    solve: { solver: S.solver, botfac: S.botfac, barofac: S.barofac,
             smoofac: S.smoofac, sadcpfac: S.sadcpfac },
    sadcp_key: (S.use_sadcp && S.sadcp_key) ? S.sadcp_key : "off",
  });
}

function scheduleSolve(delay = 60) {
  clearTimeout(timer);
  timer = setTimeout(solve, delay);
}

async function solve() {
  if (!S.station) return;
  const mySeq = ++seq;
  const firstLoad = last === null || last.station !== S.station;
  if (firstLoad) $("overlay").classList.remove("hidden");
  status("busy", firstLoad ? "preparing…" : "solving…");
  if (firstLoad) {
    $("st-load").className = "stage busy";
    $("st-build").className = "stage busy";
  }
  $("st-solve").className = "stage busy";
  try {
    const r = await fetch(`api/station/${encodeURIComponent(S.station)}/solve`,
                          { method: "POST", body: body(),
                            headers: { "Content-Type": "application/json" } });
    if (mySeq !== seq) return;                  // a newer request superseded this one
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
      status("err", `error: ${detail}`);
      $("overlay").classList.add("hidden");
      return;
    }
    const p = await r.json();
    if (mySeq !== seq) return;
    last = p;
    $("overlay").classList.add("hidden");
    const fmtStage = ms => ms == null ? "–" : (ms >= 100 ? `${(ms / 1000).toFixed(1)} s`
                                                         : `${Math.round(ms)} ms`);
    $("st-load").className = "stage done";
    $("load-ms").textContent = fmtStage(p.stages.load_ms)
      + (p.prepared ? " · cached" : "");
    $("st-build").className = "stage done";
    $("build-ms").textContent = fmtStage(p.stages.build_ms)
      + (p.prepared ? " · cached" : "");
    $("st-solve").className = "stage done";
    $("solve-ms").textContent = `${p.solve_ms} ms`;
    status("live", `solved in ${p.solve_ms} ms`);
    $("ro-drot").textContent = p.drot === null ? "–"
      : `${p.drot.toFixed(2)}°${p.drot_source === "explicit" ? "" : " igrf"}`;
    $("ro-zbottom").textContent = p.zbottom === null ? "none" : `${Math.round(p.zbottom)} m`;
    $("ro-ubar").textContent = fmtVel(p.profile.ubar);
    $("ro-vbar").textContent = fmtVel(p.profile.vbar);
    $("cli").textContent = p.cli;
    $("key-sadcp").style.display = p.sadcp ? "flex" : "none";
    $("ro-edits").textContent = p.manual_edits
      ? `${p.manual_edits} rect (✏)` : "none";
    if (p.dn_geom) { S.dn_geom = p.dn_geom; syncNearfieldControls(); }
    renderQaList(p.panels);
    draw(p);
    if (E.journal === null) refreshEdits();      // first solve: populate the edits card
    if (E.open) ensureBaseline(p);               // edit view: refresh the inset Δ
  } catch (e) {
    if (mySeq === seq) status("err", `error: ${e.message}`);
  }
}

const fmtVel = x => x === null ? "–" : `${(x * 100).toFixed(1)} cm/s`;

/* ------------------------------------------------------------------ pins */

function pinLabel() {
  const parts = [S.solver];
  if (last && last.manual_edits) parts.push(`edits ${last.manual_edits}`);
  if (S.botfac !== 1.0) parts.push(`botfac ${S.botfac}`);
  if (S.barofac !== 1.0) parts.push(`barofac ${S.barofac}`);
  if (S.smoofac !== 0.0) parts.push(`smoofac ${S.smoofac}`);
  if (S.use_sadcp && S.sadcp_key) parts.push(`sadcp ${S.sadcp_key} ${S.sadcpfac}`);
  if (S.down_only) parts.push("down-only");
  if (S.use_nearfield && S.nearfield && S.nearfield.length)
    parts.push(`nf ${S.nearfield.join(",")}`);
  if (S.dzbelow !== null) parts.push(`dzbelow ${S.dzbelow}`);
  return parts.join(" · ");
}

$("pinbtn").addEventListener("click", () => {
  if (!last) return;
  const pin = {
    id: ++pinSeq,
    color: PIN_COLORS[(pinSeq - 1) % PIN_COLORS.length],
    label: pinLabel(),
    sub: last.cli,
    z: last.profile.z, u: last.profile.u, v: last.profile.v,
  };
  pins.push(pin);
  if (deltaPin === null) deltaPin = pin.id;
  renderPins();
  if (last) draw(last);
});

function renderPins() {
  const box = $("pins");
  box.innerHTML = "";
  $("pins-hint").style.display = pins.length ? "none" : "block";
  for (const pin of pins) {
    const d = document.createElement("div");
    d.className = "pin-item" + (pin.id === deltaPin ? " selected" : "");
    d.title = "click: use as Δ reference";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.borderColor = pin.color;
    const lbl = document.createElement("span");
    lbl.className = "lbl";
    lbl.textContent = pin.label;
    const sub = document.createElement("small");
    sub.textContent = pin.sub;
    lbl.appendChild(sub);
    const x = document.createElement("span");
    x.className = "x";
    x.title = "remove pin";
    x.textContent = "✕";
    d.append(swatch, lbl, x);
    d.addEventListener("click", () => {
      deltaPin = pin.id;
      renderPins();
      if (last) draw(last);
    });
    x.addEventListener("click", ev => {
      ev.stopPropagation();
      pins.splice(pins.indexOf(pin), 1);
      if (deltaPin === pin.id) deltaPin = pins.length ? pins[0].id : null;
      renderPins();
      if (last) draw(last);
    });
    box.appendChild(d);
  }
}

function clearPins() {
  pins.length = 0;
  deltaPin = null;
  renderPins();
}

/* ------------------------------------------------------------------ QA panels */

function renderQaList(panels) {
  const box = $("qa-list");
  box.innerHTML = "";
  for (const name of panels || []) {
    const d = document.createElement("div");
    d.className = "qa-item";
    d.textContent = name;
    d.addEventListener("click", () => openPanel(name));
    box.appendChild(d);
  }
}

async function openPanel(name) {
  $("lightbox").classList.remove("hidden");
  $("lightbox-title").textContent = `${S.station} · ${name}`;
  $("lightbox-state").textContent = "rendering…";
  $("lightbox-img").removeAttribute("src");
  try {
    const r = await fetch(`api/station/${encodeURIComponent(S.station)}/qa/${name}`,
                          { method: "POST", body: body(),
                            headers: { "Content-Type": "application/json" } });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
      $("lightbox-state").textContent = `error: ${detail}`;
      return;
    }
    const blob = await r.blob();
    $("lightbox-img").src = URL.createObjectURL(blob);
    $("lightbox-state").textContent = "current config";
  } catch (e) {
    $("lightbox-state").textContent = `error: ${e.message}`;
  }
}

$("lightbox-close").addEventListener("click", () => $("lightbox").classList.add("hidden"));
$("lightbox").addEventListener("click", ev => {
  if (ev.target === $("lightbox")) $("lightbox").classList.add("hidden");
});
document.addEventListener("keydown", ev => {
  if (ev.key === "Escape") $("lightbox").classList.add("hidden");
});

/* ------------------------------------------------------------------ canvas */

const canvas = $("plot");
const ctx = canvas.getContext("2d");

const AXIS = "#bdd0e2";                          // axis ink: deliberately bright
const FRAME = "rgba(189,208,226,.45)";           // pane frames + tick marks
const GRID = "rgba(159,180,200,.13)";
const ZERO = "rgba(189,208,226,.40)";
const FONT = '13px ui-monospace,"SF Mono",Consolas,monospace';
const FONT_TITLE = '600 13px ui-monospace,"SF Mono",Consolas,monospace';

function niceStep(span, target) {
  const raw = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * mag) return m * mag;
  return 10 * mag;
}

/* depth-keyed lookup so pins align with the live grid even if lengths differ */
function zMap(pin) {
  const mu = new Map(), mv = new Map();
  for (let i = 0; i < pin.z.length; i++) {
    if (pin.z[i] === null) continue;
    mu.set(pin.z[i], pin.u[i]);
    mv.set(pin.z[i], pin.v[i]);
  }
  return { mu, mv };
}

function draw(p) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H) return;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.font = FONT;

  const z = p.profile.z, u = p.profile.u, v = p.profile.v, ue = p.profile.uerr;
  const finite = a => a.filter(x => x !== null);
  const zmax = Math.max(...finite(z), p.zbottom || 0) * 1.03;

  // symmetric velocity range over both components + ghosts (shared scale)
  let vmax = 0.05;
  const stretch = a => { for (const x of a) if (x !== null) vmax = Math.max(vmax, Math.abs(x)); };
  stretch(u); stretch(v);
  if (p.bt) { stretch(p.bt.u); stretch(p.bt.v); }
  if (p.sadcp) { stretch(p.sadcp.u); stretch(p.sadcp.v); }
  for (const pin of pins) { stretch(pin.u); stretch(pin.v); }
  vmax *= 1.12;

  // Δ values vs the selected pin
  const ref = pins.find(q => q.id === deltaPin) || null;
  let du = null, dv = null, dmax = 0.01;
  if (ref) {
    const { mu, mv } = zMap(ref);
    du = z.map((zz, i) => (u[i] !== null && mu.get(zz) != null) ? u[i] - mu.get(zz) : null);
    dv = z.map((zz, i) => (v[i] !== null && mv.get(zz) != null) ? v[i] - mv.get(zz) : null);
    for (const a of [du, dv]) for (const x of a)
      if (x !== null) dmax = Math.max(dmax, Math.abs(x));
  }
  dmax *= 1.15;

  const padL = 64, padR = 14, padT = 30, padB = 40, gap = 34;
  const dW = ref ? 110 : 0;                      // Δ-strip only when a pin is selected
  const paneW = (W - padL - padR - gap - (ref ? gap + dW : 0)) / 2;
  const panes = { u: [padL, padL + paneW], v: [padL + paneW + gap, padL + 2 * paneW + gap] };
  if (ref) panes.d = [panes.v[1] + gap, panes.v[1] + gap + dW];
  const Y = zz => padT + zz / zmax * (H - padT - padB);
  const X = (val, pane, lim) => pane[0] + (val + lim) / (2 * lim) * (pane[1] - pane[0]);

  // depth grid + tick marks + labels (left pane carries the depth axis)
  const rightEdge = ref ? panes.d[1] : panes.v[1];
  const zstep = niceStep(zmax, 8);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let zz = 0; zz <= zmax; zz += zstep) {
    ctx.strokeStyle = GRID; ctx.beginPath();
    ctx.moveTo(panes.u[0], Y(zz)); ctx.lineTo(rightEdge, Y(zz)); ctx.stroke();
    ctx.strokeStyle = FRAME; ctx.beginPath();             // tick mark on the axis
    ctx.moveTo(panes.u[0] - 5, Y(zz)); ctx.lineTo(panes.u[0], Y(zz)); ctx.stroke();
    ctx.fillStyle = AXIS;
    ctx.fillText(`${zz}`, panes.u[0] - 9, Y(zz));
  }
  // depth-axis title, rotated along the left edge
  ctx.save();
  ctx.translate(15, (padT + H - padB) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.font = FONT_TITLE; ctx.fillStyle = AXIS;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("depth · m", 0, 0);
  ctx.restore();
  ctx.font = FONT;

  const drawFrame = (pane, lim, title, tickTarget) => {
    ctx.strokeStyle = FRAME;
    ctx.strokeRect(pane[0], padT, pane[1] - pane[0], H - padT - padB);
    const vstep = niceStep(2 * lim, tickTarget);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let val = -Math.floor(lim / vstep) * vstep; val <= lim; val += vstep) {
      const x = X(val, pane, lim);
      ctx.strokeStyle = Math.abs(val) < 1e-12 ? ZERO : GRID;
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, H - padB); ctx.stroke();
      ctx.strokeStyle = FRAME; ctx.beginPath();           // tick mark below the frame
      ctx.moveTo(x, H - padB); ctx.lineTo(x, H - padB + 5); ctx.stroke();
      ctx.fillStyle = AXIS;
      ctx.fillText(`${Math.round(val * 1000) / 10}`, x, H - padB + 8);
    }
    ctx.font = FONT_TITLE; ctx.fillStyle = AXIS; ctx.textBaseline = "alphabetic";
    ctx.fillText(title, (pane[0] + pane[1]) / 2, padT - 10);
    ctx.font = FONT;
  };
  drawFrame(panes.u, vmax, "U  east  ·  cm s⁻¹", 6);
  drawFrame(panes.v, vmax, "V  north  ·  cm s⁻¹", 6);
  if (ref) drawFrame(panes.d, dmax, "Δ  ·  cm s⁻¹", 2);

  // seabed
  if (p.zbottom !== null && p.zbottom <= zmax) {
    const y = Y(p.zbottom);
    ctx.fillStyle = "rgba(255,209,102,.07)";
    ctx.fillRect(panes.u[0], y, rightEdge - panes.u[0], H - padB - y);
    ctx.strokeStyle = "rgba(255,209,102,.5)";
    ctx.setLineDash([6, 4]); ctx.beginPath();
    ctx.moveTo(panes.u[0], y); ctx.lineTo(rightEdge, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(255,209,102,.75)"; ctx.textAlign = "right";
    ctx.fillText(`seabed ${Math.round(p.zbottom)} m`, panes.v[1] - 6, y - 6);
  }

  const tracePath = (zz, comp, pane, lim) => {
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < zz.length; i++) {
      if (comp[i] === null) { pen = false; continue; }
      const x = X(comp[i], pane, lim), y = Y(zz[i]);
      pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y); pen = true;
    }
    ctx.stroke();
  };

  // ghosts (dashed, under the live profile)
  ctx.setLineDash([5, 4]); ctx.lineWidth = 1.3;
  ctx.globalAlpha = 0.6;
  for (const pin of pins) {
    ctx.strokeStyle = pin.color;
    tracePath(pin.z, pin.u, panes.u, vmax);
    tracePath(pin.z, pin.v, panes.v, vmax);
  }
  ctx.globalAlpha = 1; ctx.setLineDash([]); ctx.lineWidth = 1;

  // ±1σ bands (the LADCP error estimate is isotropic: uerr applies to u and v)
  const sigmaBand = (comp, pane, color) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < z.length; i++) {
      if (comp[i] === null || ue[i] === null) continue;
      const x = X(comp[i] - ue[i], pane, vmax), y = Y(z[i]);
      started ? ctx.lineTo(x, y) : ctx.moveTo(x, y); started = true;
    }
    for (let i = z.length - 1; i >= 0; i--) {
      if (comp[i] === null || ue[i] === null) continue;
      ctx.lineTo(X(comp[i] + ue[i], pane, vmax), Y(z[i]));
    }
    if (started) ctx.fill();
  };
  sigmaBand(u, panes.u, "rgba(57,211,200,.13)");
  sigmaBand(v, panes.v, "rgba(255,158,100,.13)");

  // bottom-track reference (dots)
  if (p.bt) {
    ctx.fillStyle = "rgba(255,209,102,.8)";
    for (let i = 0; i < p.bt.z.length; i++) {
      for (const [comp, pane] of [[p.bt.u, panes.u], [p.bt.v, panes.v]]) {
        if (comp[i] === null) continue;
        ctx.beginPath();
        ctx.arc(X(comp[i], pane, vmax), Y(p.bt.z[i]), 2.1, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }

  // ship-ADCP constraint profile (open squares, true frame; verr stays in the
  // payload but is not drawn — raw STA error bars are too wide to be useful here)
  if (p.sadcp) {
    ctx.strokeStyle = "#e8f2fb"; ctx.lineWidth = 1; ctx.globalAlpha = 0.95;
    for (let i = 0; i < p.sadcp.z.length; i++) {
      const zz = p.sadcp.z[i];
      for (const [comp, pane] of [[p.sadcp.u, panes.u], [p.sadcp.v, panes.v]]) {
        if (comp[i] === null || zz === null) continue;
        ctx.strokeRect(X(comp[i], pane, vmax) - 2.4, Y(zz) - 2.4, 4.8, 4.8);
      }
    }
    ctx.globalAlpha = 1;
  }

  // live profiles
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#39d3c8"; tracePath(z, u, panes.u, vmax);
  ctx.strokeStyle = "#ff9e64"; tracePath(z, v, panes.v, vmax);

  // Δ-strip traces + reference note
  if (ref) {
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = "#39d3c8"; tracePath(z, du, panes.d, dmax);
    ctx.strokeStyle = "#ff9e64"; tracePath(z, dv, panes.d, dmax);
    ctx.fillStyle = ref.color; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(`vs ${ref.label.length > 20 ? ref.label.slice(0, 19) + "…" : ref.label}`,
                 (panes.d[0] + panes.d[1]) / 2, H - padB + 22);
  }
  ctx.lineWidth = 1;
}

new ResizeObserver(() => { if (last) draw(last); }).observe(canvas);

/* ------------------------------------------------------------------ controls */

document.querySelectorAll(".knob").forEach(knob => {
  const input = knob.querySelector("input"), out = knob.querySelector("output");
  const k = knob.dataset.k;
  const set = (fire) => {
    const val = input.value / 10;
    out.textContent = val.toFixed(1);
    input.style.setProperty("--fill", (input.value / input.max * 100) + "%");
    S[k] = val;
    if (fire) scheduleSolve();
  };
  input.addEventListener("input", () => set(true));
  set(false);
});

$("solver").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
  $("solver").querySelectorAll("button").forEach(x => x.classList.remove("on"));
  b.classList.add("on");
  S.solver = b.dataset.v;
  // constraint weights act on the inverse only
  document.querySelectorAll(".knob").forEach(kn =>
    kn.classList.toggle("off", S.solver !== "inverse"));
  scheduleSolve(0);
}));

$("tgl-downonly").addEventListener("click", () => {
  S.down_only = !S.down_only;
  $("tgl-downonly").classList.toggle("on", S.down_only);
  scheduleSolve(0);                              // edit change: server rebuilds (~1.5 s)
});

$("tgl-sadcp").addEventListener("click", () => {
  if (!S.sadcp_sources.length) return;
  S.use_sadcp = !S.use_sadcp;
  syncSadcpControls();
  scheduleSolve(0);
});

$("sel-sadcp").addEventListener("change", () => {
  S.sadcp_key = $("sel-sadcp").value;
  syncSadcpControls();
  if (S.use_sadcp) scheduleSolve(0);
});

function syncSadcpControls() {
  $("tgl-sadcp").classList.toggle("on", S.use_sadcp);
  $("tgl-sadcp").classList.toggle("disabled", !S.sadcp_sources.length);
  $("sel-sadcp").disabled = !S.sadcp_sources.length || !S.use_sadcp;
  const src = S.sadcp_sources.find(s => s.key === S.sadcp_key);
  $("sadcp-note").textContent = src ? src.folder : "launch with --sadcp / --sadcp-codas";
}

/* editing overrides: parse on Enter/blur; empty = cruise preset (null) */
function bindEditField(input, parse) {
  const apply = () => {
    try {
      parse(input.value.trim());
      input.classList.remove("bad");
      scheduleSolve(0);
    } catch {
      input.classList.add("bad");
    }
  };
  input.addEventListener("blur", apply);
  input.addEventListener("keydown", ev => { if (ev.key === "Enter") input.blur(); });
}

/* near-field mask: bins ("3,4") or a depth range below the package ("22-38m" /
 * a single "26m" = the bin containing that depth), converted with the station's
 * real bin geometry from the last solve. Bins stay canonical (the CLI contract). */
bindEditField($("in-nearfield"), text => {
  if (text === "" || text.toLowerCase() === "none") { S.nearfield = null; return; }
  const m = text.match(/^\s*(\d+(?:\.\d+)?)\s*(?:[-–]\s*(\d+(?:\.\d+)?))?\s*m\s*$/i);
  if (m) {
    if (!S.dn_geom) throw new Error("geometry not loaded yet");
    const { first_m, cell_m, n_bins } = S.dn_geom;
    const bins = [];
    if (m[2] === undefined) {                    // single depth: the bin containing it
      const b = Math.round((parseFloat(m[1]) - first_m) / cell_m) + 1;
      if (b < 1 || b > n_bins) throw new Error("depth outside the bin range");
      bins.push(b);
    } else {                                     // range: bins whose centre lies inside
      const lo = parseFloat(m[1]), hi = parseFloat(m[2]);
      for (let b = 1; b <= n_bins; b++) {
        const c = first_m + (b - 1) * cell_m;
        if (c >= lo && c <= hi) bins.push(b);
      }
      if (!bins.length) throw new Error("no bin centre in that range");
    }
    S.nearfield = bins;
  } else {
    S.nearfield = text.split(",").map(s => {
      const n = Number(s.trim());
      if (!Number.isInteger(n) || n < 1) throw new Error("bad bin");
      return n;
    });
  }
  syncNearfieldControls();
});

$("tgl-nearfield").addEventListener("click", () => {
  S.use_nearfield = !S.use_nearfield;
  syncNearfieldControls();
  if (S.nearfield && S.nearfield.length) scheduleSolve(0);   // mask actually changed
});

function syncNearfieldControls() {
  $("tgl-nearfield").classList.toggle("on", S.use_nearfield);
  $("in-nearfield").disabled = !S.use_nearfield;
  let note = "no mask";
  if (S.use_nearfield && S.nearfield && S.nearfield.length) {
    note = `bins ${S.nearfield.join(",")}`;
    if (S.dn_geom) {
      const { first_m, cell_m } = S.dn_geom;
      const lo = first_m + (Math.min(...S.nearfield) - 1) * cell_m - cell_m / 2;
      const hi = first_m + (Math.max(...S.nearfield) - 1) * cell_m + cell_m / 2;
      note += ` ≈ ${lo.toFixed(0)}–${hi.toFixed(0)} m below package`;
    }
  } else if (S.use_nearfield) {
    note = "type bins to mask — nothing masked yet";
  }
  // the brush journal masks cells through the same edit stage (the two OR
  // together); say so here so an active journal is never a surprise
  const nEdits = E.journal ? E.journal.entries.length : 0;
  if (nEdits) note += ` · ✏ ${nEdits} brush rect(s) also active`;
  $("nearfield-note").textContent = note;
}

bindEditField($("in-dzbelow"), text => {
  if (text === "") { S.dzbelow = null; return; }
  const x = Number(text);
  if (!Number.isFinite(x) || x < 0) throw new Error("bad dzbelow");
  S.dzbelow = x;
});

$("station").addEventListener("change", () => {
  S.station = $("station").value;
  last = null;
  S.dn_geom = null;                              // refreshed by the station's first solve
  clearPins();                                   // pins are per-station (z grids differ)
  resetEditView();                               // journal + matrix are per-station too
  scheduleSolve(0);
});

function stepStation(dir) {
  const sel = $("station");
  const next = sel.selectedIndex + dir;
  if (next < 0 || next >= sel.options.length) return;
  sel.selectedIndex = next;
  sel.dispatchEvent(new Event("change"));
}
$("prev-st").addEventListener("click", () => stepStation(-1));
$("next-st").addEventListener("click", () => stepStation(1));

$("dl-lad").addEventListener("click", async () => {
  if (!S.station) return;
  try {
    const r = await fetch(`api/station/${encodeURIComponent(S.station)}/lad`,
                          { method: "POST", body: body(),
                            headers: { "Content-Type": "application/json" } });
    if (!r.ok) {
      status("err", `lad export failed: ${r.statusText}`);
      return;
    }
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${S.station}.lad`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) {
    status("err", `lad export failed: ${e.message}`);
  }
});

$("copycli").addEventListener("click", async () => {
  const cli = $("cli").textContent;
  if (!cli || cli === "–") return;
  try { await navigator.clipboard.writeText(cli); }
  catch { window.prompt("copy the command:", cli); }
  $("copycli").classList.add("flash");
  setTimeout(() => $("copycli").classList.remove("flash"), 600);
});

/* ------------------------------------------------------------------ edit view
 *
 * The brush: a per-head ensemble x bin heatmap (server-rendered, full-bleed PNG)
 * with a transparent overlay canvas for marquee + flagged rectangles. The pixel
 * mapping is purely fractional: x/width -> ensemble (0-based, joint-trimmed),
 * y/height -> bin (1-based, row 0 = bin 1). The journal on the server is the
 * single source of truth -- POST/DELETE here, and every solve picks it up. */

const E = {
  open: false,
  head: "down", field: "errvel",
  meta: null,                                    // edit/meta payload (geometry)
  journal: null,                                 // journal dict from the server
  stale: null,                                   // staleness message or null
  drag: null,                                    // {x0,y0,x1,y1} in overlay px
  baseline: null,                                // no-edits profile {key,z,u,v,ubar,vbar}
};

function resetEditView() {
  E.meta = null; E.journal = null; E.stale = null; E.drag = null; E.baseline = null;
  renderEditsList();
  if (E.open) loadEditView();
}

function setView(which) {
  E.open = which === "edit";
  $("viewseg").querySelectorAll("button").forEach(b =>
    b.classList.toggle("on", b.dataset.v === which));
  $("profile-view").classList.toggle("hidden", E.open);
  $("edit-view").classList.toggle("hidden", !E.open);
  $("plot-title").textContent = E.open ? "Raw ensemble matrix" : "Velocity solution";
  document.querySelector(".legend").style.visibility = E.open ? "hidden" : "visible";
  if (E.open) {
    loadEditView();
    if (last) ensureBaseline(last);
  } else if (last) draw(last);
}
$("viewseg").querySelectorAll("button").forEach(b =>
  b.addEventListener("click", () => setView(b.dataset.v)));

$("edit-head").querySelectorAll("button").forEach(b =>
  b.addEventListener("click", () => {
    if (b.classList.contains("disabled")) return;
    E.head = b.dataset.v;
    $("edit-head").querySelectorAll("button").forEach(x =>
      x.classList.toggle("on", x === b));
    loadHeatmap();
  }));

$("edit-field").querySelectorAll("button").forEach(b =>
  b.addEventListener("click", () => {
    E.field = b.dataset.v;
    $("edit-field").querySelectorAll("button").forEach(x =>
      x.classList.toggle("on", x === b));
    loadHeatmap();
  }));

async function loadEditView() {
  if (!S.station) return;
  $("heat-msg").classList.remove("hidden");
  $("heat-msg").textContent = "loading matrix…";
  try {
    const r = await fetch(`api/station/${encodeURIComponent(S.station)}/edit/meta`,
                          { method: "POST", body: body(),
                            headers: { "Content-Type": "application/json" } });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
      $("heat-msg").textContent = `error: ${detail}`;
      return;
    }
    const m = await r.json();
    E.meta = m;
    E.journal = m.journal;
    E.stale = m.stale;
    const upBtn = $("edit-head").querySelector('button[data-v="up"]');
    const noUp = !m.heads.up;
    upBtn.classList.toggle("disabled", noUp);
    upBtn.dataset.tip = noUp ? "no up-looker in this configuration" :
      "Up-looker (slave) raw matrix.";
    if (noUp && E.head === "up") {
      E.head = "down";
      $("edit-head").querySelectorAll("button").forEach(x =>
        x.classList.toggle("on", x.dataset.v === "down"));
    }
    renderEditsList();
    await loadHeatmap();
  } catch (e) {
    $("heat-msg").textContent = `error: ${e.message}`;
  }
}

async function loadHeatmap() {
  if (!E.meta) return;
  $("heat-msg").classList.remove("hidden");
  $("heat-msg").textContent = "rendering…";
  $("heat-ylab").textContent =
    `bin 1 (top) → bin ${(E.meta.heads[E.head] || {}).n_bins || "N"} · ${E.head}-looker`;
  $("heat-xlab").textContent = `ensemble 0 → ${E.meta.joint_n_ens - 1} (joint-trimmed)`;
  try {
    const r = await fetch(
      `api/station/${encodeURIComponent(S.station)}/edit/heatmap/${E.head}/${E.field}`,
      { method: "POST", body: body(), headers: { "Content-Type": "application/json" } });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
      $("heat-msg").textContent = `error: ${detail}`;
      return;
    }
    const blob = await r.blob();
    const img = $("heat");
    const old = img.src;
    img.src = URL.createObjectURL(blob);
    if (old) URL.revokeObjectURL(old);
    if (E.stale) {
      $("heat-msg").textContent = `journal is stale: ${E.stale}`;
    } else {
      $("heat-msg").classList.add("hidden");
    }
    drawHeatOverlay();
  } catch (e) {
    $("heat-msg").textContent = `error: ${e.message}`;
  }
}

/* fractional pixel <-> cell mapping over the overlay canvas */
function cellOf(px, py) {
  const o = $("heat-overlay");
  const g = E.meta.heads[E.head];
  const ens = Math.max(0, Math.min(E.meta.joint_n_ens - 1,
    Math.floor(px / o.clientWidth * E.meta.joint_n_ens)));
  const bin = Math.max(1, Math.min(g.n_bins,
    Math.floor(py / o.clientHeight * g.n_bins) + 1));
  return { ens, bin };
}

function drawHeatOverlay() {
  const o = $("heat-overlay");
  const dpr = window.devicePixelRatio || 1;
  const W = o.clientWidth, H = o.clientHeight;
  if (!W || !H || !E.meta) return;
  o.width = W * dpr; o.height = H * dpr;
  const c = o.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, W, H);
  const g = E.meta.heads[E.head];
  if (!g) return;
  const nE = E.meta.joint_n_ens;
  // persisted rectangles for this head
  for (const e of (E.journal ? E.journal.entries : [])) {
    if (e.head !== E.head) continue;
    const x = e.ens_first / nE * W;
    const w = (e.ens_last - e.ens_first + 1) / nE * W;
    const y = (e.bin_first - 1) / g.n_bins * H;
    const h = (e.bin_last - e.bin_first + 1) / g.n_bins * H;
    c.fillStyle = "rgba(255,92,92,.30)";
    c.strokeStyle = "rgba(255,92,92,.85)";
    c.fillRect(x, y, w, h);
    c.strokeRect(x, y, w, h);
  }
  // live marquee
  if (E.drag) {
    const { x0, y0, x1, y1 } = E.drag;
    c.setLineDash([5, 4]);
    c.strokeStyle = "#53c7e8";
    c.lineWidth = 1.6;
    c.strokeRect(Math.min(x0, x1), Math.min(y0, y1),
                 Math.abs(x1 - x0), Math.abs(y1 - y0));
    c.setLineDash([]);
    c.lineWidth = 1;
  }
}

const heatOv = $("heat-overlay");
heatOv.addEventListener("pointerdown", ev => {
  if (!E.meta || E.stale) return;
  heatOv.setPointerCapture(ev.pointerId);
  const r = heatOv.getBoundingClientRect();
  E.drag = { x0: ev.clientX - r.left, y0: ev.clientY - r.top,
             x1: ev.clientX - r.left, y1: ev.clientY - r.top };
});
heatOv.addEventListener("pointermove", ev => {
  if (!E.drag) return;
  const r = heatOv.getBoundingClientRect();
  E.drag.x1 = Math.max(0, Math.min(r.width, ev.clientX - r.left));
  E.drag.y1 = Math.max(0, Math.min(r.height, ev.clientY - r.top));
  drawHeatOverlay();
});
heatOv.addEventListener("pointerup", async () => {
  if (!E.drag) return;
  const d = E.drag;
  E.drag = null;
  const a = cellOf(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1));
  const b = cellOf(Math.max(d.x0, d.x1), Math.max(d.y0, d.y1));
  drawHeatOverlay();
  if (Math.abs(d.x1 - d.x0) < 3 && Math.abs(d.y1 - d.y0) < 3) return;   // a click
  let e0 = a.ens, e1 = b.ens;
  if ((e1 - e0 + 1) / E.meta.joint_n_ens > 0.9) {     // ~full cast -> all ensembles
    e0 = 0; e1 = E.meta.joint_n_ens - 1;
  }
  await postEdit({ head: E.head, bin_first: a.bin, bin_last: b.bin,
                   ens_first: e0, ens_last: e1, view: E.field, note: "" });
});

async function postEdit(entry) {
  try {
    const payload = JSON.parse(body());
    payload.entry = entry;
    const r = await fetch(`api/station/${encodeURIComponent(S.station)}/edits`,
                          { method: "POST", body: JSON.stringify(payload),
                            headers: { "Content-Type": "application/json" } });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
      status("err", `edit failed: ${detail}`);
      return;
    }
    applyEditsPayload(await r.json());
  } catch (e) {
    status("err", `edit failed: ${e.message}`);
  }
}

async function deleteEdit(id) {
  try {
    const r = await fetch(`api/station/${encodeURIComponent(S.station)}/edits/${id}`,
                          { method: "DELETE" });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
      status("err", `delete failed: ${detail}`);
      return;
    }
    applyEditsPayload(await r.json());
  } catch (e) {
    status("err", `delete failed: ${e.message}`);
  }
}

function applyEditsPayload(p) {
  E.journal = p.journal;
  E.stale = p.stale;
  renderEditsList();
  drawHeatOverlay();
  scheduleSolve(0);                              // edit change: server rebuilds (~1.5 s)
}

async function refreshEdits() {
  if (!S.station) return;
  try {
    const r = await fetch(`api/station/${encodeURIComponent(S.station)}/edits`);
    if (!r.ok) return;
    const p = await r.json();
    E.journal = p.journal;
    E.stale = p.stale;
    renderEditsList();
  } catch { /* the edits card is best-effort outside the edit view */ }
}

function renderEditsList() {
  syncNearfieldControls();                       // its note cross-references the journal
  const box = $("edits-list");
  box.innerHTML = "";
  const entries = E.journal ? E.journal.entries : [];
  $("edits-hint").style.display = entries.length ? "none" : "block";
  const badge = $("edits-badge");
  badge.textContent = entries.length;
  badge.classList.toggle("hidden", !entries.length);
  for (const e of entries) {
    const d = document.createElement("div");
    d.className = "pin-item edit-item";
    const lbl = document.createElement("span");
    lbl.className = "lbl";
    const allEns = E.journal.joint_n_ens &&
      e.ens_first === 0 && e.ens_last >= E.journal.joint_n_ens - 1;
    const bins = e.bin_first === e.bin_last ? `bin ${e.bin_first}`
                                            : `bins ${e.bin_first}–${e.bin_last}`;
    lbl.textContent = `${e.head === "down" ? "DN" : "UP"} · ${bins} · ` +
      (allEns ? "all ens" : `ens ${e.ens_first}–${e.ens_last}`);
    if (e.note) {
      const sub = document.createElement("small");
      sub.textContent = e.note;
      lbl.appendChild(sub);
    }
    const x = document.createElement("span");
    x.className = "x";
    x.title = "remove this edit (re-solves)";
    x.textContent = "✕";
    x.addEventListener("click", () => deleteEdit(e.id));
    d.append(lbl, x);
    box.appendChild(d);
  }
  if (E.stale && entries.length) {
    const w = document.createElement("div");
    w.className = "hint stale-warn";
    w.textContent = `⚠ journal stale — not applied: ${E.stale}`;
    box.appendChild(w);
  }
}

/* --- the inset: live solution vs the no-edits baseline, same config ---------
 *
 * The point of brushing is the solution response, but on a clean cast a one-bin
 * mask moves the profile by well under 1 cm/s -- invisible at the main plot's
 * scale. The inset re-fetches the SAME configuration with `ignore_edits: true`
 * (cached server-side like any EditConfig) and shows live vs that baseline plus
 * the Δubar/Δvbar numbers, so every brush stroke has a visible, quantified
 * effect even when it is "the edit costs nothing" -- itself an answer. */

let baselineSeq = 0;                             // drop stale baseline fetches

async function ensureBaseline(p) {
  const key = S.station + "|" + body();
  if (p.manual_edits === 0) {                    // the live solve IS the baseline
    E.baseline = { key, ...p.profile };
    drawEditInset();
    return;
  }
  if (E.baseline && E.baseline.key === key) {
    drawEditInset();
    return;
  }
  drawEditInset();                               // show "capturing baseline…" meanwhile
  const mySeq = ++baselineSeq;
  try {
    const payload = JSON.parse(body());
    payload.ignore_edits = true;
    const r = await fetch(`api/station/${encodeURIComponent(S.station)}/solve`,
                          { method: "POST", body: JSON.stringify(payload),
                            headers: { "Content-Type": "application/json" } });
    if (!r.ok || mySeq !== baselineSeq) return;
    const b = await r.json();
    if (mySeq !== baselineSeq) return;
    E.baseline = { key, ...b.profile };
    drawEditInset();
  } catch { /* inset is best-effort; the next solve retries */ }
}

function drawEditInset() {
  const c = $("edit-inset");
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (!W || !H) return;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  g.fillStyle = "rgba(13,18,24,.88)";
  g.fillRect(0, 0, W, H);
  g.strokeStyle = "rgba(189,208,226,.35)";
  g.strokeRect(0.5, 0.5, W - 1, H - 1);
  g.font = '10px ui-monospace,"SF Mono",Consolas,monospace';
  if (!last) return;
  const base = E.baseline, live = last.profile;
  const padT = 30, padB = 8, padX = 8;
  const zs = live.z.filter(x => x !== null);
  const zmax = Math.max(...zs, 1);
  let vmax = 0.02;
  for (const a of [live.u, live.v, base ? base.u : [], base ? base.v : []])
    for (const x of a) if (x !== null) vmax = Math.max(vmax, Math.abs(x));
  vmax *= 1.1;
  const Y = zz => padT + zz / zmax * (H - padT - padB);
  const X = v => padX + (v + vmax) / (2 * vmax) * (W - 2 * padX);
  const trace = (zarr, comp) => {
    g.beginPath();
    let pen = false;
    for (let i = 0; i < zarr.length; i++) {
      if (comp[i] === null || zarr[i] === null) { pen = false; continue; }
      const x = X(comp[i]), y = Y(zarr[i]);
      pen ? g.lineTo(x, y) : g.moveTo(x, y); pen = true;
    }
    g.stroke();
  };
  g.strokeStyle = "rgba(189,208,226,.25)";       // zero line
  g.beginPath(); g.moveTo(X(0), padT); g.lineTo(X(0), H - padB); g.stroke();
  if (base) {
    g.setLineDash([4, 3]); g.lineWidth = 1;
    g.strokeStyle = "rgba(160,175,195,.7)";
    trace(base.z, base.u);
    trace(base.z, base.v);
    g.setLineDash([]);
  }
  g.lineWidth = 1.3;
  g.strokeStyle = "#39d3c8"; trace(live.z, live.u);
  g.strokeStyle = "#ff9e64"; trace(live.z, live.v);
  g.fillStyle = "#bdd0e2";
  g.fillText("live vs no-edits", 8, 13);
  if (base && base.ubar !== null && live.ubar !== null) {
    const du = (live.ubar - base.ubar) * 100, dv = (live.vbar - base.vbar) * 100;
    g.fillStyle = "#e8b153";
    g.fillText(`Δū ${du >= 0 ? "+" : ""}${du.toFixed(2)} · ` +
               `Δv̄ ${dv >= 0 ? "+" : ""}${dv.toFixed(2)} cm/s`, 8, 25);
  } else {
    g.fillStyle = "#7a8494";
    g.fillText(last.manual_edits ? "capturing baseline…" : "no edits active", 8, 25);
  }
}

new ResizeObserver(() => {
  if (E.open) { drawHeatOverlay(); drawEditInset(); }
}).observe($("heat-wrap"));
$("heat").addEventListener("load", drawHeatOverlay);

/* ------------------------------------------------------------------ boot */

(async function init() {
  try {
    const info = await (await fetch("api/stations")).json();
    const sel = $("station");
    for (const label of info.stations) {
      const o = document.createElement("option");
      o.value = o.textContent = label;
      sel.appendChild(o);
    }
    S.sadcp_sources = info.sadcp_sources || [];
    const srcSel = $("sel-sadcp");
    for (const s of S.sadcp_sources) {
      const o = document.createElement("option");
      o.value = s.key;
      o.textContent = `${s.key} · ${s.source}${s.origin === "found" ? " (found)" : ""}`;
      o.title = s.folder;
      srcSel.appendChild(o);
    }
    // an explicitly flagged source defaults the constraint ON; discovered-only
    // sources sit in the dropdown until the user turns the toggle on
    const flagged = S.sadcp_sources.find(s => s.origin !== "found");
    S.sadcp_key = (flagged || S.sadcp_sources[0] || {}).key || null;
    S.use_sadcp = Boolean(flagged);
    if (S.sadcp_key) srcSel.value = S.sadcp_key;
    syncSadcpControls();
    S.station = info.stations[0];
    sel.value = S.station;
    renderPins();
    await solve();
    if (location.hash === "#demo") demoPins();   // showcase pins/Δ (also used for QA shots)
    if (location.hash === "#edit") setView("edit");   // deep-link the brush view
  } catch (e) {
    status("err", `error: ${e.message}`);
  }
})();

/* #demo: pin the default solution, then re-solve with botfac 2 so the ghost,
 * the Δ-strip and the pin list are all populated in one page load. */
function demoPins() {
  $("pinbtn").click();
  const knob = document.querySelector('.knob[data-k="botfac"] input');
  knob.value = 20;
  knob.dispatchEvent(new Event("input"));
}
