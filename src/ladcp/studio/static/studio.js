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
  nearfield: null,                              // null = preset, [] = none, [3,4] = bins
  dzbelow: null,                                // null = preset
  use_sadcp: false,
  sadcp_available: false,
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
  return { down_only: S.down_only, nearfield_dn_bins: S.nearfield, dzbelow: S.dzbelow };
}

function body() {
  return JSON.stringify({
    edit: editBody(),
    solve: { solver: S.solver, botfac: S.botfac, barofac: S.barofac,
             smoofac: S.smoofac, sadcpfac: S.sadcpfac },
    use_sadcp: S.use_sadcp,
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
    renderQaList(p.panels);
    draw(p);
  } catch (e) {
    if (mySeq === seq) status("err", `error: ${e.message}`);
  }
}

const fmtVel = x => x === null ? "–" : `${(x * 100).toFixed(1)} cm/s`;

/* ------------------------------------------------------------------ pins */

function pinLabel() {
  const parts = [S.solver];
  if (S.botfac !== 1.0) parts.push(`botfac ${S.botfac}`);
  if (S.barofac !== 1.0) parts.push(`barofac ${S.barofac}`);
  if (S.smoofac !== 0.0) parts.push(`smoofac ${S.smoofac}`);
  if (S.use_sadcp) parts.push(`sadcp ${S.sadcpfac}`);
  if (S.down_only) parts.push("down-only");
  if (S.nearfield !== null)
    parts.push(`nf ${S.nearfield.length ? S.nearfield.join(",") : "none"}`);
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
  if (!S.sadcp_available) return;
  S.use_sadcp = !S.use_sadcp;
  $("tgl-sadcp").classList.toggle("on", S.use_sadcp);
  scheduleSolve(0);
});

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

bindEditField($("in-nearfield"), text => {
  if (text === "") { S.nearfield = null; return; }
  if (text.toLowerCase() === "none") { S.nearfield = []; return; }
  S.nearfield = text.split(",").map(s => {
    const n = Number(s.trim());
    if (!Number.isInteger(n) || n < 1) throw new Error("bad bin");
    return n;
  });
});

bindEditField($("in-dzbelow"), text => {
  if (text === "") { S.dzbelow = null; return; }
  const x = Number(text);
  if (!Number.isFinite(x) || x < 0) throw new Error("bad dzbelow");
  S.dzbelow = x;
});

$("station").addEventListener("change", () => {
  S.station = $("station").value;
  last = null;
  clearPins();                                   // pins are per-station (z grids differ)
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
    S.sadcp_available = info.sadcp;
    S.use_sadcp = info.sadcp;                    // launched with --sadcp -> on by default
    const tgl = $("tgl-sadcp");
    tgl.classList.toggle("on", S.use_sadcp);
    tgl.classList.toggle("disabled", !info.sadcp);
    $("sadcp-note").textContent = info.sadcp ? info.sadcp_folder : "launch with --sadcp";
    S.station = info.stations[0];
    sel.value = S.station;
    renderPins();
    await solve();
    if (location.hash === "#demo") demoPins();   // showcase pins/Δ (also used for QA shots)
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
