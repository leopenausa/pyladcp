/* pyladcp studio — live single-station solve UI.
 *
 * One source of truth: `S` mirrors SessionConfig; every change debounces into
 * POST /api/station/{label}/solve and the canvas redraws from the JSON profile.
 * Axis legibility is a deliberate priority: bright axis ink, generous tick
 * labels, units on every pane (user requirement; more polish lands in PR 5). */
"use strict";

const $ = id => document.getElementById(id);

const S = {                                     // mirrors SessionConfig
  station: null,
  solver: "inverse",
  botfac: 1.0, barofac: 1.0, smoofac: 0.0, sadcpfac: 3.0,
  down_only: false,
  use_sadcp: false,
  sadcp_available: false,
};

let seq = 0;                                    // drop stale in-flight responses
let timer = null;
let last = null;                                // last solve payload (for redraws)

function status(cls, text) {
  const el = $("status");
  el.className = "solve-state " + cls;
  $("status-text").textContent = text;
}

function body() {
  return JSON.stringify({
    edit: { down_only: S.down_only },
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
    $("st-prep").className = "stage done";
    $("prep-ms").textContent = p.prepared ? "cached" : "built";
    $("st-solve").className = "stage done";
    $("solve-ms").textContent = `${p.solve_ms} ms`;
    status("live", `solved in ${p.solve_ms} ms`);
    $("ro-drot").textContent = p.drot === null ? "–"
      : `${p.drot.toFixed(2)}°${p.drot_source === "explicit" ? "" : " igrf"}`;
    $("ro-zbottom").textContent = p.zbottom === null ? "none" : `${Math.round(p.zbottom)} m`;
    $("ro-ubar").textContent = fmtVel(p.profile.ubar);
    $("ro-vbar").textContent = fmtVel(p.profile.vbar);
    $("cli").textContent = p.cli;
    draw(p);
  } catch (e) {
    if (mySeq === seq) status("err", `error: ${e.message}`);
  }
}

const fmtVel = x => x === null ? "–" : `${(x * 100).toFixed(1)} cm/s`;

/* ------------------------------------------------------------------ canvas */

const canvas = $("plot");
const ctx = canvas.getContext("2d");

const AXIS = "#9fb4c8";                          // high-contrast axis ink
const GRID = "rgba(159,180,200,.14)";
const ZERO = "rgba(159,180,200,.38)";
const FONT = '12px ui-monospace,"SF Mono",Consolas,monospace';

function niceStep(span, target) {
  const raw = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * mag) return m * mag;
  return 10 * mag;
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

  // symmetric velocity range over both components (shared scale aids comparison)
  let vmax = 0.05;
  for (const a of [u, v]) for (const x of a) if (x !== null) vmax = Math.max(vmax, Math.abs(x));
  if (p.bt) for (const a of [p.bt.u, p.bt.v]) for (const x of a)
    if (x !== null) vmax = Math.max(vmax, Math.abs(x));
  vmax *= 1.12;

  const padL = 64, padR = 16, padT = 30, padB = 40, gap = 36;
  const paneW = (W - padL - padR - gap) / 2;
  const panes = { u: [padL, padL + paneW], v: [padL + paneW + gap, W - padR] };
  const Y = zz => padT + zz / zmax * (H - padT - padB);
  const X = (val, pane) => pane[0] + (val + vmax) / (2 * vmax) * (pane[1] - pane[0]);

  // depth grid + labels (left pane carries the depth axis)
  const zstep = niceStep(zmax, 8);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let zz = 0; zz <= zmax; zz += zstep) {
    ctx.strokeStyle = GRID; ctx.beginPath();
    ctx.moveTo(panes.u[0], Y(zz)); ctx.lineTo(panes.v[1], Y(zz)); ctx.stroke();
    ctx.fillStyle = AXIS;
    ctx.fillText(zz ? `${zz}` : "0 m", panes.u[0] - 8, Y(zz));
  }

  for (const [name, pane] of Object.entries(panes)) {
    // frame + velocity ticks
    ctx.strokeStyle = GRID;
    ctx.strokeRect(pane[0], padT, pane[1] - pane[0], H - padT - padB);
    const vstep = niceStep(2 * vmax, 6);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let val = -Math.floor(vmax / vstep) * vstep; val <= vmax; val += vstep) {
      const x = X(val, pane);
      ctx.strokeStyle = Math.abs(val) < 1e-12 ? ZERO : GRID;
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, H - padB); ctx.stroke();
      ctx.fillStyle = AXIS;
      ctx.fillText(`${Math.round(val * 100)}`, x, H - padB + 6);
    }
    // pane title with units
    ctx.fillStyle = AXIS; ctx.textBaseline = "alphabetic";
    ctx.fillText(name === "u" ? "U  east  ·  cm s⁻¹" : "V  north  ·  cm s⁻¹",
                 (pane[0] + pane[1]) / 2, padT - 10);
  }

  // seabed
  if (p.zbottom !== null && p.zbottom <= zmax) {
    const y = Y(p.zbottom);
    ctx.fillStyle = "rgba(255,209,102,.07)";
    ctx.fillRect(panes.u[0], y, panes.v[1] - panes.u[0], H - padB - y);
    ctx.strokeStyle = "rgba(255,209,102,.5)";
    ctx.setLineDash([6, 4]); ctx.beginPath();
    ctx.moveTo(panes.u[0], y); ctx.lineTo(panes.v[1], y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(255,209,102,.75)"; ctx.textAlign = "right";
    ctx.fillText(`seabed ${Math.round(p.zbottom)} m`, panes.v[1] - 6, y - 6);
  }

  // ±1σ band on u
  ctx.fillStyle = "rgba(57,211,200,.13)";
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < z.length; i++) {
    if (u[i] === null || ue[i] === null) continue;
    const x = X(u[i] - ue[i], panes.u), y = Y(z[i]);
    started ? ctx.lineTo(x, y) : ctx.moveTo(x, y); started = true;
  }
  for (let i = z.length - 1; i >= 0; i--) {
    if (u[i] === null || ue[i] === null) continue;
    ctx.lineTo(X(u[i] + ue[i], panes.u), Y(z[i]));
  }
  if (started) ctx.fill();

  // bottom-track reference (dots)
  if (p.bt) {
    ctx.fillStyle = "rgba(255,209,102,.8)";
    for (let i = 0; i < p.bt.z.length; i++) {
      for (const [comp, pane] of [[p.bt.u, panes.u], [p.bt.v, panes.v]]) {
        if (comp[i] === null) continue;
        ctx.beginPath();
        ctx.arc(X(comp[i], pane), Y(p.bt.z[i]), 2.1, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }

  // live profiles
  for (const [comp, pane, color] of [[u, panes.u, "#39d3c8"], [v, panes.v, "#ff9e64"]]) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    let pen = false;
    for (let i = 0; i < z.length; i++) {
      if (comp[i] === null) { pen = false; continue; }
      const x = X(comp[i], pane), y = Y(z[i]);
      pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y); pen = true;
    }
    ctx.stroke(); ctx.lineWidth = 1;
  }
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

$("station").addEventListener("change", () => {
  S.station = $("station").value;
  last = null;
  scheduleSolve(0);
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
    solve();
  } catch (e) {
    status("err", `error: ${e.message}`);
  }
})();
