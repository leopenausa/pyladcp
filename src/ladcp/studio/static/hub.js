/* pyladcp cruise hub — setup wizard + dashboard over /api/hub/* (wizard phase E).
   Same engine as the terminal (`ladcp init` / `ladcp status` / `ladcp process`):
   this page only renders proposals and presses the same buttons. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
async function api(path, body) {
  const opts = body ? {method: "POST", headers: {"Content-Type": "application/json"},
                       body: JSON.stringify(body)} : {};
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

// ---------------------------------------------------------------------------
// entry: wizard when unconfigured, dashboard otherwise

async function init() {
  const st = await api("/api/hub/state");
  $("#crumbs").innerHTML = `<b>${st.dir}</b>`;
  if (st.configured) dashboard(); else wizard();
}

// ---------------------------------------------------------------------------
// the setup wizard

function steps(active) {
  const s = el("div", "steps");
  ["1 scan", "2 confirm", "3 save", "4 trial"].forEach((t, i) => {
    const sp = el("span", i === active ? "on" : "", t);
    s.appendChild(sp);
  });
  return s;
}

async function wizard() {
  const main = $("#main");
  main.replaceChildren(steps(0), el("p", "note", "scanning the cruise directory (filenames only, nothing is written)…"));
  let det;
  try { det = await api("/api/hub/detect"); }
  catch (e) { main.appendChild(el("div", "err", String(e))); return; }
  confirmStep(det);
}

function confirmStep(det) {
  const main = $("#main");
  main.replaceChildren(steps(1));

  // LADCP card
  const lad = el("div", "card");
  lad.appendChild(el("h2", "", "LADCP casts"));
  lad.appendChild(el("div", "ev",
    det.ladcp.dir === null ? det.ladcp.evidence
      : `${det.ladcp.dir || "."}/ — ${det.ladcp.evidence} (${det.ladcp.layout} layout)`));
  if (det.ladcp.stations.length) {
    const t = el("table", "st");
    t.innerHTML = "<tr><th>station</th><th>up-looker</th><th>ctd</th></tr>";
    const missing = new Set(det.ctd.missing_cnv);
    det.ladcp.stations.forEach((s) => {
      const tr = el("tr");
      tr.appendChild(el("td", "", s.label));
      tr.appendChild(el("td", "", s.slave ? "yes" : "— (single-head)"));
      tr.appendChild(el("td", "", missing.has(s.label) ? "no .cnv" : "yes"));
      t.appendChild(tr);
    });
    lad.appendChild(t);
  }
  main.appendChild(lad);

  // CTD card
  const ctd = el("div", "card");
  ctd.appendChild(el("h2", "", "CTD"));
  ctd.appendChild(el("div", "ev",
    det.ctd.dir === null ? det.ctd.evidence : `${det.ctd.dir || "."}/ — ${det.ctd.evidence}`));
  let fromHex = null;
  // offer conversion when any cast lacks a cleaned .cnv — incl. MASTER/SLAVE archives
  // (no name-paired stations, so missing_cnv is empty but n_cnv says it all)
  if (det.ctd.n_hex > 0 && (det.ctd.missing_cnv.length || det.ctd.n_cnv === 0)) {
    const l = el("label");
    fromHex = el("input");
    fromHex.type = "checkbox";
    fromHex.disabled = !det.ctd.converter;
    fromHex.checked = !!det.ctd.converter;
    l.appendChild(fromHex);
    l.appendChild(el("span", "", det.ctd.converter
      ? `convert raw .hex on the fly (CTD_project found at ${det.ctd.converter})`
      : "raw .hex found but the CTD_project converter is not installed"));
    ctd.appendChild(l);
  }
  main.appendChild(ctd);

  // ship-ADCP card (never auto-chosen — the science is the user's call)
  const sad = el("div", "card");
  sad.appendChild(el("h2", "", "ship-ADCP constraint"));
  sad.appendChild(el("div", "ev", det.sadcp.length
    ? "pick the source to constrain the inverse with:"
    : "no candidate sources found (a [sadcp] table can be added to cruise.toml later)"));
  const radios = [];
  const mk = (value, text) => {
    const l = el("label");
    const r = el("input");
    r.type = "radio"; r.name = "sadcp"; r.value = value;
    l.appendChild(r); l.appendChild(el("span", "", text));
    sad.appendChild(l); radios.push(r);
    return r;
  };
  mk("", "no ship-ADCP constraint").checked = true;
  det.sadcp.forEach((c, i) => mk(String(i), `${c.path}  (${c.source}: ${c.evidence})`));
  let navSel = null;                 // clock check only makes sense with a raw source
  if (det.nav.length && det.sadcp.some((c) => c.source === "vmdas")) {
    const l = el("label");
    navSel = el("input");
    navSel.type = "checkbox";
    l.appendChild(navSel);
    l.appendChild(el("span", "",
      `clock check against ${det.nav[0].path} (--sadcp-timeoff auto; raw VmDAS only)`));
    sad.appendChild(l);
  }
  main.appendChild(sad);

  // cruise identity card
  const cru = el("div", "card");
  cru.appendChild(el("h2", "", "cruise"));
  const nameIn = el("input"); nameIn.type = "text"; nameIn.value = det.name;
  const outIn = el("input"); outIn.type = "text"; outIn.value = "qa_out";
  const row1 = el("label"); row1.appendChild(el("span", "", "name")); row1.appendChild(nameIn);
  const row2 = el("label"); row2.appendChild(el("span", "", "output dir")); row2.appendChild(outIn);
  cru.appendChild(row1); cru.appendChild(row2);
  cru.appendChild(el("div", "note", det.preset
    ? "a registered params preset matches this name — its layers apply"
    : "no registered preset — generic operator defaults apply; any CastParams field "
      + "can be overridden per cruise or per station in [params]"));
  main.appendChild(cru);

  const act = el("div", "actions");
  const btn = el("button", "btn", "preview cruise.toml →");
  const errBox = el("div", "err");
  act.appendChild(btn);
  main.appendChild(act);
  main.appendChild(errBox);

  btn.onclick = async () => {
    const raw = {cruise: {name: nameIn.value.trim() || det.name},
                 data: {root: ".", out: outIn.value.trim() || "qa_out"}};
    if (fromHex && fromHex.checked) raw.ctd = {from_hex: true};
    const pick = radios.find((r) => r.checked && r.value !== "");
    if (pick) {
      const c = det.sadcp[Number(pick.value)];
      raw.sadcp = {folder: c.path, source: c.source};
      if (navSel && navSel.checked && c.source === "vmdas") {
        raw.sadcp.nav = det.nav[0].path;
        raw.sadcp.timeoff = "auto";
      }
    }
    try {
      const prev = await api("/api/hub/preview", raw);
      saveStep(det, raw, prev.toml);
    } catch (e) { errBox.textContent = String(e); }
  };
}

function saveStep(det, raw, toml) {
  const main = $("#main");
  main.replaceChildren(steps(2));
  const card = el("div", "card");
  card.appendChild(el("h2", "", "cruise.toml to be written"));
  card.appendChild(el("pre", "toml", toml));
  const buildIdx = det.ctd.n_hex > 0 && det.ctd.dir !== null;
  card.appendChild(el("div", "note", buildIdx
    ? "saving also builds the archive index (scans PD0 headers once)"
    : "no raw CTD .hex anchors — the archive index is skipped; stations are "
      + "enumerated by filename"));
  const act = el("div", "actions");
  const back = el("button", "btn", "← back");
  const save = el("button", "btn", "save & continue →");
  const errBox = el("div", "err");
  act.appendChild(back); act.appendChild(save);
  card.appendChild(act); card.appendChild(errBox);
  main.appendChild(card);
  back.onclick = () => confirmStep(det);
  save.onclick = async () => {
    save.disabled = true; save.textContent = "writing…";
    try {
      const res = await api("/api/hub/config",
        {config: raw, build_index: buildIdx,
         ladcp_dir: det.ladcp.dir || ".", ctd_dir: det.ctd.dir || "."});
      trialStep(det, res);
    } catch (e) { errBox.textContent = String(e); save.disabled = false;
                  save.textContent = "save & continue →"; }
  };
}

function trialStep(det, saved) {
  const main = $("#main");
  main.replaceChildren(steps(3));
  const card = el("div", "card");
  card.appendChild(el("h2", "", "trial station (suggested, skippable)"));
  let note = `wrote ${saved.written}`;
  if (saved.indexed !== null && saved.indexed !== undefined)
    note += ` · indexed ${saved.indexed} cast(s)`;
  if (saved.index_error) note += ` · index build failed: ${saved.index_error}`;
  card.appendChild(el("div", "ev", note));
  const labels = det.ladcp.stations.map((s) => s.label);
  if (!labels.length) {
    card.appendChild(el("div", "note", "no name-pairable stations to trial — go to the dashboard"));
  }
  const sel = el("select", "station-select");
  labels.forEach((l) => { const o = el("option", "", l); o.value = l; sel.appendChild(o); });
  if (labels.length) sel.value = labels[Math.floor(labels.length / 2)];
  const act = el("div", "actions");
  const run = el("button", "btn", "process trial station");
  const skip = el("button", "btn", "skip → dashboard");
  if (labels.length) { act.appendChild(sel); act.appendChild(run); }
  act.appendChild(skip);
  const out = el("div");
  card.appendChild(act); card.appendChild(out);
  main.appendChild(card);
  skip.onclick = dashboard;
  run.onclick = async () => {
    run.disabled = true;
    try {
      await api("/api/hub/process", {stations: [sel.value]});
      await watchJob(() => showScorecard(sel.value, out, card));
    } catch (e) { out.appendChild(el("div", "err", String(e))); run.disabled = false; }
  };
}

async function showScorecard(label, out, card) {
  try {
    const sc = await api(`/api/hub/scorecard/${label}`);
    out.replaceChildren(el("pre", "score", sc.text));
    if (sc.pdf) {
      const a = el("a", "", `open ${label}_report.pdf`);
      a.href = `/api/hub/report/${label}`; a.target = "_blank";
      out.appendChild(a);
    }
  } catch (e) { out.replaceChildren(el("div", "err", String(e))); }
  const act = el("div", "actions");
  const go = el("button", "btn", "go to the dashboard →");
  go.onclick = dashboard;
  act.appendChild(go);
  card.appendChild(act);
}

// ---------------------------------------------------------------------------
// the dashboard

async function dashboard() {
  const main = $("#main");
  $("#editor-link").style.display = "";
  main.replaceChildren(el("p", "note", "gathering status…"));
  let d;
  try { d = await api("/api/hub/status"); }
  catch (e) { main.replaceChildren(el("div", "err", String(e))); return; }
  $("#crumbs").innerHTML = `cruise <b>${d.cruise}</b> · ${d.config}`;
  main.replaceChildren();

  // pending casts
  const pend = d.stations.filter((s) => s.freshness !== "fresh");
  const p1 = el("div", "card");
  p1.appendChild(el("h2", "", "processing"));
  const f = d.freshness;
  p1.appendChild(el("div", "ev",
    `${d.n_stations} cast(s) — ${f.fresh} fresh, ${f.stale} stale, ${f.missing} unprocessed`));
  pend.slice(0, 12).forEach((s) => {
    const r = el("div", "row");
    r.appendChild(el("span", `badge ${s.freshness}`, s.freshness));
    r.appendChild(el("span", "lbl", s.label));
    r.appendChild(el("span", "why", s.reason));
    p1.appendChild(r);
  });
  const act = el("div", "actions");
  const bNew = el("button", "btn", pend.length ? `process ${pend.length} pending` : "nothing pending");
  bNew.disabled = !pend.length;
  const bAll = el("button", "btn", "process all");
  const errBox = el("div", "err");
  act.appendChild(bNew); act.appendChild(bAll);
  p1.appendChild(act); p1.appendChild(errBox);
  main.appendChild(p1);
  const kick = (body) => async () => {
    try { await api("/api/hub/process", body); watchJob(dashboard); }
    catch (e) { errBox.textContent = String(e); }
  };
  bNew.onclick = kick({mode: "new"});
  bAll.onclick = kick({mode: "all"});

  // QA rollup
  const q = d.qa, nproc = q.ok + q.warn + q.fail;
  if (nproc) {
    const p2 = el("div", "card");
    p2.appendChild(el("h2", "", "QA"));
    p2.appendChild(el("div", "ev", `${q.ok} ok, ${q.warn} warn, ${q.fail} fail (${nproc} processed)`));
    const scored = d.stations.filter((s) => s.qa)
      .sort((a, b) => ({fail: 0, warn: 1, ok: 2}[a.qa] - {fail: 0, warn: 1, ok: 2}[b.qa]));
    scored.forEach((s) => {
      const r = el("div", "row");
      r.appendChild(el("span", `badge ${s.qa}`, s.qa));
      r.appendChild(el("span", "lbl", s.label));
      r.appendChild(el("span", "why", s.problems.join(", ")));
      const rep = el("a", "", "report");
      rep.href = `/api/hub/report/${s.label}`; rep.target = "_blank";
      const proc = el("a", "", "re-process");
      proc.href = "#";
      proc.onclick = (ev) => { ev.preventDefault(); kick({stations: [s.label]})(); };
      r.appendChild(rep); r.appendChild(proc);
      p2.appendChild(r);
    });
    main.appendChild(p2);
  }

  // loose ends
  const loose = [];
  d.stations.forEach((s) => s.loose_ends.forEach((le) => loose.push([s.label, le])));
  if (loose.length || d.index_stale) {
    const p3 = el("div", "card");
    p3.appendChild(el("h2", "", "loose ends"));
    loose.forEach(([label, le]) => {
      const r = el("div", "row");
      r.appendChild(el("span", "lbl", label));
      r.appendChild(el("span", "why", le));
      p3.appendChild(r);
    });
    if (d.index_stale)
      p3.appendChild(el("div", "note",
        "PD0 files newer than the archive index — rebuild with ladcp-index (or re-run setup)"));
    main.appendChild(p3);
  }
}

// ---------------------------------------------------------------------------
// job polling (shared by the trial step and the dashboard buttons)

async function watchJob(onDone) {
  const bar = $("#jobbar"), fill = $("#jobfill"), text = $("#jobtext");
  bar.style.display = "block";
  for (;;) {
    let j;
    try { j = await api("/api/hub/job"); }
    catch { break; }
    const done = j.done.length;
    fill.style.width = j.total ? `${(100 * done) / j.total}%` : "0";
    const tally = j.done.map((r) => `${r.label} [${r.status}]`).slice(-3).join("  ");
    text.textContent = `processing ${done}/${j.total}` +
      (j.current ? ` · running ${j.current}` : "") + (tally ? ` · ${tally}` : "");
    if (!j.running) {
      text.textContent = `done: ${tally || "nothing"}` + (j.error ? ` · ${j.error}` : "");
      setTimeout(() => { bar.style.display = "none"; }, 4000);
      break;
    }
    await new Promise((res) => setTimeout(res, 1000));
  }
  if (onDone) onDone();
}

init().catch((e) => { $("#main").replaceChildren(el("div", "err", String(e))); });
