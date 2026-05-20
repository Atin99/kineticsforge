var activeBmsRun = 0;
var kfAssistantHistory = [];
var KF_HISTORY_KEY = "kineticsforge.runHistory.v1";
var kfRunHistory = loadRunHistory();
window.__kfAssistantHistory = kfAssistantHistory;
window.__kfRunHistory = kfRunHistory;

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function relu(x) { return Math.max(0, x); }
function expClamp(x, lo, hi) { return Math.exp(clamp(x, lo, hi)); }
function fmt(x, d) { return Number.isFinite(x) ? x.toFixed(d == null ? 2 : d) : "--"; }
function gaussian() {
  var u = 1 - Math.random();
  var v = 1 - Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Seeded PRNG: mulberry32 — deterministic 32-bit PRNG for reproducible BMS simulation
function mulberry32(seed) {
  var state = seed | 0;
  return function () {
    state = (state + 0x6D2B79F5) | 0;
    var t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function seededGaussian(rng) {
  var u = 1 - rng();
  var v = 1 - rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c];
  });
}
function num(id, fallback) {
  var el = document.getElementById(id);
  if (!el) return fallback;
  var v = parseFloat(el.value);
  return Number.isFinite(v) ? v : fallback;
}
function setHtml(id, html) {
  var el = document.getElementById(id);
  if (el) el.innerHTML = html;
}
function showToast(message, kind) {
  var dock = document.getElementById("toast-dock");
  if (!dock) {
    dock = document.createElement("div");
    dock.id = "toast-dock";
    dock.className = "toast-dock";
    document.body.appendChild(dock);
  }
  var note = document.createElement("div");
  note.className = "toast " + (kind || "info");
  note.textContent = message;
  dock.appendChild(note);
  setTimeout(function () { note.classList.add("show"); }, 20);
  setTimeout(function () {
    note.classList.remove("show");
    setTimeout(function () { note.remove(); }, 260);
  }, 4200);
}
function setReadouts(id, items) {
  var el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = items.map(function (it) {
    return '<div class="readout"><div class="k">' + escapeHtml(it.k) + '</div><div class="v">' + escapeHtml(it.v) + '</div></div>';
  }).join("");
}
function downloadCSV(rows, filename) {
  if (!rows || !rows.length) return;
  var keys = Object.keys(rows[0]);
  var body = [keys.join(",")].concat(rows.map(function (row) {
    return keys.map(function (k) {
      var v = row[k] == null ? "" : String(row[k]);
      return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
    }).join(",");
  })).join("\n");
  var blob = new Blob([body], { type: "text/csv;charset=utf-8" });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadJSON(payload, filename) {
  if (!payload) return;
  var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadText(text, filename, type) {
  var blob = new Blob([String(text || "")], { type: type || "text/plain;charset=utf-8" });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function loadRunHistory() {
  try {
    var raw = localStorage.getItem(KF_HISTORY_KEY);
    var parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.slice(0, 40) : [];
  } catch (err) {
    return [];
  }
}

function saveRunHistory() {
  try {
    localStorage.setItem(KF_HISTORY_KEY, JSON.stringify(kfRunHistory.slice(0, 40)));
  } catch (err) {
    // Local storage can be unavailable in locked-down browsers. History is optional.
  }
}

function runKindLabel(kind) {
  return ({
    diagnostics: "Diagnostics",
    upload: "Upload",
    upload_compare: "Upload A/B",
    upload_batch: "Batch Upload",
    bms: "BMS",
    bms_sweep: "BMS Sweep",
    materials: "Materials",
    recycling: "Recycling"
  })[kind] || kind;
}

function compactNumber(x, d) {
  if (!Number.isFinite(Number(x))) return "--";
  var n = Number(x);
  if (Math.abs(n) >= 1000000) return (n / 1000000).toFixed(1) + "M";
  if (Math.abs(n) >= 1000) return (n / 1000).toFixed(1) + "k";
  return n.toFixed(d == null ? 2 : d);
}

function mean(arr) {
  if (!arr || !arr.length) return 0;
  var s = 0;
  for (var i = 0; i < arr.length; i++) s += Number(arr[i]) || 0;
  return s / arr.length;
}

function stddev(arr) {
  if (!arr || arr.length < 2) return 0;
  var m = mean(arr);
  var s2 = 0;
  for (var i = 0; i < arr.length; i++) {
    var d = (Number(arr[i]) || 0) - m;
    s2 += d * d;
  }
  return Math.sqrt(s2 / arr.length);
}

function formatConfidence(confidence) {
  if (!Number.isFinite(Number(confidence))) return "--";
  return Math.round(clamp(Number(confidence), 0, 1) * 100) + "%";
}

function recordRun(kind, summary) {
  var payload = Object.assign({
    id: Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 7),
    kind: kind,
    label: runKindLabel(kind),
    at: new Date().toISOString()
  }, summary || {});
  kfRunHistory.unshift(payload);
  kfRunHistory = kfRunHistory.slice(0, 40);
  window.__kfRunHistory = kfRunHistory;
  saveRunHistory();
  renderDecisionConsole();
}

function latestRunLabel() {
  if (!kfRunHistory.length) return "None";
  var r = kfRunHistory[0];
  return r.label + " " + new Date(r.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function severityRank(level) {
  return ({ critical: 0, warn: 1, ok: 2, info: 3 })[level] == null ? 4 : ({ critical: 0, warn: 1, ok: 2, info: 3 })[level];
}

function severityTag(level) {
  if (level === "critical") return '<span class="tag critical">Critical</span>';
  if (level === "warn") return '<span class="tag warn">Watch</span>';
  if (level === "ok") return '<span class="tag ok">Ready</span>';
  return '<span class="tag">Info</span>';
}

function addDecision(items, item) {
  items.push(Object.assign({
    severity: "info",
    source: "Workbench",
    owner: "Operator",
    evidence: "--",
    action: "Run a panel to create evidence.",
    next: "No experiment ticket yet.",
    confidence: null
  }, item || {}));
}

function topLoss(out) {
  if (!out || !out.cap) return null;
  return [
    ["SEI/desolvation", out.sei[out.sei.length - 1]],
    ["P2-O2", out.p2[out.p2.length - 1]],
    ["Jahn-Teller", out.jt[out.jt.length - 1]],
    ["rate stress", out.rate[out.rate.length - 1]],
    ["residual", out.residual[out.residual.length - 1]]
  ].sort(function (a, b) { return b[1] - a[1]; })[0];
}

function buildDecisionItems() {
  var items = [];
  var diag = window.__kfDiag;
  if (diag && diag.cap && diag.cap.length) {
    var eol = diag.cap[diag.cap.length - 1];
    var fade = (1 - eol) * 100;
    var top = topLoss(diag);
    var diagConf = computeDiagConfidence(diag);
    var conf = diagConf.confidence;
    var residual = diag.residual[diag.residual.length - 1];
    var totalLoss = diag.sei[diag.sei.length - 1] + diag.p2[diag.p2.length - 1] + diag.jt[diag.jt.length - 1] + diag.rate[diag.rate.length - 1] + residual;
    var residualFrac = totalLoss > 1e-9 ? residual / totalLoss : 0;
    var sev = eol < 0.80 ? "critical" : (fade > 12 || residualFrac > 0.30 ? "warn" : "ok");
    var action = eol < 0.80
      ? "Do not advance this operating point without a lower-stress validation run."
      : residualFrac > 0.30
        ? "Calibrate the residual channel against measured cycle-capacity rows."
        : "Use this as the baseline degradation case for the next comparison.";
    addDecision(items, {
      severity: sev,
      source: "Diagnostics",
      owner: "Cell R&D",
      evidence: "EOL " + eol.toFixed(3) + ", fade " + fade.toFixed(1) + "%, dominant " + top[0],
      action: action,
      next: "Cycle 3 replicate cells at " + num("temp-slider", 45).toFixed(0) + " C and " + num("crate-slider", 1).toFixed(1) + "C; compare measured fade to the exported curve.",
      confidence: conf,
      confidence_detail: diagConf.detail
    });
  }

  var bms = window.__kfBms;
  if (bms && bms.frames && bms.frames.length) {
    var frame = bms.frames[bms.frames.length - 1];
    var tmax = Math.max.apply(null, frame.temps);
    var risk = Number(frame.maxRisk);
    if (!Number.isFinite(risk)) risk = 0;
    var gate = bms.threshold || 0.42;
    var bmsConf = computeBmsConfidence(bms, frame);
    var bmsSev = risk >= gate ? "critical" : risk >= gate * 0.80 ? "warn" : "ok";
    addDecision(items, {
      severity: bmsSev,
      source: "BMS",
      owner: "Pack Safety",
      evidence: "C" + frame.maxCell + " risk " + risk.toFixed(3) + " / gate " + gate.toFixed(2) + ", Tmax " + tmax.toFixed(1) + " C, seed " + (bms.seed == null ? "--" : bms.seed),
      action: risk >= gate
        ? "Cool or isolate C" + frame.maxCell + " first; inspect impedance rise before assuming pack-wide failure."
        : "Keep monitoring; this run stays below the configured action gate.",
      next: "Repeat the run across 5 seeds and one lower threshold to estimate false-negative sensitivity before pack policy changes.",
      confidence: bmsConf.confidence,
      confidence_detail: bmsConf.detail
    });
  }
  var sweep = window.__kfBmsSweep;
  if (sweep && Number.isFinite(sweep.meanRisk) && Number.isFinite(sweep.stdRisk) && Number.isFinite(sweep.alertRate)) {
    var sameScenario = !bms || !bms.scenarioKey || !sweep.scenarioKey || bms.scenarioKey === sweep.scenarioKey;
    if (sameScenario) {
      var sweepSev = sweep.alertRate >= 0.7 ? "warn" : sweep.alertRate <= 0.15 ? "ok" : "info";
      var sweepConf = clamp(0.74 - sweep.stdRisk * 1.8 - Math.abs(sweep.alertRate - 0.5) * 0.10, 0.26, 0.86);
      addDecision(items, {
        severity: sweepSev,
        source: "BMS Sweep",
        owner: "Pack Safety",
        evidence: sweep.count + " seeds, alert rate " + (sweep.alertRate * 100).toFixed(1) + "%, mean maxRisk " + sweep.meanRisk.toFixed(3) + " +/- " + sweep.stdRisk.toFixed(3),
        action: sweep.alertRate >= 0.7
          ? "This configuration trips frequently across seeds; tighten cooling or reduce thermal stress before policy rollout."
          : sweep.alertRate <= 0.15
            ? "This configuration is robust across sampled seeds; keep it as a baseline while verifying with measured telemetry."
            : "Seed sensitivity is moderate; keep this setup in watch mode and validate against additional pack logs.",
        next: "Run one physical telemetry capture with the same threshold and compare observed alert frequency to this sweep.",
        confidence: sweepConf,
        confidence_detail: "Seed sweep variability check; higher sigma reduces reliability of one-off seeded runs."
      });
    }
  }

  var mat = window.__kfMaterials;
  if (mat && Number.isFinite(mat.score)) {
    var matConf = computeMaterialsConfidence(mat);
    var synthReady = mat.stability > 0.72 && mat.fade500 < 0.16 && mat.chargeRisk < 0.28 && mat.oxygenRisk < 0.45;
    var matSev = synthReady ? "ok" : (mat.oxygenRisk > 0.55 || mat.chargeRisk > 0.45 || mat.fade500 > 0.24 ? "critical" : "warn");
    var blockers = [];
    if (mat.oxygenRisk > 0.45) blockers.push("oxygen risk");
    if (mat.chargeRisk > 0.28) blockers.push("charge balance");
    if (mat.fade500 > 0.16) blockers.push("fade@500");
    if (mat.stability <= 0.72) blockers.push("stability");
    addDecision(items, {
      severity: matSev,
      source: "Materials",
      owner: "Synthesis",
      evidence: "score " + mat.score.toFixed(3) + ", stability " + mat.stability.toFixed(2) + ", fade@500 " + (mat.fade500 * 100).toFixed(1) + "%",
      action: synthReady ? "Move this composition into a small synthesis queue." : "Keep in simulation queue until " + blockers.slice(0, 2).join(" and ") + " improve.",
      next: synthReady
        ? "Prepare a coin-cell batch with XRD phase check before cycling."
        : "Run the same composition with Al/Ti toggles and lower upper-voltage stress; export the best candidate.",
      confidence: matConf.confidence,
      confidence_detail: matConf.detail
    });
  }

  var rec = window.__kfRecycling;
  if (rec && Number.isFinite(rec.totalRecovered)) {
    var recConf = computeRecyclingConfidence(rec);
    var mn = rec.targets && rec.targets.length ? rec.targets[0] : 0;
    var recSev = rec.marginProxy < 0 ? "critical" : (mn < 0.82 || rec.purity < 0.86 ? "warn" : "ok");
    addDecision(items, {
      severity: recSev,
      source: "Recycling",
      owner: "Process",
      evidence: "recovered " + rec.totalRecovered.toFixed(1) + " kg, Mn " + (mn * 100).toFixed(1) + "%, purity " + (rec.purity * 100).toFixed(1) + "%, margin INR " + compactNumber(rec.marginProxy, 0),
      action: recSev === "ok" ? "Recipe clears the current pilot gate." : "Do not pilot as-is; improve recovery, purity, or cost first.",
      next: recSev === "ok"
        ? "Run a bench leach with duplicate assays for Mn/Fe/Na and compare against the exported 90% interval."
        : "Sweep acid concentration, time, and particle size before committing black mass.",
      confidence: recConf.confidence,
      confidence_detail: recConf.detail
    });
  }

  var byod = window.__kfByod;
  if (byod && byod.predictions) {
    var uploadConf = computeUploadConfidence(byod);
    var pred = byod.predictions || {};
    var warnings = Array.isArray(byod.warnings) ? byod.warnings : [];
    var soh = Number(pred.soh);
    var c = Number(pred.confidence);
    var uploadSev = warnings.length || c < 0.55 ? "warn" : (soh < 0.85 ? "critical" : "ok");
    addDecision(items, {
      severity: uploadSev,
      source: "Upload",
      owner: "Data",
      evidence: (byod.filename || "upload") + ": SOH " + (Number.isFinite(soh) ? (soh * 100).toFixed(1) + "%" : "--") + ", confidence " + (Number.isFinite(c) ? (c * 100).toFixed(0) + "%" : "--") + ", warnings " + warnings.length,
      action: warnings.length ? "Fix parser/data warnings before making a cell claim." : "Use the extracted features as the measured-data anchor.",
      next: "Export JSON, attach the original cycler file, and rerun after adding temperature and impedance columns if available.",
      confidence: uploadConf.confidence,
      confidence_detail: uploadConf.detail
    });
  }

  if (!items.length) {
    addDecision(items, {
      severity: "info",
      source: "Workbench",
      owner: "Operator",
      evidence: "No current panel outputs.",
      action: "Run Diagnostics, BMS, Materials, Recycling, or Upload to populate the action queue.",
      next: "Start with Diagnostics for cell chemistry or BMS for pack safety.",
      confidence: null
    });
  }
  items.sort(function (a, b) { return severityRank(a.severity) - severityRank(b.severity); });
  window.__kfDecisionItems = items;
  return items;
}

function renderDecisionConsole() {
  var body = document.getElementById("decision-action-body");
  if (!body) return;
  var items = buildDecisionItems();
  var open = items.filter(function (x) { return x.severity === "critical" || x.severity === "warn"; }).length;
  var critical = items.filter(function (x) { return x.severity === "critical"; }).length;
  var confs = items.map(function (x) { return Number(x.confidence); }).filter(Number.isFinite);
  var avgConf = confs.length ? confs.reduce(function (a, b) { return a + b; }, 0) / confs.length : null;
  var set = function (id, value) {
    var el = document.getElementById(id);
    if (el) el.textContent = value;
  };
  set("decision-open", String(open));
  set("decision-critical", String(critical));
  set("decision-confidence", avgConf == null ? "--" : Math.round(avgConf * 100) + "%");
  set("decision-last-run", latestRunLabel());
  body.innerHTML = items.map(function (it) {
    var confDetail = it.confidence_detail ? '<div class="decision-conf-note">' + escapeHtml(it.confidence_detail) + "</div>" : "";
    return "<tr>"
      + "<td>" + severityTag(it.severity) + "</td>"
      + "<td>" + escapeHtml(it.source) + "</td>"
      + "<td>" + escapeHtml(it.evidence) + "</td>"
      + "<td>" + escapeHtml(formatConfidence(it.confidence)) + confDetail + "</td>"
      + "<td>" + escapeHtml(it.action) + "</td>"
      + "<td>" + escapeHtml(it.owner) + "</td>"
      + "</tr>";
  }).join("");

  var ticketBox = document.getElementById("decision-ticket-list");
  if (ticketBox) {
    ticketBox.innerHTML = items.slice(0, 6).map(function (it) {
      return '<div class="experiment-card ' + escapeHtml(it.severity) + '">'
        + '<div class="experiment-head"><span>' + escapeHtml(it.source) + '</span>' + severityTag(it.severity) + '</div>'
        + '<div class="experiment-action">' + escapeHtml(it.next) + '</div>'
        + '<div class="experiment-meta">Owner: ' + escapeHtml(it.owner) + ' | Confidence: ' + escapeHtml(formatConfidence(it.confidence)) + ' | Evidence: ' + escapeHtml(it.evidence) + '</div>'
        + '</div>';
    }).join("");
  }

  var historyBody = document.getElementById("decision-history-body");
  if (historyBody) {
    historyBody.innerHTML = kfRunHistory.slice(0, 10).map(function (r) {
      return "<tr>"
        + "<td>" + escapeHtml(new Date(r.at).toLocaleString()) + "</td>"
        + "<td>" + escapeHtml(r.label || runKindLabel(r.kind)) + "</td>"
        + "<td>" + escapeHtml(r.summary || "--") + "</td>"
        + "<td>" + escapeHtml(r.key_metric || "--") + "</td>"
        + "</tr>";
    }).join("") || '<tr><td colspan="4">Run a panel to start local history.</td></tr>';
  }
}

function decisionMemoMarkdown() {
  var items = buildDecisionItems();
  var lines = [
    "# KineticsForge Decision Memo",
    "",
    "Generated: " + new Date().toISOString(),
    "Claim level: simulation-backed unless an uploaded measured dataset is referenced.",
    "",
    "## Action Queue"
  ];
  items.forEach(function (it) {
    lines.push("- [" + it.severity.toUpperCase() + "] " + it.source + " | " + it.action + " | Evidence: " + it.evidence + " | Owner: " + it.owner);
  });
  lines.push("", "## Experiment Tickets");
  items.forEach(function (it, idx) {
    lines.push((idx + 1) + ". " + it.next);
  });
  lines.push("", "## Recent Runs");
  (kfRunHistory.slice(0, 10)).forEach(function (r) {
    lines.push("- " + r.at + " | " + (r.label || runKindLabel(r.kind)) + " | " + (r.summary || "--") + " | " + (r.key_metric || "--"));
  });
  return lines.join("\n");
}

function exportDecisionMemoMarkdown() {
  downloadText(decisionMemoMarkdown(), "kineticsforge_decision_memo.md", "text/markdown;charset=utf-8");
}

function exportDecisionMemoJSON() {
  downloadJSON({
    format: "kineticsforge_decision_memo_v1",
    generated_at: new Date().toISOString(),
    claim_level: "simulation-backed unless current upload data is referenced",
    actions: buildDecisionItems(),
    run_history: kfRunHistory.slice(0, 20)
  }, "kineticsforge_decision_memo.json");
}

function exportExperimentTicketsCSV() {
  var rows = buildDecisionItems().map(function (it, idx) {
    return {
      ticket_id: "KF-" + String(idx + 1).padStart(3, "0"),
      severity: it.severity,
      source: it.source,
      owner: it.owner,
      evidence: it.evidence,
      action: it.action,
      next_experiment: it.next,
      confidence: it.confidence == null ? "" : it.confidence
    };
  });
  downloadCSV(rows, "kineticsforge_experiment_tickets.csv");
}

function clearRunHistory() {
  kfRunHistory = [];
  window.__kfRunHistory = kfRunHistory;
  saveRunHistory();
  renderDecisionConsole();
  showToast("Local run history cleared.", "ok");
}

// Diffusion-limited SEI: single coefficient replaces old dual-term (constant + sqrt) model.
var SEI_GROWTH_COEFF = 0.048;
var SEI_REF_EA = 0.56;
var SEI_REF_TEMP = 318.15;
var GLOBAL_DEGRADATION_SCALE = 0.052;
var JT_LOSS_COEFF = 6.5e-3;
var DESOLV_LOSS_COEFF = 2.5e-4;
var BV_RATE_LOSS_COEFF = 1.2e-4;
var RESIDUAL_LOSS_COEFF = 1.0e-5;
var RECYCLING_MC_SAMPLES = 200;

// Navigation
function navigate(p) {
  document.querySelectorAll(".section").forEach(function (s) { s.classList.remove("active"); });
  document.querySelectorAll(".nav-links a").forEach(function (a) { a.classList.remove("active"); });
  var s = document.getElementById("sec-" + p);
  if (s) {
    s.classList.add("active");
    s.style.animation = "none";
    s.offsetHeight;
    s.style.animation = "";
  }
  var l = document.getElementById("nav-" + p);
  if (l) l.classList.add("active");
  if (p === "decisions") renderDecisionConsole();
  window.scrollTo(0, 0);
}

function animC(el, t, d) {
  if (!el) return;
  d = d || 1200;
  var s = performance.now();
  var format = t > 9999 ? function (v) { return (v / 1e6).toFixed(1) + "M"; } : function (v) { return Math.round(v).toString(); };
  requestAnimationFrame(function step(n) {
    var p = Math.min((n - s) / d, 1);
    el.textContent = format(t * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(step);
  });
}

// Canvas primitives
function makeCanvas(id) {
  var c = document.getElementById(id);
  if (!c) return null;
  var cv = c.querySelector("canvas");
  if (!cv) {
    cv = document.createElement("canvas");
    c.appendChild(cv);
  }
  var r = c.getBoundingClientRect();
  var w = Math.max(220, Math.floor(r.width - 32));
  var h = Math.max(120, Math.floor(r.height - 32));
  cv.width = w;
  cv.height = h;
  return cv;
}

function drawGrid(ctx, p, pw, ph) {
  ctx.strokeStyle = "rgba(255,26,26,0.07)";
  ctx.lineWidth = 1;
  for (var i = 0; i <= 4; i++) {
    var y = p.t + ph * i / 4;
    ctx.beginPath();
    ctx.moveTo(p.l, y);
    ctx.lineTo(p.l + pw, y);
    ctx.stroke();
  }
}

function drawMultiLine(cv, series, opts) {
  if (!cv || !series.length) return;
  opts = opts || {};
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  var p = { t: 28, r: 18, b: 34, l: 52 };
  var pw = W - p.l - p.r, ph = H - p.t - p.b;
  var vals = [];
  series.forEach(function (s) { vals = vals.concat(s.values.filter(Number.isFinite)); });
  if (!vals.length) return;
  var yMn = opts.yMin != null ? opts.yMin : Math.min.apply(null, vals);
  var yMx = opts.yMax != null ? opts.yMax : Math.max.apply(null, vals);
  if (Math.abs(yMx - yMn) < 1e-9) { yMx += 0.1; yMn -= 0.1; }
  ctx.clearRect(0, 0, W, H);
  drawGrid(ctx, p, pw, ph);
  if (opts.title) {
    ctx.fillStyle = "#777";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillText(opts.title, p.l + 6, 15);
  }
  if (opts.band && opts.band.lo && opts.band.hi && opts.band.lo.length === opts.band.hi.length && opts.band.lo.length > 1) {
    ctx.beginPath();
    opts.band.hi.forEach(function (v, i) {
      var x = p.l + i / (opts.band.hi.length - 1) * pw;
      var y = p.t + ph - (v - yMn) / (yMx - yMn) * ph;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    for (var bi = opts.band.lo.length - 1; bi >= 0; bi--) {
      var bv = opts.band.lo[bi];
      var bx = p.l + bi / (opts.band.lo.length - 1) * pw;
      var by = p.t + ph - (bv - yMn) / (yMx - yMn) * ph;
      ctx.lineTo(bx, by);
    }
    ctx.closePath();
    ctx.fillStyle = opts.band.color || "rgba(255,26,26,0.12)";
    ctx.fill();
  }
  series.forEach(function (s) {
    var values = s.values;
    if (values.length < 2) return;
    ctx.beginPath();
    for (var i = 0; i < values.length; i++) {
      var x = p.l + i / (values.length - 1) * pw;
      var y = p.t + ph - (values[i] - yMn) / (yMx - yMn) * ph;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.strokeStyle = s.color || "#ff1a1a";
    ctx.lineWidth = s.width || 2;
    ctx.shadowColor = s.color || "#ff1a1a";
    ctx.shadowBlur = s.glow ? 8 : 0;
    ctx.stroke();
    ctx.shadowBlur = 0;
  });
  ctx.fillStyle = "#5f5f5f";
  ctx.font = "10px JetBrains Mono, monospace";
  for (var j = 0; j <= 4; j++) {
    var v = yMn + (yMx - yMn) * (1 - j / 4);
    ctx.fillText(fmt(v, opts.yDigits == null ? 2 : opts.yDigits), 2, p.t + ph * j / 4 + 4);
  }
  if (opts.xMax != null) {
    ctx.fillText("0", p.l, H - 6);
    ctx.fillText(String(opts.xMax), p.l + pw - 32, H - 6);
  }
  if (opts.legend) {
    var lx = W - 150, ly = 14;
    series.forEach(function (s, i) {
      ctx.fillStyle = s.color || "#ff1a1a";
      ctx.fillRect(lx, ly + i * 13 - 7, 14, 2);
      ctx.fillStyle = "#777";
      ctx.fillText(s.name || "", lx + 20, ly + i * 13 - 3);
    });
  }
  if (opts.points && opts.points.length) {
    ctx.fillStyle = opts.pointColor || "#ffffff";
    opts.points.forEach(function (pt) {
      if (!Number.isFinite(pt.x) || !Number.isFinite(pt.y)) return;
      var x = p.l + clamp(pt.x / Math.max(1, opts.xMax || pt.x), 0, 1) * pw;
      var y = p.t + ph - (pt.y - yMn) / (yMx - yMn) * ph;
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

function drawLine(cv, vals, opts) {
  drawMultiLine(cv, [{ name: opts && opts.name, values: vals, color: opts && opts.color || "#ff1a1a", glow: true }], opts);
}

function drawScatter(cv, pts, opts) {
  if (!cv || !pts.length) return;
  opts = opts || {};
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  var p = { t: 28, r: 18, b: 34, l: 52 };
  var pw = W - p.l - p.r, ph = H - p.t - p.b;
  var xs = pts.map(function (p) { return p.x; });
  var ys = pts.map(function (p) { return p.y; });
  var xMn = Math.min.apply(null, xs) - 6, xMx = Math.max.apply(null, xs) + 6;
  var yMn = Math.max(0, Math.min.apply(null, ys) - 0.05), yMx = Math.min(1.15, Math.max.apply(null, ys) + 0.05);
  ctx.clearRect(0, 0, W, H);
  drawGrid(ctx, p, pw, ph);
  if (opts.title) {
    ctx.fillStyle = "#777";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillText(opts.title, p.l + 6, 15);
  }
  pts.forEach(function (pt) {
    var x = p.l + (pt.x - xMn) / (xMx - xMn) * pw;
    var y = p.t + ph - (pt.y - yMn) / (yMx - yMn) * ph;
    var r = pt.selected ? 7 : pt.front ? 4.5 : 3.2;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = pt.selected ? "rgba(255,26,26,1)" : pt.front ? "rgba(255,92,92,0.86)" : "rgba(255,120,120,0.35)";
    ctx.shadowColor = "#ff1a1a";
    ctx.shadowBlur = pt.selected ? 12 : 0;
    ctx.fill();
    ctx.shadowBlur = 0;
  });
  ctx.fillStyle = "#606060";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillText("Capacity (mAh/g)", p.l + pw / 2 - 42, H - 6);
  ctx.fillText("Stability", 2, p.t + ph / 2);
}

// Na-ion degradation physics mirror from core/phase_transition.py.
function degradationKnobs() {
  return {
    seiScale: num("diag-k-sei", 1.0),
    seiEa: num("diag-ea-sei", 0.56),
    p2Rate: num("diag-p2-k", 0.0028),
    p2Soc: num("diag-p2-soc", 0.78),
    jtScale: num("diag-jt-scale", 1.0),
    bvScale: num("diag-bv-scale", 1.0),
    stressExp: num("diag-stress-exp", 0.55),
    residualScale: num("diag-residual-scale", 1.0)
  };
}
function diagnosticComposition() {
  return {
    Na: clamp(num("diag-na", 1.02), 0.60, 1.20),
    Mn: clamp(num("diag-mn", 0.52), 0.05, 0.95),
    Fe: clamp(num("diag-fe", 0.43), 0.05, 0.95),
    dopant_frac: clamp(num("diag-dop", 0.05), 0, 0.25)
  };
}
function setDiagnosticComposition(comp) {
  [["diag-na", comp.Na], ["diag-mn", comp.Mn], ["diag-fe", comp.Fe], ["diag-dop", comp.dopant_frac == null ? 0.05 : comp.dopant_frac]].forEach(function (item) {
    var el = document.getElementById(item[0]);
    if (el && Number.isFinite(item[1])) el.value = Number(item[1]).toFixed(item[0] === "diag-dop" ? 3 : 2);
  });
  updateDiag();
}

function naIonTerms(state, comp, T, cfg) {
  cfg = cfg || degradationKnobs();
  var kB = 8.617e-5;
  var soc = clamp(state.soc, 0, 1);
  var mn = clamp(comp.Mn, 0, 1.5);
  var fe = clamp(comp.Fe, 0, 1.5);
  var dop = clamp(comp.dopant_frac || 0, 0, 0.25);
  var jt = clamp(cfg.jtScale * mn * clamp(1.15 - soc, 0, 1) * expClamp((T - 298.15) * 0.018, -4, 4) * Math.exp(-0.45 * fe - 0.70 * dop), 0, 4);
  var socCrit = clamp(cfg.p2Soc - 0.09 * mn + 0.06 * fe + 0.18 * dop, 0.55, 0.95);
  var p2Gate = sigmoid((soc - socCrit) / 0.045);
  var p2o2Rate = clamp(cfg.p2Rate * p2Gate * expClamp((T - 298.15) * 0.024 / 25, -3, 3) * (1 + 0.35 * jt), 0, 0.08);
  // Na+ desolvation barrier: 0.4-0.6 eV (Jian et al., Komaba et al.) — not 0.18 eV (SEI migration)
  var barrier = 0.50 + 0.025 * mn - 0.014 * fe - 0.050 * dop;
  var desolv = clamp(Math.exp(clamp(barrier / (kB * T + 1e-10), -2, 4)) * (1 + 0.25 * relu(soc - 0.85)), 0.2, 30);
  var beta = clamp(0.48 - 0.035 * Math.log1p(desolv) + 0.025 * clamp(soc - 0.5, -0.5, 0.5), 0.25, 0.75);
  // Unified SEI: Arrhenius factor normalized to reference T=318.15K
  var seiRef = Math.exp(-cfg.seiEa / (kB * SEI_REF_TEMP));
  var seiArrhenius = cfg.seiScale * Math.exp(-cfg.seiEa / (kB * T)) / Math.max(seiRef, 1e-30);
  return { jt: jt, p2o2Rate: p2o2Rate, desolv: desolv, beta: beta, seiRate: seiArrhenius, socCrit: socCrit };
}

function simulateDegradation(options) {
  options = options || {};
  var T = (options.temperatureC != null ? options.temperatureC : parseFloat(document.getElementById("temp-slider").value)) + 273.15;
  var cRate = options.cRate != null ? options.cRate : parseFloat(document.getElementById("crate-slider").value);
  var nCycles = options.cycles != null ? options.cycles : parseInt(document.getElementById("cycles-slider").value, 10);
  var enableP2 = document.getElementById("sw-p2o2").checked;
  var enableJt = document.getElementById("sw-jt").checked;
  var enableSei = document.getElementById("sw-sei").checked;
  var enableNeural = document.getElementById("sw-neural").checked;
  var cfg = Object.assign({}, degradationKnobs(), options.cfg || {});
  var comp = options.comp || diagnosticComposition();
  var Q = 1.0, V = 3.34;
  var cap = [Q], voltage = [V], p2Cum = [0], jtCum = [0], seiCum = [0], rateCum = [0], resCum = [0];
  var socSeries = [0.78], p2RateSeries = [0], jtSeries = [0], desolvSeries = [0], dominant = ["SEI"];
  var p2 = 0, jt = 0, sei = 0, rate = 0, res = 0, knee = -1;
  var stress = 0.6 + Math.pow(cRate, cfg.stressExp);
  for (var i = 1; i <= nCycles; i++) {
    var sohWindow = clamp(Q, 0.50, 1.0);
    var socBase = 0.78 + 0.04 * Math.min(1, cRate / 2.4);
    var usableSoc = 0.62 + 0.38 * sohWindow;
    var soc = clamp(0.55 + (socBase - 0.55) * usableSoc + 0.022 * Math.sin(i * 0.17) * usableSoc, 0.55, 0.98);
    var terms = naIonTerms({ Q: Q, V: V, soc: soc }, comp, T, cfg);
    var scale = GLOBAL_DEGRADATION_SCALE * stress;
    var sqrtIncrement = Math.sqrt(i) - Math.sqrt(i - 1);
    var seiLoss = enableSei ? Q * terms.seiRate * SEI_GROWTH_COEFF * sqrtIncrement * scale : 0;
    var p2Loss = enableP2 ? Q * 0.65 * terms.p2o2Rate * scale : 0;
    var jtLoss = enableJt ? Q * JT_LOSS_COEFF * terms.jt * scale : 0;
    var desolvLoss = Q * DESOLV_LOSS_COEFF * Math.log1p(terms.desolv) * scale;
    var exchangeProxy = clamp(0.34 + 0.18 * comp.Fe - 0.08 * Math.log1p(terms.desolv) + 0.04 * (1 - terms.beta), 0.08, 0.9);
    var eta = Math.asinh(cRate / (2 * exchangeProxy));
    var rateStress = 1 + 0.20 * Math.pow(Math.max(0, cRate - 1.5), 2);
    var rateLoss = Q * cfg.bvScale * BV_RATE_LOSS_COEFF * eta * eta * rateStress * scale;
    var residualLoss = enableNeural ? Q * cfg.residualScale * RESIDUAL_LOSS_COEFF * sigmoid((i / nCycles - 0.62) / 0.16) * (0.8 + 0.35 * cRate) : 0;
    var dQ = seiLoss + p2Loss + jtLoss + desolvLoss + rateLoss + residualLoss;
    Q = clamp(Q - dQ, 0.25, 1.02);
    p2 += p2Loss; jt += jtLoss; sei += seiLoss + desolvLoss; rate += rateLoss; res += residualLoss;
    var vDegradation = p2 * 0.15 + jt * 0.08 + sei * 0.05 + rate * 0.04;
    V = clamp(3.34 - vDegradation, 2.4, 3.5);
    cap.push(Q); voltage.push(V); p2Cum.push(p2 * 100); jtCum.push(jt * 100); seiCum.push(sei * 100); rateCum.push(rate * 100); resCum.push(res * 100);
    socSeries.push(soc); p2RateSeries.push(terms.p2o2Rate); jtSeries.push(terms.jt); desolvSeries.push(Math.log1p(terms.desolv));
    var termsNow = [
      ["P2-O2", p2Loss],
      ["JT", jtLoss],
      ["SEI", seiLoss + desolvLoss],
      ["Rate", rateLoss],
      ["Residual", residualLoss]
    ].sort(function (a, b) { return b[1] - a[1]; });
    dominant.push(termsNow[0][0]);
    if (knee < 0 && i > 10) {
      var d2 = cap[i] - 2 * cap[i - 1] + cap[i - 2];
      if (d2 < -1.6e-5) knee = i;
    }
  }
  return { cap: cap, voltage: voltage, p2: p2Cum, jt: jtCum, sei: seiCum, rate: rateCum, residual: resCum, knee: knee, nCycles: nCycles, soc: socSeries, p2Rate: p2RateSeries, jtRaw: jtSeries, desolv: desolvSeries, dominant: dominant, comp: comp, cfg: cfg };
}

function updateDiag() {
  document.getElementById("temp-val").textContent = document.getElementById("temp-slider").value + " C";
  document.getElementById("crate-val").textContent = parseFloat(document.getElementById("crate-slider").value).toFixed(1) + "C";
  document.getElementById("cycles-val").textContent = document.getElementById("cycles-slider").value;
  ["diag-na", "diag-mn", "diag-fe", "diag-dop"].forEach(function (id) {
    var el = document.getElementById(id);
    var val = document.getElementById(id + "-val");
    if (el && val) val.textContent = parseFloat(el.value).toFixed(id === "diag-dop" ? 3 : 2);
  });
}

function runDiagnostics() {
  var out = simulateDegradation();
  out.band = uncertaintyBand(out);
  var cv = makeCanvas("diag-chart");
  var vcv = makeCanvas("diag-voltage-chart");
  var mech = makeCanvas("diag-mech-chart");
  var surf = makeCanvas("diag-phase-chart");
  var frame = 0, total = 72, step = Math.max(2, Math.floor(out.cap.length / total));
  function anim() {
    frame++;
    var n = Math.min(frame * step, out.cap.length);
    drawMultiLine(cv, [{ name: "capacity", values: out.cap.slice(0, n), color: "#ff1a1a", glow: true }], { yMin: 0.65, yMax: 1.02, xMax: out.nCycles, title: "Capacity fade with SEI sensitivity range", color: "#ff1a1a", yDigits: 3, band: n > 4 ? { lo: out.band.lo.slice(0, n), hi: out.band.hi.slice(0, n), color: "rgba(255,26,26,0.12)" } : null });
    drawLine(vcv, out.voltage.slice(0, n), { yMin: 3.20, yMax: 3.36, xMax: out.nCycles, title: "Average discharge voltage from accumulated loss terms", color: "#ff9f1a", yDigits: 3 });
    drawMultiLine(mech, [
      { name: "SEI+desolv", values: out.sei.slice(0, n), color: "#ff9f1a" },
      { name: "P2-O2", values: out.p2.slice(0, n), color: "#ff1a1a", glow: true },
      { name: "JT", values: out.jt.slice(0, n), color: "#d946ef" },
      { name: "rate", values: out.rate.slice(0, n), color: "#22c55e" },
      { name: "residual", values: out.residual.slice(0, n), color: "#38bdf8" }
    ], { yMin: 0, xMax: out.nCycles, title: "Cumulative loss contribution (%)", legend: true, yDigits: 1 });
    if (n < out.cap.length) requestAnimationFrame(anim);
  }
  anim();
  drawDegradationSurface(surf, out);
  var eol = out.cap[out.cap.length - 1];
  var r80 = out.cap.findIndex(function (v) { return v < 0.8; });
  window.__kfDiag = out;
  document.getElementById("diag-eol").textContent = eol.toFixed(3);
  document.getElementById("diag-fade").textContent = ((1 - eol) * 100).toFixed(1) + "%";
  document.getElementById("diag-knee").textContent = out.knee > 0 ? out.knee : "N/A";
  document.getElementById("diag-rul").textContent = r80 > 0 ? r80 : ">" + out.nCycles;
  var totals = [
    ["P2-O2 phase transition", out.p2[out.p2.length - 1]],
    ["Jahn-Teller distortion", out.jt[out.jt.length - 1]],
    ["SEI + Na desolvation", out.sei[out.sei.length - 1]],
    ["Butler-Volmer rate stress", out.rate[out.rate.length - 1]],
    ["bounded residual", out.residual[out.residual.length - 1]]
  ].sort(function (a, b) { return b[1] - a[1]; });
  var recommendation = totals[0][0].indexOf("P2") >= 0
    ? "Lower upper cutoff voltage or add Al/Ti stabilization before spending lab cycles."
    : totals[0][0].indexOf("SEI") >= 0
      ? "Prioritize electrolyte/additive screening and lower-temperature cycling validation."
      : totals[0][0].indexOf("Jahn") >= 0
        ? "Reduce Mn3+ fraction or test Ti/Fe compensation."
        : totals[0][0].indexOf("Butler") >= 0
          ? "Reduce C-rate or improve interfacial kinetics before deeper cycling."
          : "Residual is large: calibrate the residual term against holdout cycles before making a claim.";
  setHtml("diag-decision", "<strong>Dominant mechanism:</strong> " + totals[0][0] + " (" + totals[0][1].toFixed(2) + "% loss contribution). <strong>Next experiment:</strong> " + recommendation);
  drawMechanismPie(out);
  setHtml("diag-map-note", "State map readout: dominant=" + totals[0][0] + ", EOL=" + eol.toFixed(3) + ", fade=" + ((1 - eol) * 100).toFixed(1) + "%, RUL80=" + (r80 > 0 ? r80 : ">" + out.nCycles) + ".");
  var cal = document.getElementById("diag-cal-result");
  if (cal && !cal.textContent.trim()) cal.textContent = "Paste cycle,capacity rows and run calibration to fit SEI/P2/JT/stress coefficients.";
  var diagConf = computeDiagConfidence(out);
  renderConfidence("diag-confidence", diagConf.confidence, diagConf.detail);
  recordRun("diagnostics", {
    summary: "Dominant " + totals[0][0] + ", fade " + ((1 - eol) * 100).toFixed(1) + "% over " + out.nCycles + " cycles",
    key_metric: "EOL " + eol.toFixed(3),
    confidence: diagConf.confidence
  });
}

function uncertaintyBand(base) {
  var lowCfg = Object.assign({}, base.cfg, {
    seiScale: base.cfg.seiScale * 0.85,
    p2Rate: base.cfg.p2Rate * 0.85,
    jtScale: base.cfg.jtScale * 0.85,
    residualScale: base.cfg.residualScale * 0.85
  });
  var highCfg = Object.assign({}, base.cfg, {
    seiScale: base.cfg.seiScale * 1.15,
    p2Rate: base.cfg.p2Rate * 1.15,
    jtScale: base.cfg.jtScale * 1.15,
    residualScale: base.cfg.residualScale * 1.15
  });
  var common = {
    temperatureC: parseFloat(document.getElementById("temp-slider").value),
    cRate: parseFloat(document.getElementById("crate-slider").value),
    cycles: base.nCycles,
    comp: base.comp
  };
  var low = simulateDegradation(Object.assign({}, common, { cfg: lowCfg })).cap;
  var high = simulateDegradation(Object.assign({}, common, { cfg: highCfg })).cap;
  return {
    lo: low.map(function (v, i) { return Math.min(v, high[i]); }),
    hi: low.map(function (v, i) { return Math.max(v, high[i]); })
  };
}

function parseCalibrationData() {
  var el = document.getElementById("diag-cal-data");
  if (!el) return [];
  return el.value.split(/\n+/).map(function (line) {
    var parts = line.trim().split(/[,\s]+/).map(parseFloat);
    if (parts.length < 2 || !Number.isFinite(parts[0]) || !Number.isFinite(parts[1])) return null;
    var capacity = parts[1] > 1.5 ? parts[1] / 100 : parts[1];
    return { cycle: Math.max(0, Math.round(parts[0])), capacity: clamp(capacity, 0.25, 1.05) };
  }).filter(Boolean).sort(function (a, b) { return a.cycle - b.cycle; });
}

function calibrationError(curve, data) {
  var sse = 0;
  data.forEach(function (pt) {
    var idx = clamp(pt.cycle, 0, curve.length - 1);
    var err = curve[idx] - pt.capacity;
    sse += err * err;
  });
  return sse / Math.max(1, data.length);
}

function calibrateDiagnostics() {
  var data = parseCalibrationData();
  var result = document.getElementById("diag-cal-result");
  if (data.length < 3) {
    if (result) result.textContent = "Need at least 3 rows: cycle, capacity_fraction_or_percent.";
    return;
  }
  var baseCfg = degradationKnobs();
  var maxCycle = Math.max(parseInt(document.getElementById("cycles-slider").value, 10), data[data.length - 1].cycle);
  var common = {
    temperatureC: parseFloat(document.getElementById("temp-slider").value),
    cRate: parseFloat(document.getElementById("crate-slider").value),
    cycles: maxCycle,
    comp: diagnosticComposition()
  };
  var before = simulateDegradation(Object.assign({}, common, { cfg: baseCfg }));
  var best = { cfg: baseCfg, err: calibrationError(before.cap, data), curve: before.cap };
  [0.55, 0.75, 1.0, 1.30, 1.65].forEach(function (seiM) {
    [0.55, 0.80, 1.0, 1.30, 1.70].forEach(function (p2M) {
      [0.55, 0.85, 1.0, 1.35, 1.80].forEach(function (jtM) {
        [0.92, 1.0, 1.08].forEach(function (stressM) {
          var cfg = Object.assign({}, baseCfg, {
            seiScale: clamp(baseCfg.seiScale * seiM, 0.05, 10),
            p2Rate: clamp(baseCfg.p2Rate * p2M, 0.0002, 0.025),
            jtScale: clamp(baseCfg.jtScale * jtM, 0.05, 4),
            stressExp: clamp(baseCfg.stressExp * stressM, 0.25, 3)
          });
          var out = simulateDegradation(Object.assign({}, common, { cfg: cfg }));
          var err = calibrationError(out.cap, data);
          if (err < best.err) best = { cfg: cfg, err: err, curve: out.cap };
        });
      });
    });
  });
  document.getElementById("diag-k-sei").value = best.cfg.seiScale.toFixed(3);
  document.getElementById("diag-p2-k").value = best.cfg.p2Rate.toFixed(5);
  document.getElementById("diag-jt-scale").value = best.cfg.jtScale.toFixed(3);
  document.getElementById("diag-stress-exp").value = best.cfg.stressExp.toFixed(3);
  var mean = data.reduce(function (a, b) { return a + b.capacity; }, 0) / data.length;
  var sst = data.reduce(function (a, b) { return a + Math.pow(b.capacity - mean, 2); }, 0);
  var r2 = sst > 1e-12 ? 1 - best.err * data.length / sst : 1;
  var rmse = Math.sqrt(best.err);
  drawMultiLine(makeCanvas("diag-cal-chart"), [
    { name: "before", values: before.cap, color: "#777" },
    { name: "calibrated", values: best.curve, color: "#ff1a1a", glow: true }
  ], { yMin: 0.65, yMax: 1.02, xMax: maxCycle, title: "Calibration fit: measured points vs simulated curves", legend: true, yDigits: 3, points: data.map(function (d) { return { x: d.cycle, y: d.capacity }; }), pointColor: "#ffffff" });
  if (result) result.textContent = "Fitted SEI scale=" + best.cfg.seiScale.toFixed(2) + ", P2 rate=" + best.cfg.p2Rate.toFixed(4) + ", JT scale=" + best.cfg.jtScale.toFixed(2) + ". RMSE=" + rmse.toFixed(4) + ", R2=" + r2.toFixed(3) + ".";
  runDiagnostics();
}

function exportDiagnosticsCSV() {
  var out = window.__kfDiag || simulateDegradation();
  var rows = out.cap.map(function (cap, i) {
    return { cycle: i, capacity: cap, voltage: out.voltage[i], sei_pct: out.sei[i], p2_pct: out.p2[i], jt_pct: out.jt[i], rate_pct: out.rate[i], residual_pct: out.residual[i] };
  });
  downloadCSV(rows, "kineticsforge_diagnostics.csv");
}

function exportBmsCSV() {
  var sim = window.__kfBms || {};
  var rows = [];
  (sim.frames || []).forEach(function (frame) {
    frame.risks.forEach(function (risk, i) {
      rows.push({ time_s: frame.t, cell: "C" + i, risk: risk, temp_C: frame.temps[i], dTdt_K_s: frame.slopes[i], rct_ohm: frame.rcts[i], rsei_ohm: frame.rseis[i] });
    });
  });
  downloadCSV(rows, "kineticsforge_bms.csv");
}

function exportMaterialsCSV() {
  var mat = window.__kfMaterials || {};
  var comp = mat.comp || {};
  downloadCSV([{ Na: comp.Na, Mn: comp.Mn, Fe: comp.Fe, Al: !!comp.al, Ti: !!comp.ti, capacity_mAh_g: mat.Q0, voltage_V: mat.avgVoltage, stability: mat.stability, fade500: mat.fade500, score: mat.score, oxygen_risk: mat.oxygenRisk, charge_risk: mat.chargeRisk }], "kineticsforge_materials.csv");
}

function exportRecyclingCSV() {
  var rec = window.__kfRecycling || {};
  var rows = (rec.elements || []).map(function (el, i) {
    return { element: el.n, wt_fraction: el.wt, recovery: rec.targets ? rec.targets[i] : "", recovered_kg: rec.mass ? rec.mass * el.wt * rec.targets[i] : "" };
  });
  rows.push({ element: "TOTAL", wt_fraction: "", recovery: "", recovered_kg: rec.totalRecovered || "" });
  downloadCSV(rows, "kineticsforge_recycling.csv");
}

// ── Mechanism Attribution Donut Chart ────────────────────────────────────
function drawMechanismPie(out) {
  var cv = document.getElementById("diag-pie-canvas");
  var legend = document.getElementById("diag-pie-legend");
  if (!cv || !legend) return;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  var cx = W / 2, cy = H / 2, R = Math.min(cx, cy) - 8, r = R * 0.52;
  var totalFade = (1 - out.cap[out.cap.length - 1]);
  if (totalFade < 1e-6) { legend.innerHTML = "No significant fade detected."; return; }
  var slices = [
    { name: "SEI + Desolvation", value: out.sei[out.sei.length - 1], color: "#ff9f1a" },
    { name: "P2-O2 Transition", value: out.p2[out.p2.length - 1], color: "#ff1a1a" },
    { name: "Jahn-Teller", value: out.jt[out.jt.length - 1], color: "#d946ef" },
    { name: "Rate Polarization", value: out.rate[out.rate.length - 1], color: "#22c55e" },
    { name: "Residual", value: out.residual[out.residual.length - 1], color: "#38bdf8" }
  ];
  var sum = slices.reduce(function (a, s) { return a + s.value; }, 0);
  if (sum < 1e-6) return;
  ctx.clearRect(0, 0, W, H);
  var start = -Math.PI / 2;
  slices.forEach(function (s) {
    var frac = s.value / sum;
    var sweep = frac * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, R, start, start + sweep);
    ctx.arc(cx, cy, r, start + sweep, start, true);
    ctx.closePath();
    ctx.fillStyle = s.color;
    ctx.globalAlpha = 0.88;
    ctx.fill();
    ctx.globalAlpha = 1.0;
    s.pct = (frac * 100).toFixed(1);
    start += sweep;
  });
  // Center label
  ctx.fillStyle = "#ddd";
  ctx.font = "bold 15px JetBrains Mono, monospace";
  ctx.textAlign = "center";
  ctx.fillText((totalFade * 100).toFixed(1) + "%", cx, cy - 3);
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillStyle = "#888";
  ctx.fillText("total fade", cx, cy + 13);
  ctx.textAlign = "start";
  // Legend
  legend.innerHTML = slices.map(function (s) {
    return '<div style="display:flex;align-items:center;gap:0.5rem">'
      + '<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:' + s.color + '"></span>'
      + '<span>' + s.name + ': <strong>' + s.pct + '%</strong></span></div>';
  }).join("");
}

// ── JSON Exports (all panels) ───────────────────────────────────────────
function exportDiagnosticsJSON() {
  var out = window.__kfDiag || simulateDegradation();
  var payload = {
    format: "kineticsforge_diagnostics_v1",
    generated_at: new Date().toISOString(),
    parameters: {
      temperature_C: parseFloat(document.getElementById("temp-slider").value),
      c_rate: parseFloat(document.getElementById("crate-slider").value),
      cycles: parseInt(document.getElementById("cycles-slider").value, 10),
      composition: out.comp,
      knobs: out.cfg
    },
    results: {
      eol_capacity: out.cap[out.cap.length - 1],
      total_fade_pct: ((1 - out.cap[out.cap.length - 1]) * 100),
      knee_cycle: out.knee > 0 ? out.knee : null,
      rul_80: out.cap.findIndex(function (v) { return v < 0.8; }),
      mechanism_pct: {
        sei_desolv: out.sei[out.sei.length - 1],
        p2o2: out.p2[out.p2.length - 1],
        jt: out.jt[out.jt.length - 1],
        rate: out.rate[out.rate.length - 1],
        residual: out.residual[out.residual.length - 1]
      }
    },
    curves: {
      capacity: out.cap,
      voltage: out.voltage
    }
  };
  downloadJSON(payload, "kineticsforge_diagnostics.json");
}

function exportBmsJSON() {
  var sim = window.__kfBms || {};
  var payload = {
    format: "kineticsforge_bms_v1",
    generated_at: new Date().toISOString(),
    parameters: {
      n_cells: sim.n,
      duration_s: sim.duration,
      seed: sim.seed,
      fault_cell: sim.faultCell
    },
    alerts: sim.alerts || [],
    max_risk: sim.maxRisk,
    frames: (sim.frames || []).map(function (f) {
      return { t: f.t, temps: f.temps, risks: f.risks, rcts: f.rcts, rseis: f.rseis, slopes: f.slopes };
    })
  };
  downloadJSON(payload, "kineticsforge_bms.json");
}

function exportMaterialsJSON() {
  var mat = window.__kfMaterials || {};
  var payload = {
    format: "kineticsforge_materials_v1",
    generated_at: new Date().toISOString(),
    composition: mat.comp,
    predicted: {
      capacity_mAh_g: mat.Q0,
      voltage_V: mat.avgVoltage,
      stability: mat.stability,
      fade_500: mat.fade500,
      score: mat.score,
      oxygen_risk: mat.oxygenRisk,
      charge_balance_risk: mat.chargeRisk,
      energy_density: mat.energyDensity,
      cost_usd_kwh: mat.costKwh
    },
    candidates: mat.candidates || []
  };
  downloadJSON(payload, "kineticsforge_materials.json");
}

function exportRecyclingJSON() {
  var rec = window.__kfRecycling || {};
  var payload = {
    format: "kineticsforge_recycling_v1",
    generated_at: new Date().toISOString(),
    parameters: {
      mass_kg: rec.mass,
      acid_molarity: rec.acid,
      temperature_C: rec.tempC,
      leach_time_min: rec.leachMin,
      particle_um: rec.particleUm
    },
    recoveries: {},
    total_recovered_kg: rec.totalRecovered,
    uncertainty_interval: rec.interval,
    purity_proxy: rec.purity,
    margin_proxy_inr: rec.margin,
    cost_estimate_inr: rec.cost
  };
  (rec.elements || []).forEach(function (el, i) {
    payload.recoveries[el.n] = {
      wt_fraction: el.wt,
      recovery_rate: rec.targets ? rec.targets[i] : null,
      recovered_kg: rec.mass && rec.targets ? rec.mass * el.wt * rec.targets[i] : null
    };
  });
  downloadJSON(payload, "kineticsforge_recycling.json");
}

// ── Formation Efficiency Scorer (BYOD uploads) ─────────────────────────
function renderFormationScore(data) {
  var container = document.getElementById("byod-formation");
  if (!container) return;
  var pred = (data.predictions || {});
  var m14 = ((pred.model_outputs || {}).M14_FormationProtocol || {});
  var m11 = ((pred.model_outputs || {}).M11_ElectrolyteHealth || {});
  var features = data.features || {};
  var ce = features.early_coulombic_efficiency;
  var seiQ = m14.sei_quality;
  var lifeIdx = m14.life_index;
  var robust = m14.robustness_index;
  var proto = m14.suggested_protocol || {};
  if (seiQ == null && ce == null) { container.innerHTML = ""; return; }
  var ceStr = ce != null ? (ce * 100).toFixed(1) + "%" : "--";
  var optimal = "97%+";
  var gap = ce != null ? Math.max(0, 0.97 - ce) * 100 : null;
  var gapStr = gap != null && gap > 0.1 ? gap.toFixed(1) + "% below optimal" : "near optimal";
  function gauge(val, label) {
    if (val == null) return "";
    var pct = Math.round(clamp(val, 0, 1) * 100);
    var color = pct >= 75 ? "#22c55e" : pct >= 50 ? "#ff9f1a" : "#ff1a1a";
    return '<div style="margin:0.5rem 0"><div style="font-size:0.72rem;color:#888;margin-bottom:3px">' + label + '</div>'
      + '<div style="background:#1a1a1a;border-radius:4px;height:10px;overflow:hidden"><div style="height:100%;border-radius:4px;width:' + pct + '%;background:' + color + ';transition:width 0.6s ease"></div></div>'
      + '<div style="font-size:0.7rem;color:#aaa;margin-top:2px">' + pct + '%</div></div>';
  }
  var html = '<div class="panel-title"><span class="indicator"></span> Formation Efficiency</div>';
  html += '<div style="display:flex;gap:1.5rem;flex-wrap:wrap;align-items:flex-start">';
  html += '<div style="flex:1;min-width:180px">';
  html += gauge(seiQ, "SEI Quality");
  html += gauge(lifeIdx, "Lifetime Index");
  html += gauge(robust, "Robustness");
  html += '</div>';
  html += '<div style="flex:1;min-width:200px;font-size:0.78rem;color:#bbb;line-height:1.7">';
  html += '<div>Formation CE: <strong style="color:#fff">' + ceStr + '</strong> (optimal: ' + optimal + ')</div>';
  html += '<div>Status: <strong style="color:' + (gap != null && gap > 3 ? "#ff9f1a" : "#22c55e") + '">' + gapStr + '</strong></div>';
  if (proto.formation_c_rate != null) {
    html += '<div style="margin-top:0.5rem;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:6px">';
    html += '<div style="font-size:0.68rem;color:#666;margin-bottom:3px">SUGGESTED PROTOCOL</div>';
    html += '<div>Formation C-rate: <strong>' + proto.formation_c_rate + 'C</strong></div>';
    html += '<div>Rest time: <strong>' + proto.rest_time_hours + 'h</strong></div>';
    html += '</div>';
  }
  html += '</div></div>';
  container.innerHTML = html;
}

function fmtFeatureValue(v) {
  if (v == null || !Number.isFinite(Number(v))) return "--";
  var n = Number(v);
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(2);
  if (Math.abs(n) >= 1) return n.toFixed(4);
  return n.toPrecision(4);
}

function drawDqdv(cv, pts, opts) {
  if (!cv || !pts || pts.length < 2) return;
  opts = opts || {};
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  var p = { t: 28, r: 18, b: 34, l: 52 };
  var pw = W - p.l - p.r, ph = H - p.t - p.b;
  var xs = pts.map(function (x) { return Number(x.voltage); }).filter(Number.isFinite);
  var ys = pts.map(function (x) { return Number(x.dqdv); }).filter(Number.isFinite);
  if (!xs.length || !ys.length) return;
  var xMn = Math.min.apply(null, xs), xMx = Math.max.apply(null, xs);
  var yMn = Math.min.apply(null, ys), yMx = Math.max.apply(null, ys);
  if (Math.abs(yMx - yMn) < 1e-9) { yMx += 1; yMn -= 1; }
  ctx.clearRect(0, 0, W, H);
  drawGrid(ctx, p, pw, ph);
  // dQ/dV line
  ctx.strokeStyle = "#ff1a1a";
  ctx.lineWidth = 2;
  ctx.shadowColor = "#ff1a1a";
  ctx.shadowBlur = 8;
  ctx.beginPath();
  pts.forEach(function (pt, i) {
    var x = p.l + (pt.voltage - xMn) / Math.max(1e-9, xMx - xMn) * pw;
    var y = p.t + ph - (pt.dqdv - yMn) / (yMx - yMn) * ph;
    if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;
  // d²Q/dV² overlay (second derivative fingerprint)
  if (pts.length > 6) {
    ctx.strokeStyle = "rgba(56,189,248,0.5)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    var d2 = [];
    for (var k = 1; k < pts.length - 1; k++) {
      var dv1 = pts[k].voltage - pts[k - 1].voltage;
      var dv2 = pts[k + 1].voltage - pts[k].voltage;
      if (Math.abs(dv1) < 1e-9 || Math.abs(dv2) < 1e-9) continue;
      d2.push({ voltage: pts[k].voltage, val: (pts[k + 1].dqdv - 2 * pts[k].dqdv + pts[k - 1].dqdv) / ((dv1 + dv2) / 2) });
    }
    if (d2.length > 2) {
      var d2Vals = d2.map(function (x) { return x.val; });
      var d2Mn = Math.min.apply(null, d2Vals), d2Mx = Math.max.apply(null, d2Vals);
      if (Math.abs(d2Mx - d2Mn) > 1e-9) {
        ctx.beginPath();
        d2.forEach(function (pt, i) {
          var x = p.l + (pt.voltage - xMn) / Math.max(1e-9, xMx - xMn) * pw;
          var y = p.t + ph - (pt.val - d2Mn) / (d2Mx - d2Mn) * ph;
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        });
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }
  // Peak annotations
  var peaks = opts.peaks || [];
  var peakLabels = [
    { range: [2.0, 2.8], label: "Na ordering" },
    { range: [2.8, 3.35], label: "Fe³⁺/²⁺" },
    { range: [3.35, 3.65], label: "Fe⁴⁺/³⁺" },
    { range: [3.65, 4.0], label: "Mn³⁺/⁴⁺" },
    { range: [4.0, 4.6], label: "O²⁻ redox" }
  ];
  peaks.forEach(function (peakV, idx) {
    var x = p.l + (peakV - xMn) / Math.max(1e-9, xMx - xMn) * pw;
    // Find y at peak
    var closestPt = pts.reduce(function (best, pt) {
      return Math.abs(pt.voltage - peakV) < Math.abs(best.voltage - peakV) ? pt : best;
    });
    var y = p.t + ph - (closestPt.dqdv - yMn) / (yMx - yMn) * ph;
    // Draw marker
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ff1a1a";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // Label
    var label = "Peak " + (idx + 1);
    peakLabels.forEach(function (pl) {
      if (peakV >= pl.range[0] && peakV < pl.range[1]) label = pl.label;
    });
    ctx.fillStyle = "#ddd";
    ctx.font = "bold 9px JetBrains Mono, monospace";
    var yOff = idx % 2 === 0 ? -14 : -26;
    ctx.fillText(label, x - 15, y + yOff);
    ctx.fillStyle = "#888";
    ctx.font = "8px JetBrains Mono, monospace";
    ctx.fillText(peakV.toFixed(2) + "V", x - 12, y + yOff + 10);
  });
  ctx.fillStyle = "#777";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillText("dQ/dV vs voltage  (dashed: d²Q/dV²)", p.l + 6, 15);
  ctx.fillText(xMn.toFixed(2) + "V", p.l, H - 6);
  ctx.fillText(xMx.toFixed(2) + "V", p.l + pw - 44, H - 6);
}

function renderBYOD(data) {
  window.__kfByod = data;
  window.__kfByodCompare = null;
  window.__kfByodBatch = null;
  var schemaBody = document.getElementById("byod-schema-body");
  var featureBody = document.getElementById("byod-feature-body");
  var warnings = document.getElementById("byod-warnings");
  var schema = data.schema || {};
  var mapping = schema.mapping || {};
  var confidence = schema.confidence || {};
  if (schemaBody) {
    var keys = Object.keys(mapping);
    schemaBody.innerHTML = keys.length ? keys.map(function (k) {
      return "<tr><td>" + escapeHtml(k) + "</td><td>" + escapeHtml(mapping[k]) + "</td><td>" + fmtFeatureValue(confidence[k]) + "</td></tr>";
    }).join("") : '<tr><td colspan="3">No confident mappings found.</td></tr>';
  }
  if (featureBody) {
    var names = data.feature_names || [];
    var feats = data.features || {};
    var avail = data.feature_availability || {};
    featureBody.innerHTML = names.map(function (name) {
      if (!avail[name]) return "";
      return "<tr><td>" + escapeHtml(name) + "</td><td>" + escapeHtml(fmtFeatureValue(feats[name])) + '</td><td><span class="tag ok">present</span></td></tr>';
    }).join("") || '<tr><td colspan="3">No tier-1 features were available.</td></tr>';
  }
  var pred = data.predictions || {};
  var present = (data.feature_mask || []).filter(function (x) { return !!x; }).length;
  setHtml("byod-soh", pred.soh != null ? (pred.soh * 100).toFixed(1) + "%" : "--");
  setHtml("byod-confidence", pred.confidence != null ? (pred.confidence * 100).toFixed(0) + "%" : "--");
  setHtml("byod-cycle80", pred.cycle_80_estimate != null ? String(pred.cycle_80_estimate) : "--");
  setHtml("byod-features", present + "/" + ((data.feature_names || []).length || 27));
  var dqdv = data.dqdv || {};
  drawDqdv(makeCanvas("byod-dqdv-chart"), dqdv.points || [], { peaks: (dqdv.peaks || []).map(Number) });
  setHtml("byod-dqdv-note", dqdv.peaks && dqdv.peaks.length ? "Detected peak voltages: " + dqdv.peaks.map(function (v) { return Number(v).toFixed(3) + "V"; }).join(", ") : "No reliable dQ/dV peaks detected. Upload discharge voltage-capacity traces for this panel.");
  var models = pred.model_outputs || {};
  var modelRows = Object.keys(models).map(function (id) {
    var m = models[id];
    var desc = Object.keys(m).filter(function (k) { return k !== "source" && k !== "checkpoint_source" && k !== "suggested_protocol" && k !== "research_preview"; }).map(function (k) {
      return k + "=" + (typeof m[k] === "number" ? fmtFeatureValue(m[k]) : m[k]);
    }).join(", ");
    var hasCheckpoint = m.checkpoint_source === "trained_forward";
    var tag = hasCheckpoint ? "ok" : (m.research_preview ? "warn" : "ok");
    return '<div class="switch-row"><div class="switch-label"><span class="name">' + escapeHtml(id) + '</span><span class="desc">' + escapeHtml(desc) + '</span></div><span class="tag ' + tag + '">' + escapeHtml(hasCheckpoint ? "trained forward" : (m.source || "derived")) + "</span></div>";
  }).join("");
  setHtml("byod-model-output", modelRows || '<div class="switch-row"><div class="switch-label"><span class="name">No outputs</span><span class="desc">Feature extraction did not produce enough inputs.</span></div><span class="tag warn">Low data</span></div>');
  var chem = models.M13_ChemIdentifier ? models.M13_ChemIdentifier.predicted_family : "unknown";
  var m11 = models.M11_ElectrolyteHealth || {};
  setHtml("byod-decision", "<strong>Upload result:</strong> SOH " + (pred.soh != null ? (pred.soh * 100).toFixed(1) + "%" : "--") + ", chemistry " + escapeHtml(chem) + ", sodium plating risk " + fmtFeatureValue(m11.checkpoint_sodium_plating_probability != null ? m11.checkpoint_sodium_plating_probability : m11.sodium_plating_probability) + ", inference " + escapeHtml(pred.inference_mode || "rules_only") + ". <strong>Use:</strong> export the arrays and validate low-confidence or research-preview outputs before decisions.");
  var uploadConf = computeUploadConfidence(data);
  renderConfidence("byod-confidence-detail", uploadConf.confidence, uploadConf.detail);
  if (warnings) {
    var list = data.warnings && data.warnings.length ? data.warnings : ["No warnings from parser."];
    warnings.innerHTML = list.map(function (w) { return '<div class="' + (w.indexOf("low") >= 0 || w.indexOf("not") >= 0 ? "warn" : "info") + '">' + escapeHtml(w) + "</div>"; }).join("");
  }
  renderFormationScore(data);
  recordRun("upload", {
    summary: (data.filename || "upload") + ", SOH " + (pred.soh != null ? (pred.soh * 100).toFixed(1) + "%" : "--") + ", confidence " + (pred.confidence != null ? (pred.confidence * 100).toFixed(0) + "%" : "--"),
    key_metric: present + " features",
    confidence: uploadConf.confidence
  });
}

function analyzeBYOD() {
  var input = document.getElementById("byod-file");
  if (!input || !input.files || !input.files.length) {
    showToast("Choose a cycler CSV/TXT/XLSX file first.", "warn");
    return;
  }
  var file = input.files[0];
  var status = document.getElementById("byod-status");
  if (status) status.textContent = "Uploading and extracting features from " + file.name + "...";
  var form = new FormData();
  form.append("file", file);
  fetch("/api/byod/analyze", { method: "POST", body: form })
    .then(function (res) {
      if (!res.ok) {
        return res.json().catch(function () { return {}; }).then(function (body) {
          throw new Error(body.detail || ("Upload analysis failed with HTTP " + res.status));
        });
      }
      return res.json();
    })
    .then(function (data) {
      renderBYOD(data);
      if (status) status.textContent = "Analyzed " + data.rows_read + " rows from " + data.filename + ". Session " + data.session_id + " expires automatically.";
      showToast("Upload analyzed. Features and dQ/dV are ready.", "ok");
    })
    .catch(function (err) {
      if (status) status.textContent = err.message;
      showToast(err.message, "error");
      console.warn(err);
    });
}

function compareBYOD() {
  var a = document.getElementById("byod-file");
  var b = document.getElementById("byod-file-b");
  if (!a || !b || !a.files.length || !b.files.length) {
    showToast("Choose two cycler files for A/B comparison.", "warn");
    return;
  }
  var status = document.getElementById("byod-status");
  if (status) status.textContent = "Comparing " + a.files[0].name + " against " + b.files[0].name + "...";
  var form = new FormData();
  form.append("file_a", a.files[0]);
  form.append("file_b", b.files[0]);
  fetch("/api/byod/compare", { method: "POST", body: form })
    .then(function (res) {
      if (!res.ok) {
        return res.json().catch(function () { return {}; }).then(function (body) {
          throw new Error(body.detail || ("Compare failed with HTTP " + res.status));
        });
      }
      return res.json();
    })
    .then(function (data) {
      window.__kfByodCompare = data;
      window.__kfByod = null;
      window.__kfByodBatch = null;
      setHtml("byod-soh", "--");
      setHtml("byod-confidence", "--");
      setHtml("byod-cycle80", "--");
      setHtml("byod-features", "--");
      var schemaBody = document.getElementById("byod-schema-body");
      var featureBody = document.getElementById("byod-feature-body");
      if (schemaBody) schemaBody.innerHTML = '<tr><td colspan="3">Schema table is available for single-file analysis.</td></tr>';
      if (featureBody) featureBody.innerHTML = '<tr><td colspan="3">Feature table is available for single-file analysis.</td></tr>';
      setHtml("byod-model-output", '<div class="switch-row"><div class="switch-label"><span class="name">Compare mode</span><span class="desc">Model card details are shown for single-file analysis only.</span></div><span class="tag">A/B</span></div>');
      setHtml("byod-dqdv-note", "dQ/dV fingerprint is displayed for single-file analysis.");
      renderConfidence("byod-confidence-detail", null, "");
      var c = data.comparison || {};
      var delta = c.delta || {};
      setHtml("byod-decision", "<strong>A/B result:</strong> " + escapeHtml(c.decision || "no decision") + ". SOH delta " + fmtFeatureValue(delta.soh) + ", fade delta " + fmtFeatureValue(delta.fade_fraction_per_cycle) + ". <strong>Use:</strong> compare only matched protocols.");
      recordRun("upload_compare", {
        summary: c.decision || "Compared matched uploads",
        key_metric: "SOH delta " + fmtFeatureValue(delta.soh),
        confidence: null
      });
      if (status) status.textContent = "Compared two uploads. Sessions: " + (data.file_a_session_id || "--") + " / " + (data.file_b_session_id || "--");
      showToast("A/B comparison ready.", "ok");
    })
    .catch(function (err) {
      if (status) status.textContent = err.message;
      showToast(err.message, "error");
      console.warn(err);
    });
}

function analyzeBYODBatch() {
  var input = document.getElementById("byod-batch-file");
  if (!input || !input.files || !input.files.length) {
    showToast("Choose a ZIP of cycler files first.", "warn");
    return;
  }
  var status = document.getElementById("byod-status");
  if (status) status.textContent = "Analyzing batch ZIP " + input.files[0].name + "...";
  var form = new FormData();
  form.append("file", input.files[0]);
  fetch("/api/byod/batch", { method: "POST", body: form })
    .then(function (res) {
      if (!res.ok) {
        return res.json().catch(function () { return {}; }).then(function (body) {
          throw new Error(body.detail || ("Batch analysis failed with HTTP " + res.status));
        });
      }
      return res.json();
    })
    .then(function (data) {
      window.__kfByodBatch = data;
      window.__kfByod = null;
      window.__kfByodCompare = null;
      setHtml("byod-soh", "--");
      setHtml("byod-confidence", "--");
      setHtml("byod-cycle80", "--");
      setHtml("byod-features", "--");
      var schemaBody = document.getElementById("byod-schema-body");
      var featureBody = document.getElementById("byod-feature-body");
      if (schemaBody) schemaBody.innerHTML = '<tr><td colspan="3">Schema table is available for single-file analysis.</td></tr>';
      if (featureBody) featureBody.innerHTML = '<tr><td colspan="3">Feature table is available for single-file analysis.</td></tr>';
      setHtml("byod-model-output", '<div class="switch-row"><div class="switch-label"><span class="name">Batch mode</span><span class="desc">Per-file model cards are available inside exported batch summaries.</span></div><span class="tag">ZIP</span></div>');
      setHtml("byod-dqdv-note", "dQ/dV fingerprint is displayed for single-file analysis.");
      renderConfidence("byod-confidence-detail", null, "");
      var stats = data.stats || {};
      setHtml("byod-decision", "<strong>Batch result:</strong> " + escapeHtml(data.decision || "batch analyzed") + ". Cells " + escapeHtml(String(data.files_analyzed || 0)) + ", SOH mean " + fmtFeatureValue(stats.soh_mean) + ", SOH std " + fmtFeatureValue(stats.soh_std) + ", outliers " + escapeHtml(String((data.outliers || []).length)) + ".");
      recordRun("upload_batch", {
        summary: (data.files_analyzed || 0) + " files, outliers " + ((data.outliers || []).length),
        key_metric: "SOH mean " + fmtFeatureValue(stats.soh_mean),
        confidence: null
      });
      var warnings = document.getElementById("byod-warnings");
      if (warnings) {
        warnings.innerHTML = (data.outliers || []).slice(0, 12).map(function (o) {
          return '<div class="warn">' + escapeHtml(o.filename || "cell") + ": " + escapeHtml((o.reasons || []).join(", ")) + "</div>";
        }).join("") || '<div class="info">No batch outliers flagged.</div>';
      }
      if (status) status.textContent = "Batch analyzed. Session " + (data.batch_session_id || "--") + " expires automatically.";
      showToast("Batch report ready.", "ok");
    })
    .catch(function (err) {
      if (status) status.textContent = err.message;
      showToast(err.message, "error");
      console.warn(err);
    });
}

function exportBYODCSV() {
  var data = window.__kfByod;
  if (window.__kfByodBatch && !data) {
    downloadCSV((window.__kfByodBatch.summaries || []).map(function (r) {
      return { kind: "batch_cell", session_id: r.session_id, filename: r.filename, soh: r.soh, confidence: r.confidence, cycle_80_estimate: r.cycle_80_estimate, fade_fraction_per_cycle: r.fade_fraction_per_cycle, anomaly_score: r.anomaly_score };
    }), "kineticsforge_byod_batch.csv");
    return;
  }
  if (window.__kfByodCompare && !data) {
    var comp = window.__kfByodCompare.comparison || {};
    downloadCSV((comp.top_feature_deltas || []).map(function (r) {
      return { kind: "compare_delta", feature: r.feature, a: r.a, b: r.b, delta: r.delta, relative_delta: r.relative_delta };
    }), "kineticsforge_byod_compare.csv");
    return;
  }
  if (!data) {
    showToast("Run an upload analysis before exporting.", "warn");
    return;
  }
  var rows = [];
  (data.cycle_summary || []).forEach(function (r) {
    rows.push({ kind: "cycle", name: r.cycle, discharge_capacity_ah: r.discharge_capacity_ah, charge_capacity_ah: r.charge_capacity_ah, ce: r.ce });
  });
  Object.keys(data.features || {}).forEach(function (k) {
    rows.push({ kind: "feature", name: k, discharge_capacity_ah: data.features[k], charge_capacity_ah: "", ce: "" });
  });
  downloadCSV(rows, "kineticsforge_byod_analysis.csv");
}

function exportBYODJSON() {
  var batch = window.__kfByodBatch;
  var compare = window.__kfByodCompare;
  var data = window.__kfByod;
  var sid = data && data.session_id ? data.session_id : (batch && batch.batch_session_id ? batch.batch_session_id : null);
  if (sid) {
    fetch("/api/byod/session/" + encodeURIComponent(sid) + "/export-json")
      .then(function (res) {
        if (!res.ok) throw new Error("JSON export failed with HTTP " + res.status);
        return res.json();
      })
      .then(function (payload) {
        downloadJSON(payload, batch ? "kineticsforge_byod_batch.json" : "kineticsforge_byod_session.json");
      })
      .catch(function (err) {
        showToast(err.message, "error");
        console.warn(err);
      });
    return;
  }
  if (compare) {
    downloadJSON(compare, "kineticsforge_byod_compare.json");
    return;
  }
  showToast("Run an upload analysis before exporting.", "warn");
}

function testMaterialInDiagnostics() {
  var comp = {
    Na: parseFloat(document.getElementById("na-slider").value),
    Mn: parseFloat(document.getElementById("mn-slider").value),
    Fe: parseFloat(document.getElementById("fe-slider").value),
    dopant_frac: (document.getElementById("sw-al").checked ? 0.04 : 0) + (document.getElementById("sw-ti").checked ? 0.03 : 0)
  };
  setDiagnosticComposition(comp);
  navigate("diagnostics");
  runDiagnostics();
}

function drawDegradationSurface(cv, out) {
  if (!cv) return;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  var x0 = 48, y0 = 28, plotW = W - 88, plotH = H - 62;
  var n = Math.min(96, out.cap.length);
  var step = Math.max(1, Math.floor(out.cap.length / n));
  var colors = { "P2-O2": "#ff1a1a", "JT": "#d946ef", "SEI": "#ff9f1a", "Rate": "#22c55e", "Residual": "#38bdf8" };
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillStyle = "#858585";
  ctx.fillText("cycle ->   y=SOC window   color=dominant loss   brightness=stress", 12, 15);
  ctx.strokeStyle = "rgba(255,26,26,0.12)";
  ctx.strokeRect(x0, y0, plotW, plotH);
  for (var gy = 0; gy <= 4; gy++) {
    var yy = y0 + gy * plotH / 4;
    ctx.strokeStyle = "rgba(255,26,26,0.06)";
    ctx.beginPath();
    ctx.moveTo(x0, yy);
    ctx.lineTo(x0 + plotW, yy);
    ctx.stroke();
  }
  for (var i = 0; i < n; i++) {
    var idx = Math.min(out.cap.length - 1, i * step);
    var soc = out.soc[idx] || 0.8;
    var intensity = clamp((out.p2Rate[idx] || 0) * 150 + (out.jtRaw[idx] || 0) * 0.16 + (out.desolv[idx] || 0) * 0.10, 0.04, 1);
    var x = x0 + (i / Math.max(1, n - 1)) * plotW;
    var y = y0 + plotH - ((soc - 0.55) / 0.43) * plotH;
    var band = 4 + intensity * 13;
    var w = Math.max(3, plotW / n * 0.9);
    ctx.fillStyle = colors[out.dominant[idx]] || "#ff1a1a";
    ctx.globalAlpha = 0.22 + intensity * 0.72;
    ctx.fillRect(x - w / 2, y - band / 2, w, band);
    ctx.globalAlpha = 0.14 + intensity * 0.22;
    ctx.fillRect(x - w / 2, y - band * 1.7, w, band * 3.4);
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(232,232,232,0.55)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (var j = 0; j < n; j++) {
    var idc = Math.min(out.cap.length - 1, j * step);
    var cx = x0 + (j / Math.max(1, n - 1)) * plotW;
    var cy = y0 + plotH - clamp((out.cap[idc] - 0.65) / 0.37, 0, 1) * plotH;
    if (j === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
  }
  ctx.stroke();
  ctx.fillStyle = "#606060";
  ctx.fillText("SOC high", 2, y0 + 8);
  ctx.fillText("SOC low", 4, y0 + plotH);
  ctx.fillText("capacity trace", W - 118, y0 + plotH + 18);
  var lx = x0, ly = y0 + plotH + 16;
  [["SEI", "#ff9f1a"], ["P2-O2", "#ff1a1a"], ["JT", "#d946ef"], ["Rate", "#22c55e"], ["Residual", "#38bdf8"]].forEach(function (item, k) {
    ctx.fillStyle = item[1];
    ctx.fillRect(lx + k * 62, ly - 8, 10, 4);
    ctx.fillStyle = "#777";
    ctx.fillText(item[0], lx + 14 + k * 62, ly - 4);
  });
}

// BMS thermal graph simulation
function buildTopology(n) {
  var cols = n <= 8 ? n : Math.ceil(Math.sqrt(n * 1.4));
  var rows = Math.ceil(n / cols);
  var pos = [];
  var edges = [];
  var nbr = Array.from({ length: n }, function () { return []; });
  for (var i = 0; i < n; i++) pos.push({ x: i % cols, y: Math.floor(i / cols) });
  for (var a = 0; a < n; a++) {
    for (var b = a + 1; b < n; b++) {
      var dx = Math.abs(pos[a].x - pos[b].x), dy = Math.abs(pos[a].y - pos[b].y);
      if ((dx === 1 && dy === 0) || (dx === 0 && dy === 1)) {
        edges.push([a, b]);
        nbr[a].push(b);
        nbr[b].push(a);
      }
    }
  }
  return { cols: cols, rows: rows, pos: pos, edges: edges, neighbors: nbr };
}

function bmsKnobs(useAsym) {
  return {
    cth: Math.max(10, num("bms-cth", 95)),
    kedge: Math.max(0, num("bms-kedge", 0.18)),
    cooling: Math.max(0, num("bms-cool", 0.045)),
    load: Math.max(0.05, num("bms-load", 1.0)),
    rctGate: Math.max(0.001, num("bms-rct-gate", 0.043)),
    threshold: useAsym ? num("bms-risk-thresh", 0.42) : Math.max(0.48, num("bms-risk-thresh", 0.55)),
    ambient: num("bms-ambient", 45) + 273.15
  };
}

function bmsScenarioKey(n, duration, fault, eis, asym, ambient, knobs) {
  knobs = knobs || bmsKnobs(asym);
  return [
    n,
    duration,
    fault ? 1 : 0,
    eis ? 1 : 0,
    asym ? 1 : 0,
    Number(ambient).toFixed(2),
    Number(knobs.cth).toFixed(2),
    Number(knobs.kedge).toFixed(4),
    Number(knobs.cooling).toFixed(4),
    Number(knobs.load).toFixed(4),
    Number(knobs.rctGate).toFixed(4),
    Number(knobs.threshold).toFixed(4)
  ].join("|");
}

function simulateBmsPhysics(n, duration, injectFault, useEis, useAsym, seed) {
  var topology = buildTopology(n);
  var cfg = bmsKnobs(useAsym);
  var steps = clamp(Math.round(duration), 60, 240);
  var dt = duration / steps;
  var ambient = cfg.ambient;
  // Use seeded PRNG for reproducible results
  var rng = mulberry32(seed != null ? seed : 42);
  var cells = [];
  var faultCell = injectFault ? Math.floor(rng() * n) : -1;
  for (var i = 0; i < n; i++) {
    cells.push({
      T: ambient + seededGaussian(rng) * 0.25,
      r0: 0.033 * (1 + seededGaussian(rng) * 0.025),
      sei: 0.010 + rng() * 0.002,
      risk: 0,
      rawHist: []
    });
  }
  var frames = [];
  var alerts = [];
  var threshold = clamp(cfg.threshold, 0.05, 0.95);
  var failureTime = duration * 0.84;
  for (var s = 0; s <= steps; s++) {
    var t = s * dt;
    var prevT = cells.map(function (c) { return c.T; });
    var raw = [];
    for (var c = 0; c < n; c++) {
      var cell = cells[c];
      var isFault = c === faultCell;
      var faultDrive = isFault ? Math.pow(sigmoid((t - duration * 0.46) / Math.max(3, duration * 0.07)), 2) : 0;
      var arrh = Math.exp(-0.28 / (8.617e-5) * (1 / cell.T - 1 / ambient));
      cell.sei += dt * (1.0e-6 * arrh + faultDrive * 7.0e-5);
      var rInt = cell.r0 + 0.18 * cell.sei + faultDrive * 0.020;
      var qOhm = cfg.load * (34 * rInt + faultDrive * 14.0);
      var coupling = 0;
      topology.neighbors[c].forEach(function (j) { coupling += cfg.kedge * (prevT[j] - prevT[c]); });
      var dTdt = (qOhm + coupling - cfg.cooling * (prevT[c] - ambient)) / cfg.cth;
      cell.T = clamp(prevT[c] + dt * dTdt, 290, 390);
      var rSei = 0.006 + 0.080 * cell.sei + faultDrive * 0.010;
      var rCt = 0.028 * Math.exp(1800 * (1 / cell.T - 1 / ambient)) * (1 + 3.5 * cell.sei + faultDrive * 1.6);
      cell.dTdt = dTdt;
      cell.rSei = rSei;
      cell.rCt = rCt;
      var tempScore = sigmoid((cell.T - 333.15) / 4.5);
      var slopeScore = sigmoid((dTdt * 60 - 1.2) / 0.7);
      var eisScore = useEis ? sigmoid((rCt + rSei - cfg.rctGate) / 0.009) : 0.25 * tempScore;
      // Use prevT for neighbor score to avoid cell-ordering artifacts
      var neighborTemp = 0;
      topology.neighbors[c].forEach(function (j) { neighborTemp += sigmoid((prevT[j] - 273.15 - 60) / 5.0); });
      neighborTemp = topology.neighbors[c].length ? neighborTemp / topology.neighbors[c].length : 0;
      raw[c] = clamp(0.34 * tempScore + 0.21 * slopeScore + 0.27 * eisScore + 0.18 * neighborTemp, 0, 1);
      cell.rawHist.push(raw[c]);
      var h = cell.rawHist;
      var lookback = 0.40 * _histBack(h, 30) + 0.28 * _histBack(h, 60) + 0.20 * _histBack(h, 120) + 0.12 * _histBack(h, 240);
      cell.risk = clamp(0.78 * cell.risk + 0.22 * lookback, 0, 1);
    }
    var risks = cells.map(function (c) { return c.risk; });
    var temps = cells.map(function (c) { return c.T - 273.15; });
    var slopes = cells.map(function (c) { return c.dTdt || 0; });
    var rcts = cells.map(function (c) { return c.rCt || 0; });
    var rseis = cells.map(function (c) { return c.rSei || 0; });
    var maxRisk = Math.max.apply(null, risks);
    var maxCell = risks.indexOf(maxRisk);
    if (s % Math.max(4, Math.floor(steps / 16)) === 0 || maxRisk > threshold) {
      var line = maxRisk > threshold
        ? { kind: "warn", text: "t=" + Math.round(t) + "s ALERT C" + maxCell + " risk=" + maxRisk.toFixed(3) + " lead=" + Math.max(0, Math.round(failureTime - t)) + "s" }
        : { kind: "info", text: "t=" + Math.round(t) + "s nominal max=" + maxRisk.toFixed(3) + " Tmax=" + Math.max.apply(null, temps).toFixed(1) + " C" };
      if (!alerts.length || alerts[alerts.length - 1].text !== line.text) alerts.push(line);
    }
    frames.push({ t: t, risks: risks.slice(), temps: temps.slice(), slopes: slopes.slice(), rcts: rcts.slice(), rseis: rseis.slice(), maxRisk: maxRisk, maxCell: maxCell });
  }
  var finalFrame = frames[frames.length - 1] || {};
  return { topology: topology, frames: frames, alerts: alerts, faultCell: faultCell, threshold: threshold, seed: seed != null ? seed : 42, maxRisk: finalFrame.maxRisk, maxCell: finalFrame.maxCell };
}

// Lookback window average helper (extracted from inner function to avoid closure-in-loop)
function _histBack(h, w) {
  var from = Math.max(0, h.length - w);
  var sum = 0;
  for (var k = from; k < h.length; k++) sum += h[k];
  return sum / Math.max(1, h.length - from);
}

function drawBmsThermal(cv, sim, frame) {
  if (!cv || !sim || !frame) return;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  var top = sim.topology;
  var pad = 24;
  var cell = Math.min((W - pad * 2) / top.cols, (H - pad * 2) / top.rows) * 0.72;
  var x0 = (W - (top.cols - 1) * cell * 1.45 - cell) / 2;
  var y0 = (H - (top.rows - 1) * cell * 1.45 - cell) / 2 + 6;
  function xy(i) {
    return { x: x0 + top.pos[i].x * cell * 1.45, y: y0 + top.pos[i].y * cell * 1.45 };
  }
  top.edges.forEach(function (e) {
    var a = xy(e[0]), b = xy(e[1]);
    var grad = Math.abs(frame.temps[e[0]] - frame.temps[e[1]]);
    ctx.strokeStyle = "rgba(255,26,26," + clamp(0.12 + grad / 30, 0.12, 0.65) + ")";
    ctx.lineWidth = 1 + clamp(grad / 8, 0, 2);
    ctx.beginPath();
    ctx.moveTo(a.x + cell / 2, a.y + cell / 2);
    ctx.lineTo(b.x + cell / 2, b.y + cell / 2);
    ctx.stroke();
  });
  frame.temps.forEach(function (temp, i) {
    var p = xy(i);
    var hot = clamp((temp - 35) / 35, 0, 1);
    var risk = frame.risks[i];
    var z = 7 + hot * 24 + risk * 20;
    var main = "rgba(" + Math.round(80 + 175 * hot) + "," + Math.round(20 + 50 * (1 - hot)) + ",0," + (0.25 + 0.65 * Math.max(hot, risk)) + ")";
    ctx.strokeStyle = i === sim.faultCell ? "#ffb020" : "rgba(255,26,26," + (0.2 + risk * 0.7) + ")";
    ctx.lineWidth = i === sim.faultCell ? 2 : 1;
    ctx.shadowColor = "#ff1a1a";
    ctx.shadowBlur = risk > sim.threshold ? 12 : 0;
    ctx.fillStyle = "rgba(60,8,0,0.55)";
    ctx.beginPath();
    ctx.moveTo(p.x, p.y + cell);
    ctx.lineTo(p.x + cell, p.y + cell);
    ctx.lineTo(p.x + cell, p.y + cell - z);
    ctx.lineTo(p.x, p.y + cell - z);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = main;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y + cell - z);
    ctx.lineTo(p.x + cell, p.y + cell - z);
    ctx.lineTo(p.x + cell * 0.82, p.y + cell - z - 10);
    ctx.lineTo(p.x + cell * 0.18, p.y + cell - z - 10);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#f5f5f5";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillText("C" + i, p.x + 6, p.y + cell - z - 13);
    ctx.fillStyle = "#d0d0d0";
    ctx.fillText(temp.toFixed(1) + "C", p.x + 6, p.y + cell - 8);
  });
  ctx.fillStyle = "#777";
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillText("Thermal coupling: Cth dT/dt = q + sum(kij(Tj-Ti)) - h(T-Ta)", 12, 14);
}

function updateBMS() {
  document.getElementById("pack-val").textContent = document.getElementById("pack-slider").value + " cells";
  document.getElementById("dur-val").textContent = document.getElementById("dur-slider").value + "s";
  var amb = document.getElementById("bms-ambient-val");
  if (amb) amb.textContent = num("bms-ambient", 45).toFixed(0) + " C";
}

function runBMS() {
  var runId = ++activeBmsRun;
  var n = parseInt(document.getElementById("pack-slider").value, 10);
  var dur = parseInt(document.getElementById("dur-slider").value, 10);
  var fault = document.getElementById("sw-fault").checked;
  var eis = document.getElementById("sw-eis").checked;
  var asym = document.getElementById("sw-asym").checked;
  var seedEl = document.getElementById("bms-seed");
  var seed = seedEl ? parseInt(seedEl.value, 10) : 42;
  if (!Number.isFinite(seed)) seed = 42;
  var sim = simulateBmsPhysics(n, dur, fault, eis, asym, seed);
  var ambientC = num("bms-ambient", 45);
  var scenarioKey = bmsScenarioKey(n, dur, fault, eis, asym, ambientC);
  window.__kfBms = {
    n: n,
    duration: dur,
    threshold: sim.threshold,
    faultCell: sim.faultCell,
    frames: sim.frames,
    alerts: sim.alerts,
    ambient_C: ambientC,
    seed: seed,
    maxRisk: sim.maxRisk,
    scenarioKey: scenarioKey
  };
  var sweepNote = document.getElementById("bms-sweep-note");
  if (sweepNote) {
    if (window.__kfBmsSweep && window.__kfBmsSweep.scenarioKey === scenarioKey) {
      sweepNote.textContent = "Sweep " + window.__kfBmsSweep.count + " seeds: alert rate " + Math.round(window.__kfBmsSweep.alertRate * 100) + "%, mean max-risk " + window.__kfBmsSweep.meanRisk.toFixed(3) + " +/- " + window.__kfBmsSweep.stdRisk.toFixed(3) + ".";
    } else {
      sweepNote.textContent = "Run a seed sweep to quantify how sensitive this setup is to stochastic initialization.";
    }
  }
  var grid = document.getElementById("bms-grid");
  grid.innerHTML = "";
  grid.style.gridTemplateColumns = "repeat(" + Math.min(n, 8) + ", minmax(54px, 1fr))";
  for (var i = 0; i < n; i++) {
    var d = document.createElement("div");
    d.className = "cell-tile";
    d.dataset.cell = "C" + i;
    d.dataset.fault = i === sim.faultCell ? "1" : "0";
    d.innerHTML = '<div class="cell-name">C' + i + '</div><div class="cell-risk">0.00</div><div class="cell-temp">-- C</div>';
    grid.appendChild(d);
  }
  var log = document.getElementById("bms-log");
  log.innerHTML = '<div class="cmd">$ bms --thermal-ode --cells=' + n + " --fault=" + (sim.faultCell >= 0 ? "C" + sim.faultCell : "none") + ' --seed=' + seed + '</div><div class="info">Pack graph built with ' + sim.topology.edges.length + " thermal edges. Ambient=" + ambientC.toFixed(0) + " C EIS=" + (eis ? "on" : "off") + " threshold=" + sim.threshold.toFixed(2) + " seed=" + seed + "</div>";
  var cv = makeCanvas("bms-thermal-chart");
  var trendCv = makeCanvas("bms-trend-chart");
  var idx = 0;
  function paint() {
    if (runId !== activeBmsRun) return;
    idx = Math.min(idx + 2, sim.frames.length - 1);
    var frame = sim.frames[idx];
    Array.from(grid.children).forEach(function (el, i) {
      var r = frame.risks[i];
      var hot = clamp((frame.temps[i] - 35) / 35, 0, 1);
      el.style.background = "rgba(" + Math.round(60 + 195 * Math.max(r, hot)) + ",8,0," + (0.18 + 0.65 * Math.max(r, hot)) + ")";
      el.style.borderColor = "rgba(255,26,26," + (0.15 + r * 0.75) + ")";
      el.style.boxShadow = r > sim.threshold ? "0 0 24px rgba(255,26,26,0.45)" : "none";
      el.dataset.risk = r.toFixed(4);
      el.dataset.temp = frame.temps[i].toFixed(2);
      el.dataset.slope = frame.slopes[i].toFixed(5);
      el.dataset.rct = frame.rcts[i].toFixed(5);
      el.dataset.rsei = frame.rseis[i].toFixed(5);
      el.dataset.hot = hot > 0.35 ? "1" : "0";
      el.querySelector(".cell-risk").textContent = r.toFixed(2);
      el.querySelector(".cell-temp").textContent = frame.temps[i].toFixed(1) + " C";
    });
    drawBmsThermal(cv, sim, frame);
    var upto = sim.frames.slice(0, idx + 1);
    drawMultiLine(trendCv, [
      { name: "max risk", values: upto.map(function (f) { return f.maxRisk; }), color: "#ff1a1a", glow: true },
      { name: "Tmax/100", values: upto.map(function (f) { return Math.max.apply(null, f.temps) / 100; }), color: "#ff9f1a" }
    ], { yMin: 0, yMax: 1, xMax: dur, title: "Risk and pack temperature trend", legend: true, yDigits: 2 });
    if (idx < sim.frames.length - 1) requestAnimationFrame(paint);
    else {
      sim.alerts.forEach(function (a) { log.innerHTML += '<div class="' + a.kind + '">' + escapeHtml(a.text) + "</div>"; });
      log.innerHTML += '<div class="ok">Complete. fault=' + (sim.faultCell >= 0 ? "C" + sim.faultCell : "none") + " maxRisk=" + frame.maxRisk.toFixed(3) + "</div>";
      log.scrollTop = log.scrollHeight;
      var action = frame.maxRisk > sim.threshold
        ? "Cool or isolate C" + frame.maxCell + " first; inspect impedance rise before pack-level thermal spread."
        : "No cell crossed the action threshold. Continue monitoring; lower the threshold if false negatives matter more than false positives.";
      window.__kfBms.maxRisk = frame.maxRisk;
      window.__kfBms.maxCell = frame.maxCell;
      window.__kfBms.finalTmax = Math.max.apply(null, frame.temps);
      var bmsConf = computeBmsConfidence(window.__kfBms, frame);
      setHtml("bms-decision", "<strong>BMS output:</strong> C" + frame.maxCell + " has max risk " + frame.maxRisk.toFixed(3) + ". <strong>Action:</strong> " + action);
      setReadouts("bms-readout", [
        { k: "Highest cell", v: "C" + frame.maxCell },
        { k: "Risk / gate", v: frame.maxRisk.toFixed(3) + " / " + sim.threshold.toFixed(2) },
        { k: "Tmax", v: Math.max.apply(null, frame.temps).toFixed(1) + " C" },
        { k: "Rct + RSEI", v: (frame.rcts[frame.maxCell] + frame.rseis[frame.maxCell]).toFixed(3) + " ohm" }
      ]);
      renderConfidence("bms-confidence", bmsConf.confidence, bmsConf.detail);
      recordRun("bms", {
        summary: "C" + frame.maxCell + " risk " + frame.maxRisk.toFixed(3) + " / gate " + sim.threshold.toFixed(2) + ", seed " + seed,
        key_metric: Math.max.apply(null, frame.temps).toFixed(1) + " C Tmax",
        confidence: bmsConf.confidence
      });
    }
  }
  requestAnimationFrame(paint);
}

function runBmsSweep() {
  var n = parseInt(document.getElementById("pack-slider").value, 10);
  var dur = parseInt(document.getElementById("dur-slider").value, 10);
  var fault = document.getElementById("sw-fault").checked;
  var eis = document.getElementById("sw-eis").checked;
  var asym = document.getElementById("sw-asym").checked;
  var ambientC = num("bms-ambient", 45);
  var seedEl = document.getElementById("bms-seed");
  var sweepEl = document.getElementById("bms-sweep-count");
  var seedBase = seedEl ? parseInt(seedEl.value, 10) : 42;
  if (!Number.isFinite(seedBase)) seedBase = 42;
  var sweepCount = sweepEl ? parseInt(sweepEl.value, 10) : 16;
  if (!Number.isFinite(sweepCount)) sweepCount = 16;
  sweepCount = Math.max(3, Math.min(80, sweepCount));
  var scenarioKey = bmsScenarioKey(n, dur, fault, eis, asym, ambientC);

  var maxRisks = [];
  var tmaxs = [];
  var maxCells = {};
  var alerts = 0;
  for (var i = 0; i < sweepCount; i++) {
    var sim = simulateBmsPhysics(n, dur, fault, eis, asym, seedBase + i);
    var frame = sim.frames[sim.frames.length - 1];
    if (!frame) continue;
    var maxRisk = Number(frame.maxRisk);
    if (!Number.isFinite(maxRisk)) continue;
    maxRisks.push(maxRisk);
    tmaxs.push(Math.max.apply(null, frame.temps));
    maxCells[frame.maxCell] = (maxCells[frame.maxCell] || 0) + 1;
    if (maxRisk > sim.threshold) alerts += 1;
  }
  if (!maxRisks.length) {
    showToast("Sweep failed to produce any frames.", "warn");
    return;
  }
  var dominantCell = Object.keys(maxCells).map(function (k) {
    return { cell: Number(k), hits: maxCells[k] };
  }).sort(function (a, b) { return b.hits - a.hits; })[0];
  var meanRisk = mean(maxRisks);
  var stdRisk = stddev(maxRisks);
  var alertRate = alerts / maxRisks.length;
  var meanTmax = mean(tmaxs);
  window.__kfBmsSweep = {
    scenarioKey: scenarioKey,
    count: maxRisks.length,
    startSeed: seedBase,
    endSeed: seedBase + maxRisks.length - 1,
    meanRisk: meanRisk,
    stdRisk: stdRisk,
    alertRate: alertRate,
    meanTmax: meanTmax,
    dominantCell: dominantCell ? dominantCell.cell : null
  };

  var note = document.getElementById("bms-sweep-note");
  if (note) {
    note.textContent = "Sweep " + maxRisks.length + " seeds: alert rate " + Math.round(alertRate * 100) + "%, mean max-risk " + meanRisk.toFixed(3) + " \u00B1 " + stdRisk.toFixed(3) + ", dominant cell C" + (dominantCell ? dominantCell.cell : "--") + ".";
  }
  var log = document.getElementById("bms-log");
  if (log) {
    log.innerHTML += '<div class="info">Sweep seeds ' + seedBase + "-" + (seedBase + maxRisks.length - 1) + ": alert rate " + (alertRate * 100).toFixed(1) + "%, mean maxRisk " + meanRisk.toFixed(3) + ", sigma " + stdRisk.toFixed(3) + ", dominant C" + (dominantCell ? dominantCell.cell : "--") + ".</div>";
    log.scrollTop = log.scrollHeight;
  }

  if (window.__kfBms && window.__kfBms.scenarioKey === scenarioKey && window.__kfBms.frames && window.__kfBms.frames.length) {
    var last = window.__kfBms.frames[window.__kfBms.frames.length - 1];
    var bmsConf = computeBmsConfidence(window.__kfBms, last);
    renderConfidence("bms-confidence", bmsConf.confidence, bmsConf.detail);
  }
  var sweepConf = clamp(0.74 - stdRisk * 1.8 - Math.abs(alertRate - 0.5) * 0.10, 0.26, 0.86);
  recordRun("bms_sweep", {
    summary: maxRisks.length + " seeds, alert rate " + (alertRate * 100).toFixed(1) + "%, dominant C" + (dominantCell ? dominantCell.cell : "--"),
    key_metric: "mean risk " + meanRisk.toFixed(3) + " +/- " + stdRisk.toFixed(3),
    confidence: sweepConf
  });
}

// Materials screening mirror from modules/cathode/screener.py.
function dopantEffects(comp) {
  if (comp.ti) return { fade: 0.90, life: 1.10, cap: 0.99, vol: 0.90, rate: 1.08 };
  if (comp.al) return { fade: 0.82, life: 1.18, cap: 0.97, vol: 0.85, rate: 1.05 };
  return { fade: 1, life: 1, cap: 1, vol: 1, rate: 1 };
}
function materialKnobs() {
  return {
    wCap: num("mat-w-cap", 0.32),
    wStab: num("mat-w-stab", 0.32),
    wFade: num("mat-w-fade", 0.22),
    wCost: num("mat-w-cost", 0.14),
    upperV: num("mat-upper-v", 4.10),
    ehullSlope: Math.max(1, num("mat-ehull-slope", 20))
  };
}
function scoreComposition(comp, T, cfg) {
  T = T || 318.15;
  cfg = cfg || materialKnobs();
  var eff = dopantEffects(comp);
  var dopFrac = (comp.al ? 0.04 : 0) + (comp.ti ? 0.03 : 0);
  var q0 = (120 + 40 * comp.Mn - 20 * comp.Fe) * eff.cap * (1 - 0.5 * Math.abs(comp.Na - 1.0));
  var Ea = 0.55 + 0.1 * comp.Mn - 0.03 * comp.Fe;
  var kFade = 1e-4 * (1 + 0.2 * comp.Fe) * Math.exp(-Ea * 96485 / (8.314 * T));
  var jt = 1 + 0.3 * Math.max(0, comp.Mn - 0.5);
  var ss = 1 / (1 + Math.exp(-8 * (0.5 - comp.Mn)));
  var feStab = 0.9 + 0.2 * comp.Fe;
  var effFade = kFade * jt * eff.fade / feStab;
  var voltageStress = 1 + 1.8 * sigmoid((cfg.upperV - 4.05) / 0.08);
  var fade500 = clamp(1 - Math.exp(-effFade * voltageStress * Math.pow(500, 1.15)), 0.02, 0.48);
  var cycleLife = 400 * eff.life / jt * (comp.Mn > 0.6 ? 0.85 : 1);
  var rateCap = (0.85 + 0.1 * comp.Fe - 0.05 * comp.Mn) * eff.rate + (comp.al ? 0.03 : 0);
  // Voltage from weighted redox couples: Fe3+/4+ ~3.20V, Mn3+/4+ ~3.75V in P2-type
  var feW = comp.Fe / Math.max(comp.Mn + comp.Fe, 0.01);
  var mnW = comp.Mn / Math.max(comp.Mn + comp.Fe, 0.01);
  var avgVoltage = mnW * 3.75 + feW * 3.20 + 0.15 * (1.0 - comp.Na);
  var energyDensity = q0 * avgVoltage;
  var volChange = (2.0 + 3.0 * comp.Mn - comp.Fe) * eff.vol;
  var eForm = -4.2 - 0.6 * comp.Mn - 0.35 * comp.Fe - 0.4 * comp.Na - (comp.al ? 0.048 : 0) - (comp.ti ? 0.054 : 0);
  var eHullProducts = -4.0 - 0.3 * comp.Mn - 0.2 * comp.Fe;
  var ehull = Math.max(0, eForm - eHullProducts + 0.05);
  var phaseStab = 1 / (1 + Math.exp(cfg.ehullSlope * (ehull - 0.05)));
  var thermalAbuse = clamp((250 - 30 * Math.max(0, comp.Mn - 0.5) + 15 * comp.Fe + (comp.al ? 8 : 0) + (comp.ti ? 7.5 : 0) - 180) / 120, 0, 1);
  var oxygenRisk = clamp(0.22 + Math.max(0, comp.Mn - 0.55) + Math.max(0, 1 - comp.Na) * 0.8 + 0.24 * sigmoid((cfg.upperV - 4.15) / 0.07) - (comp.al ? 0.06 : 0) - (comp.ti ? 0.08 : 0), 0, 1);
  var mixingRisk = clamp(0.18 + Math.abs(comp.Mn - comp.Fe) * 0.35 + Math.max(0, 0.98 - comp.Na) * 1.2 + (comp.ti ? 0.03 : comp.al ? -0.02 : 0), 0, 1);
  var moisture = clamp(0.20 + Math.max(0, comp.Na - 0.98) * 0.9 + Math.max(0, 1 - comp.Na) * 2.2, 0, 1);
  var jtRisk = clamp((comp.Mn - 0.48) * 1.8 - (comp.ti ? 0.18 : 0), 0, 1);
  var defectScore = clamp(1 - (0.24 * oxygenRisk + 0.22 * mixingRisk + 0.20 * moisture + 0.24 * jtRisk), 0, 1);
  var costKg = comp.Na * 3.1 * 0.23 + comp.Mn * 2.4 * 0.55 + comp.Fe * 0.45 * 0.56 + dopFrac * (comp.ti ? 11.0 * 0.479 : comp.al ? 2.7 * 0.27 : 0) + 2.5;
  var costKwh = costKg / Math.max(energyDensity / 1000, 0.01);
  var stability = clamp(0.28 * (1 - fade500) + 0.18 * ss * feStab + 0.18 * phaseStab + 0.16 * thermalAbuse + 0.20 * defectScore, 0, 1);
  // Mn valence from charge balance: Na + Mn*v_Mn + Fe*3 + dop*v_dop = 4 (for AMO2)
  var dopCharge = (comp.al ? 0.04 * 3.0 : 0) + (comp.ti ? 0.03 * 4.0 : 0);
  var mnOx = clamp((4.0 - comp.Na - comp.Fe * 3.0 - dopCharge) / Math.max(comp.Mn, 0.01), 3.0, 4.0);
  var feOx = 3.0;
  var totalCharge = comp.Na + comp.Mn * mnOx + comp.Fe * feOx + dopCharge;
  var chargeBalanceRisk = clamp(Math.abs(totalCharge - 4.0) / 1.4, 0, 1);
  var score = cfg.wCap * (q0 / 180) + cfg.wStab * stability + cfg.wFade * (1 - fade500) + cfg.wCost * Math.max(0, 1 - costKwh / 200) - 0.08 * chargeBalanceRisk;
  return { Q0: q0, Q500: q0 * (1 - fade500), fade500: fade500, cycleLife: cycleLife, avgVoltage: avgVoltage, stability: stability, jtIndex: jtRisk, energyDensity: energyDensity, costKwh: costKwh, score: score, oxygenRisk: oxygenRisk, chargeRisk: chargeBalanceRisk };
}

function generateCandidates(selected, cfg) {
  var pts = [];
  for (var na = 0.84; na <= 1.12; na += 0.04) {
    for (var mn = 0.20; mn <= 0.82; mn += 0.04) {
      var fe = clamp(1.0 - mn - ((selected.al ? 0.04 : 0) + (selected.ti ? 0.03 : 0)), 0.12, 0.82);
      ["none", "al", "ti"].forEach(function (d) {
        var comp = { Na: na, Mn: mn, Fe: fe, al: d === "al", ti: d === "ti" };
        var prop = scoreComposition(comp, 318.15, cfg);
        pts.push({ comp: comp, prop: prop });
      });
    }
  }
  return pts;
}

function paretoMark(items) {
  items.forEach(function (it) { it.front = true; });
  for (var i = 0; i < items.length; i++) {
    for (var j = 0; j < items.length; j++) {
      if (i === j) continue;
      var a = items[j].prop, b = items[i].prop;
      var ge = a.Q0 >= b.Q0 && -a.fade500 >= -b.fade500 && a.cycleLife >= b.cycleLife && -a.costKwh >= -b.costKwh;
      var gt = a.Q0 > b.Q0 || -a.fade500 > -b.fade500 || a.cycleLife > b.cycleLife || -a.costKwh > -b.costKwh;
      if (ge && gt) { items[i].front = false; break; }
    }
  }
}

function updateMat() {
  document.getElementById("na-val").textContent = parseFloat(document.getElementById("na-slider").value).toFixed(2);
  document.getElementById("mn-val").textContent = parseFloat(document.getElementById("mn-slider").value).toFixed(2);
  document.getElementById("fe-val").textContent = parseFloat(document.getElementById("fe-slider").value).toFixed(2);
}

function runScreening() {
  var cfg = materialKnobs();
  var selected = {
    Na: parseFloat(document.getElementById("na-slider").value),
    Mn: parseFloat(document.getElementById("mn-slider").value),
    Fe: parseFloat(document.getElementById("fe-slider").value),
    al: document.getElementById("sw-al").checked,
    ti: document.getElementById("sw-ti").checked
  };
  var selectedProp = scoreComposition(selected, 318.15, cfg);
  var items = generateCandidates(selected, cfg);
  items.push({ comp: selected, prop: selectedProp, selected: true });
  paretoMark(items);
  document.getElementById("mat-cap").textContent = selectedProp.Q0.toFixed(0);
  document.getElementById("mat-volt").textContent = selectedProp.avgVoltage.toFixed(2) + "V";
  document.getElementById("mat-stab").textContent = selectedProp.stability.toFixed(2);
  document.getElementById("mat-jt").textContent = selectedProp.jtIndex.toFixed(2);
  selectedProp.comp = selected;
  window.__kfMaterials = selectedProp;
  var cv = makeCanvas("mat-chart");
  var pts = items.map(function (it) {
    return { x: it.prop.Q0, y: it.prop.stability, front: it.front, selected: !!it.selected };
  });
  var frame = 0;
  function anim() {
    frame++;
    var n = Math.min(frame * 8, pts.length);
    drawScatter(cv, pts.slice(0, n), { title: "Computed Pareto front: capacity vs stability" });
    if (n < pts.length) requestAnimationFrame(anim);
  }
  anim();
  drawCompositionLandscape(makeCanvas("mat-landscape-chart"), items, selected);
  var synth = selectedProp.stability > 0.72 && selectedProp.fade500 < 0.16 && selectedProp.chargeRisk < 0.28;
  var advice = synth
    ? "Good candidate for a small coin-cell synthesis queue, with XRD phase check before cycling."
    : "Keep in simulation queue; improve stability/charge balance before spending lab synthesis effort.";
  setHtml("mat-decision", "<strong>Screening output:</strong> score " + selectedProp.score.toFixed(3) + ", fade500 " + (100 * selectedProp.fade500).toFixed(1) + "%, oxygen risk " + selectedProp.oxygenRisk.toFixed(2) + ". <strong>Decision:</strong> " + advice + ' <button class="btn btn-ghost" style="margin-left:0.75rem" onclick="testMaterialInDiagnostics()">Test in Diagnostics &rarr;</button>');
  setReadouts("mat-risk-readout", [
    { k: "Objective", v: selectedProp.score.toFixed(3) },
    { k: "Fade@500", v: (100 * selectedProp.fade500).toFixed(1) + "%" },
    { k: "Oxygen risk", v: selectedProp.oxygenRisk.toFixed(2) },
    { k: "Cost proxy", v: "$" + selectedProp.costKwh.toFixed(0) + "/kWh" }
  ]);
  var matConf = computeMaterialsConfidence(selectedProp);
  renderConfidence("mat-confidence", matConf.confidence, matConf.detail);
  recordRun("materials", {
    summary: "score " + selectedProp.score.toFixed(3) + ", stability " + selectedProp.stability.toFixed(2) + ", fade@500 " + (100 * selectedProp.fade500).toFixed(1) + "%",
    key_metric: selectedProp.Q0.toFixed(0) + " mAh/g",
    confidence: matConf.confidence
  });
}

function drawCompositionLandscape(cv, items, selected) {
  if (!cv || !items.length) return;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillStyle = "#777";
  ctx.fillText("Composition score landscape: x=Mn, y=Na, height/color=objective", 12, 15);
  var grid = items.filter(function (it) { return it.comp.al === selected.al && it.comp.ti === selected.ti; });
  var minS = Math.min.apply(null, grid.map(function (g) { return g.prop.score; }));
  var maxS = Math.max.apply(null, grid.map(function (g) { return g.prop.score; }));
  var x0 = W * 0.18, y0 = H * 0.78, sx = W * 0.48, sy = H * 0.34;
  grid.sort(function (a, b) { return (a.comp.Na + a.comp.Mn) - (b.comp.Na + b.comp.Mn); });
  grid.forEach(function (it) {
    var mn = (it.comp.Mn - 0.2) / 0.62;
    var na = (it.comp.Na - 0.84) / 0.28;
    var s = (it.prop.score - minS) / Math.max(1e-6, maxS - minS);
    var x = x0 + mn * sx + na * 34;
    var y = y0 - na * sy - s * 52;
    var color = "rgba(255," + Math.round(50 + 120 * s) + "," + Math.round(26 + 80 * (1 - s)) + "," + (0.38 + 0.45 * s) + ")";
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 16, y - 7);
    ctx.lineTo(x + 32, y);
    ctx.lineTo(x + 16, y + 7);
    ctx.closePath();
    ctx.fill();
  });
  var selectedItem = { comp: selected, prop: scoreComposition(selected, 318.15, materialKnobs()) };
  var smn = (selectedItem.comp.Mn - 0.2) / 0.62;
  var sna = (selectedItem.comp.Na - 0.84) / 0.28;
  var ss = (selectedItem.prop.score - minS) / Math.max(1e-6, maxS - minS);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x0 + smn * sx + sna * 34 + 16, y0 - sna * sy - ss * 52, 8, 0, Math.PI * 2);
  ctx.stroke();
}

// Recycling: shrinking-core leaching with Bayesian recovery priors.
function betaMean(a, b) { return a / (a + b); }
function shrinkingCoreConversion(k, tMin) {
  // Reaction-controlled: X = 1 - (1 - kt)^3. Clamp final conversion, not intermediate.
  var kt = k * tMin;
  if (kt >= 1.0) return 0.995;
  return clamp(1 - Math.pow(Math.max(1 - kt, 0), 3), 0, 0.995);
}
function recoveryForElement(el, acid, tempC, tMin, bayes) {
  var R = 8.314;
  var T = tempC + 273.15;
  var tempFactor = Math.exp(-el.Ea / R * (1 / T - 1 / 353.15));
  var k = el.k0 * Math.pow(acid, el.order) * tempFactor * Math.pow(50 / el.particle, 0.35);
  var x = shrinkingCoreConversion(k, tMin);
  if (!bayes || !el.prior) return x;
  return clamp(0.75 * x + 0.25 * betaMean(el.prior[0], el.prior[1]), 0, 0.995);
}

function updateRecycling() {
  document.getElementById("bm-val").textContent = document.getElementById("bm-slider").value;
  document.getElementById("acid-val").textContent = parseFloat(document.getElementById("acid-slider").value).toFixed(1);
  document.getElementById("leach-val").textContent = document.getElementById("leach-slider").value + " C";
}

function recyclingKnobs() {
  return {
    time: Math.max(5, num("rec-time", 120)),
    particle: Math.max(2, num("rec-particle", 50)),
    acidOrder: Math.max(0.05, num("rec-acid-order", 0.95)),
    eaMn: Math.max(5000, num("rec-ea-mn", 27000))
  };
}

function recyclingScenarioSeed(mass, acid, temp, cfg) {
  var text = [mass, acid, temp, cfg.time, cfg.particle, cfg.acidOrder, cfg.eaMn].map(function (x) {
    return Number(x).toFixed(4);
  }).join("|");
  var h = 2166136261;
  for (var i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function runRecycling() {
  var mass = parseFloat(document.getElementById("bm-slider").value);
  var acid = parseFloat(document.getElementById("acid-slider").value);
  var temp = parseFloat(document.getElementById("leach-slider").value);
  var mc = document.getElementById("sw-mc").checked;
  var bay = document.getElementById("sw-bayes").checked;
  var cfg = recyclingKnobs();
  var tFinal = cfg.time;
  var elements = [
    { n: "Mn", wt: 0.22, k0: 0.0038, Ea: cfg.eaMn, order: cfg.acidOrder, particle: cfg.particle, prior: [8.8, 1.2] },
    { n: "Fe", wt: 0.11, k0: 0.0029, Ea: 30000, order: Math.max(0.1, cfg.acidOrder - 0.10), particle: cfg.particle * 1.1, prior: [7.2, 2.8] },
    { n: "Na", wt: 0.05, k0: 0.0062, Ea: 19000, order: Math.max(0.1, cfg.acidOrder - 0.35), particle: cfg.particle * 0.9, prior: [6.5, 3.5] },
    { n: "Al", wt: 0.04, k0: 0.0011, Ea: 36000, order: cfg.acidOrder + 0.10, particle: cfg.particle * 1.3 },
    { n: "Cu", wt: 0.015, k0: 0.0007, Ea: 34000, order: Math.max(0.1, cfg.acidOrder - 0.15), particle: cfg.particle * 1.4 }
  ];
  var bars = document.getElementById("recycle-bars");
  bars.innerHTML = "";
  var targets = elements.map(function (el) { return recoveryForElement(el, acid, temp, tFinal, bay); });
  elements.forEach(function (el, i) {
    var row = document.createElement("div");
    row.className = "recovery-row";
    row.innerHTML = '<div class="bar-head"><span>' + el.n + '</span><span id="rv-' + el.n + '">0.0%</span></div><div class="progress-bar"><div class="fill" id="rb-' + el.n + '" style="width:0%"></div></div>';
    bars.appendChild(row);
    setTimeout(function () {
      document.getElementById("rb-" + el.n).style.width = (targets[i] * 100).toFixed(1) + "%";
      document.getElementById("rv-" + el.n).textContent = (targets[i] * 100).toFixed(1) + "%";
    }, 80 + i * 50);
  });
  var timeline = [];
  for (var t = 0; t <= tFinal; t += 4) {
    timeline.push({
      t: t,
      Mn: recoveryForElement(elements[0], acid, temp, t, bay) * 100,
      Fe: recoveryForElement(elements[1], acid, temp, t, bay) * 100,
      Na: recoveryForElement(elements[2], acid, temp, t, bay) * 100
    });
  }
  drawMultiLine(makeCanvas("recycle-chart"), [
    { name: "Mn", values: timeline.map(function (x) { return x.Mn; }), color: "#ff1a1a", glow: true },
    { name: "Fe", values: timeline.map(function (x) { return x.Fe; }), color: "#ff9f1a" },
    { name: "Na", values: timeline.map(function (x) { return x.Na; }), color: "#38bdf8" }
  ], { yMin: 0, yMax: 100, xMax: tFinal, title: "Shrinking-core conversion over leach time", legend: true, yDigits: 0 });

  var totalRecovered = 0;
  targets.forEach(function (r, i) { totalRecovered += mass * elements[i].wt * r; });
  var mcTotals = [];
  if (mc) {
    var rng = mulberry32(recyclingScenarioSeed(mass, acid, temp, cfg));
    for (var s = 0; s < RECYCLING_MC_SAMPLES; s++) {
      var tot = 0;
      elements.forEach(function (el, i) {
        var feedNoise = clamp(1 + seededGaussian(rng) * 0.08, 0.75, 1.25);
        var assayNoise = clamp(1 + seededGaussian(rng) * 0.025, 0.92, 1.08);
        tot += mass * el.wt * feedNoise * targets[i] * assayNoise;
      });
      mcTotals.push(tot);
    }
    mcTotals.sort(function (a, b) { return a - b; });
  }
  var lo = mc ? mcTotals[Math.floor(mcTotals.length * 0.05)] : totalRecovered;
  var hi = mc ? mcTotals[Math.floor(mcTotals.length * 0.95)] : totalRecovered;
  var acidKg = acid * 0.098 * mass;
  var heatKwh = Math.max(0, temp - 25) * mass * 0.00116;
  var cost = acidKg * 8.5 + heatKwh * 8.0 + mass * 150;
  var impurityPenalty = clamp(targets[3] * 0.28 + targets[4] * 0.36, 0, 0.8);
  var productPurity = clamp(0.94 - impurityPenalty * 0.18 + (targets[0] + targets[1] + targets[2]) * 0.012, 0.70, 0.98);
  var log = document.getElementById("recycle-log");
  log.innerHTML = '<div class="cmd">$ recycling --shrinking-core --mass=' + mass + "kg --acid=" + acid + "M --temp=" + temp + 'C</div>';
  log.innerHTML += '<div class="info">ODE: 1 - (1-X)^(1/3) = k(C_acid,T,Rp)t, with Arrhenius temperature scaling.</div>';
  if (mc) log.innerHTML += '<div class="info">Monte Carlo: ' + RECYCLING_MC_SAMPLES + ' feedstock and assay samples with deterministic scenario seed. Recovery physics unchanged per sample.</div>';
  if (bay) log.innerHTML += '<div class="info">Bayesian priors: Mn Beta(8.8,1.2), Fe Beta(7.2,2.8), Na Beta(6.5,3.5).</div>';
  log.innerHTML += '<div class="ok">Recovered metals: ' + totalRecovered.toFixed(1) + "kg, 90% interval " + lo.toFixed(1) + "-" + hi.toFixed(1) + "kg</div>";
  log.innerHTML += '<div class="info">Process cost estimate: INR ' + cost.toFixed(0) + " per batch; product purity proxy " + (productPurity * 100).toFixed(1) + "%</div>";
  var marginProxy = totalRecovered * 620 * productPurity - cost;
  var decision = marginProxy > 0 && targets[0] > 0.82 && productPurity > 0.86
    ? "Run this recipe as a pilot batch; Mn recovery and economics are inside the current gate."
    : "Do not run as-is; adjust acid/time/particle size or improve impurity control before pilot scale.";
  window.__kfRecycling = { totalRecovered: totalRecovered, lo: lo, hi: hi, purity: productPurity, cost: cost, marginProxy: marginProxy, elements: elements, targets: targets, mass: mass };
  setHtml("recycle-decision", "<strong>Recipe output:</strong> recovered " + totalRecovered.toFixed(1) + " kg, Mn " + (targets[0] * 100).toFixed(1) + "%, purity proxy " + (productPurity * 100).toFixed(1) + "%, cost INR " + cost.toFixed(0) + ". <strong>Decision:</strong> " + decision);
  setReadouts("recycle-readout", [
    { k: "Recovered", v: totalRecovered.toFixed(1) + " kg" },
    { k: "90% interval", v: lo.toFixed(1) + "-" + hi.toFixed(1) + " kg" },
    { k: "Purity proxy", v: (productPurity * 100).toFixed(1) + "%" },
    { k: "Margin proxy", v: "INR " + marginProxy.toFixed(0) }
  ]);
  var recConf = computeRecyclingConfidence(window.__kfRecycling);
  renderConfidence("recycle-confidence", recConf.confidence, recConf.detail);
  recordRun("recycling", {
    summary: "recovered " + totalRecovered.toFixed(1) + " kg, purity " + (productPurity * 100).toFixed(1) + "%, margin INR " + compactNumber(marginProxy, 0),
    key_metric: "Mn " + (targets[0] * 100).toFixed(1) + "%",
    confidence: recConf.confidence
  });
}

function setAssistantOpen(open) {
  var dock = document.getElementById("assistant-dock");
  var input = document.getElementById("assistant-input");
  if (!dock) return;
  dock.classList.toggle("open", !!open);
  if (open && input) setTimeout(function () { input.focus(); }, 120);
}

// Simple markdown renderer for chatbot responses
function renderMd(text) {
  if (!text) return "";
  var html = escapeHtml(text);
  // Bold: **text** or __text__
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
  // Italic fallback without lookbehind (works on older WebKit/Safari engines)
  html = html.replace(/(^|[\s(>])\*([^*\n][^*\n]*?)\*(?=[\s).,!?;:]|$)/gm, '$1<em>$2</em>');
  // Inline code: `text`
  html = html.replace(/`([^`]+)`/g, '<code style="background:rgba(255,26,26,0.12);padding:1px 4px;border-radius:3px;font-size:0.9em">$1</code>');
  // Bullet points: lines starting with - or *
  html = html.replace(/^(\s*)[\-\*]\s+(.+)/gm, '$1<span style="color:#ff5c5c">•</span> $2');
  // Numbered lists: lines starting with 1. 2. etc
  html = html.replace(/^(\s*)\d+\.\s+(.+)/gm, '$1<span style="color:#ff5c5c">▸</span> $2');
  // Newlines
  html = html.replace(/\n/g, '<br>');
  return html;
}

function assistantMessage(kind, text, meta) {
  var body = document.getElementById("assistant-body");
  if (!body) return null;
  var msg = document.createElement("div");
  msg.className = "assistant-msg " + kind;
  // Use markdown rendering for bot messages, plain for user messages
  if (kind.indexOf("user") >= 0) {
    msg.innerHTML = escapeHtml(text || "").replace(/\n/g, "<br>");
  } else {
    msg.innerHTML = renderMd(text || "");
  }
  if (meta) {
    var small = document.createElement("div");
    small.className = "assistant-meta";
    small.textContent = meta;
    msg.appendChild(small);
  }
  body.appendChild(msg);
  body.scrollTop = body.scrollHeight;
  return msg;
}

function updateAssistantFoot(data) {
  var foot = document.getElementById("assistant-foot");
  if (!foot) return;
  if (!data) {
    foot.textContent = "Cloud assistant uses server-side OpenRouter. Memory off.";
    return;
  }
  var source = data.source === "openrouter" ? "Cloud" : "Compact fallback";
  var model = data.model && data.model !== "none" ? " · " + data.model : "";
  var setup = data.setup_required ? " · add OPENROUTER_API_KEY for cloud mode" : "";
  var memory = data.memory && data.memory !== "off" ? " · local context" : " · memory off";
  foot.textContent = source + model + memory + setup;
}

function rememberAssistantTurn(role, text) {
  var clean = String(text || "").trim();
  if (!clean) return;
  kfAssistantHistory.push({ role: role, text: clean.slice(0, 900), at: Date.now() });
  if (kfAssistantHistory.length > 12) {
    kfAssistantHistory.splice(0, kfAssistantHistory.length - 12);
  }
  window.__kfAssistantHistory = kfAssistantHistory;
}

function textOf(id) {
  var el = document.getElementById(id);
  return el ? el.textContent.trim() : "";
}

function activeSection() {
  var active = document.querySelector(".section.active");
  return active ? active.id.replace(/^sec-/, "") : "general";
}

function collectAssistantState() {
  var cellDetails = Array.from(document.querySelectorAll("#bms-grid .cell-tile")).map(function (el) {
    var nameEl = el.querySelector(".cell-name");
    var riskEl = el.querySelector(".cell-risk");
    var tempEl = el.querySelector(".cell-temp");
    return {
      cell: el.dataset.cell || (nameEl ? nameEl.textContent.trim() : ""),
      risk: parseFloat(el.dataset.risk || (riskEl ? riskEl.textContent : "")),
      temp_C: parseFloat(el.dataset.temp || (tempEl ? tempEl.textContent : "")),
      fault: el.dataset.fault === "1",
      hot: el.dataset.hot === "1"
    };
  }).filter(function (c) {
    return c.cell && Number.isFinite(c.risk);
  });
  var risks = cellDetails.map(function (c) {
    return c.risk;
  }).filter(Number.isFinite);
  var bmsMeta = window.__kfBms || {};
  var diag = window.__kfDiag || {};
  var mat = window.__kfMaterials || {};
  var rec = window.__kfRecycling || {};
  var byod = window.__kfByod || {};
  var byodPred = byod.predictions || {};
  var byodModels = byodPred.model_outputs || {};
  var decisions = buildDecisionItems();
  return {
    section: activeSection(),
    diagnostics: {
      temperature_C: parseFloat(document.getElementById("temp-slider").value),
      c_rate: parseFloat(document.getElementById("crate-slider").value),
      cycles: parseInt(document.getElementById("cycles-slider").value, 10),
      na: num("diag-na", 1.02),
      mn: num("diag-mn", 0.52),
      fe: num("diag-fe", 0.43),
      eol_capacity: textOf("diag-eol"),
      fade: textOf("diag-fade"),
      voltage_end: diag.voltage ? diag.voltage[diag.voltage.length - 1] : null,
      knee: textOf("diag-knee"),
      rul80: textOf("diag-rul"),
      sei_loss: diag.sei ? diag.sei[diag.sei.length - 1] : null,
      p2_loss: diag.p2 ? diag.p2[diag.p2.length - 1] : null,
      jt_loss: diag.jt ? diag.jt[diag.jt.length - 1] : null,
      rate_loss: diag.rate ? diag.rate[diag.rate.length - 1] : null,
      residual_loss: diag.residual ? diag.residual[diag.residual.length - 1] : null,
      decision: textOf("diag-decision")
    },
    bms: {
      cells: parseInt(document.getElementById("pack-slider").value, 10),
      duration_seconds: parseInt(document.getElementById("dur-slider").value, 10),
      ambient_C: num("bms-ambient", 45),
      seed: bmsMeta.seed || num("bms-seed", 42),
      inject_fault: document.getElementById("sw-fault").checked,
      max_risk: risks.length ? Math.max.apply(null, risks) : null,
      threshold: Number.isFinite(bmsMeta.threshold) ? bmsMeta.threshold : num("bms-risk-thresh", 0.42),
      fault_cell: Number.isFinite(bmsMeta.faultCell) && bmsMeta.faultCell >= 0 ? "C" + bmsMeta.faultCell : "none",
      cell_details: cellDetails,
      decision: textOf("bms-decision")
    },
    materials: {
      na: parseFloat(document.getElementById("na-slider").value),
      mn: parseFloat(document.getElementById("mn-slider").value),
      fe: parseFloat(document.getElementById("fe-slider").value),
      capacity: textOf("mat-cap"),
      voltage: textOf("mat-volt"),
      stability: textOf("mat-stab"),
      jt_index: textOf("mat-jt"),
      fade500_pct: mat.fade500 ? mat.fade500 * 100 : null,
      oxygen_risk: mat.oxygenRisk,
      charge_risk: mat.chargeRisk,
      cost_kwh: mat.costKwh,
      score: mat.score,
      decision: textOf("mat-decision")
    },
    recycling: {
      mass_kg: parseFloat(document.getElementById("bm-slider").value),
      acid_molarity: parseFloat(document.getElementById("acid-slider").value),
      temperature_C: parseFloat(document.getElementById("leach-slider").value),
      recovered_kg: rec.totalRecovered,
      interval_kg: Number.isFinite(rec.lo) ? rec.lo.toFixed(1) + "-" + rec.hi.toFixed(1) : "",
      purity_proxy: rec.purity,
      margin_proxy_inr: rec.marginProxy,
      decision: textOf("recycle-decision")
    },
    upload: {
      filename: byod.filename,
      rows_read: byod.rows_read,
      schema_score: byod.schema ? byod.schema.score : null,
      features_present: byod.feature_mask ? byod.feature_mask.filter(function (x) { return !!x; }).length : 0,
      soh: byodPred.soh,
      confidence: byodPred.confidence,
      cycle_80: byodPred.cycle_80_estimate,
      inference_mode: byodPred.inference_mode,
      checkpoint_status: byod.checkpoint_inference ? byod.checkpoint_inference.status : null,
      model_outputs: {
        M1_CathodeUDE: byodModels.M1_CathodeUDE || null,
        M11_ElectrolyteHealth: byodModels.M11_ElectrolyteHealth || null,
        M13_ChemIdentifier: byodModels.M13_ChemIdentifier || null,
        M14_FormationProtocol: byodModels.M14_FormationProtocol || null
      },
      warnings: byod.warnings || []
    },
    decisions: {
      open_actions: decisions.filter(function (x) { return x.severity === "critical" || x.severity === "warn"; }).length,
      critical_actions: decisions.filter(function (x) { return x.severity === "critical"; }).length,
      latest_run: latestRunLabel(),
      items: decisions.slice(0, 6).map(function (x) {
        return {
          severity: x.severity,
          source: x.source,
          owner: x.owner,
          evidence: x.evidence,
          action: x.action,
          next: x.next,
          confidence: x.confidence
        };
      })
    },
    conversation: (window.__kfAssistantHistory || []).slice(-8)
  };
}

function askAssistant(question) {
  var clean = String(question || "").trim();
  if (!clean) return;
  setAssistantOpen(true);
  assistantMessage("user", clean);
  var pending = assistantMessage("bot loading", "Thinking...");
  var form = document.getElementById("assistant-form");
  var btn = form ? form.querySelector("button") : null;
  if (btn) btn.disabled = true;
  var assistantState = collectAssistantState();
  rememberAssistantTurn("user", clean);

  fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: clean, section: activeSection(), state: assistantState })
  })
    .then(function (res) {
      if (!res.ok) throw new Error("Assistant endpoint returned " + res.status);
      return res.json();
    })
    .then(function (data) {
      var answer = data.answer || "I could not form an answer from the available project context.";
      if (pending) {
        pending.className = "assistant-msg bot";
        pending.innerHTML = renderMd(answer);
        if (data.source === "openrouter" && data.context && data.context.length) {
          var meta = document.createElement("div");
          meta.className = "assistant-meta";
          meta.textContent = "Context: " + data.context.slice(0, 3).join(", ");
          pending.appendChild(meta);
        }
      }
      rememberAssistantTurn("assistant", answer);
      updateAssistantFoot(data);
      if (data.warning) assistantMessage("bot note", data.warning);
    })
    .catch(function (err) {
      if (pending) {
        pending.className = "assistant-msg bot";
        pending.textContent = "The assistant endpoint is offline. Run the FastAPI server and try again.";
      }
      updateAssistantFoot(null);
      showToast("Assistant request failed: " + err.message, "error");
      console.warn(err);
    })
    .finally(function () {
      if (btn) btn.disabled = false;
      var input = document.getElementById("assistant-input");
      if (input) input.focus();
    });
}

function initAssistant() {
  var launch = document.getElementById("assistant-launch");
  var close = document.getElementById("assistant-close");
  var form = document.getElementById("assistant-form");
  var input = document.getElementById("assistant-input");
  if (launch) launch.addEventListener("click", function () {
    var dock = document.getElementById("assistant-dock");
    setAssistantOpen(!(dock && dock.classList.contains("open")));
  });
  if (close) close.addEventListener("click", function () { setAssistantOpen(false); });
  document.querySelectorAll(".assistant-prompts button").forEach(function (btn) {
    btn.addEventListener("click", function () {
      askAssistant(btn.getAttribute("data-q"));
    });
  });
  if (form) {
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var q = input ? input.value : "";
      if (input) input.value = "";
      askAssistant(q);
    });
  }
}

function initArchitecture() {
  var cards = [
    { t: "Universal Differential Equations", d: "Physics ODE terms for SEI, P2-O2, Jahn-Teller coupling, Na desolvation, and rate stress are explicit; residuals are bounded correction terms." },
    { t: "Pack Thermal Graph", d: "Pack monitoring uses topology-aware thermal coupling, EIS drift, and multi-scale lookback windows." },
    { t: "qNEHVI Bayesian Screening", d: "Composition candidates are scored on capacity, fade, life, and cost, then filtered by noisy hypervolume-style Pareto improvement." },
    { t: "Bayesian Recycling Loop", d: "Recovery priors are beta distributions and process conversion follows shrinking-core leaching kinetics." },
    { t: "Uncertainty Propagation", d: "Composition, cell, pack, and recycling predictions carry uncertainty bounds instead of single unqualified numbers." },
    { t: "Evidence Registry", d: "Prediction claims are tied to local datasets, validation gates, and model provenance." },
    { t: "Na-ion Phase Physics", d: "P2-O2 structural transition, Mn3+ Jahn-Teller distortion, and Na+ desolvation are first-class physics terms." },
    { t: "EIS Feature Extraction", d: "Randles-circuit features R_ct, R_SEI, and Warburg coefficient feed the pack risk model." },
    { t: "Regional Climate Model", d: "Operating conditions can be conditioned on local hot-weather temperature and humidity profiles." },
    { t: "BYOD Feature Masking", d: "Uploads produce a feature vector plus availability mask so missing EIS or early-cycle fields are visible instead of silently zero-filled." }
  ];
  var grid = document.getElementById("arch-grid");
  if (grid && !grid.children.length) {
    cards.forEach(function (c) {
      var d = document.createElement("div");
      d.className = "panel";
      d.innerHTML = '<div class="panel-title"><span class="indicator"></span> ' + c.t + '</div><div class="panel-desc">' + c.d + "</div>";
      grid.appendChild(d);
    });
  }
  var models = [
    ["Cathode UDE", "UDE physics plus residual", "Runnable", "371K cycles", "~150K"],
    ["SOH Estimator", "MLP regression", "Runnable", "371K cycles", "~45K"],
    ["Cycle Life", "Classifier", "Runnable", "371K cycles", "~35K"],
    ["Fade Rate", "MLP regression", "Runnable", "371K cycles", "~20K"],
    ["BMS Pack Graph", "Thermal graph risk", "Label Gate", "pack ODE cases", "~80K"],
    ["RUL Predictor", "MLP regression", "Runnable", "371K cycles", "~50K"],
    ["Anomaly AE", "Autoencoder", "Runnable", "371K cycles", "~30K"],
    ["Joint SOH+RUL", "Multi-task MLP", "Runnable", "371K cycles", "~120K"],
    ["Knee Detector", "Conv1D + FC", "Runnable", "371K cycles", "~60K"],
    ["Chem Ranker", "Embedding MLP", "Runnable", "371K cycles", "~15K"],
    ["Electrolyte Health", "EIS diagnostic head", "Runnable", "NASA EIS", "8.8K"],
    ["Replenishability", "Recovery preview", "Research Preview", "capacity windows", "9.6K"],
    ["Chem Identifier", "Early-cycle classifier", "Runnable", "dQ/dV + features", "80.6K"],
    ["Formation Protocol", "Formation quality", "Research Preview", "first cycles", "19.7K"]
  ];
  var tb = document.querySelector("#model-table tbody");
  if (tb && !tb.children.length) {
    models.forEach(function (m) {
      var tr = document.createElement("tr");
      var tagClass = (m[2] === "Label Gate" || m[2] === "Research Preview") ? "warn" : "ok";
      tr.innerHTML = '<td style="font-weight:600">' + m[0] + "</td><td>" + m[1] + '</td><td><span class="tag ' + tagClass + '">' + m[2] + "</span></td><td>" + m[3] + '</td><td style="font-family:JetBrains Mono;font-size:11px">' + m[4] + "</td>";
      tb.appendChild(tr);
    });
  }
}

function modelStatusLabel(status, checkpointPresent) {
  if (!checkpointPresent) return "Missing";
  if (status === "research_preview") return "Preview";
  if (status === "label_gate") return "Label Gate";
  if (status === "trained_checkpoint") return "Trained";
  return String(status || "Unknown").replace(/_/g, " ");
}

function modelStatusClass(label) {
  return /missing|preview|gate/i.test(label) ? "warn" : "ok";
}

function renderModelRegistryFromApi(data) {
  var tb = document.querySelector("#model-table tbody");
  if (!tb || !data || !Array.isArray(data.models)) return;
  tb.innerHTML = "";
  data.models.forEach(function (m) {
    var tr = document.createElement("tr");
    var label = modelStatusLabel(m.status, !!m.checkpoint_present);
    var checkpoint = m.checkpoint_present
      ? (m.checkpoint_source_zip || m.checkpoint_file || "checkpoint present")
      : "not found";
    tr.innerHTML = '<td style="font-weight:600">' + escapeHtml(m.id || m.name || "") +
      "</td><td>" + escapeHtml(m.type || "") +
      '</td><td><span class="tag ' + modelStatusClass(label) + '">' + escapeHtml(label) +
      "</span></td><td>" + escapeHtml(checkpoint) +
      '</td><td style="font-family:JetBrains Mono;font-size:11px">' + escapeHtml(String(m.params || "--")) +
      "</td>";
    tb.appendChild(tr);
  });
}

function renderCheckpointProvenance(data) {
  var body = document.getElementById("checkpoint-provenance-body");
  if (!body) return;
  var models = Array.isArray(data && data.models) ? data.models : [];
  var sourceCounts = {};
  models.forEach(function (m) {
    var src = m.checkpoint_source_zip || (m.checkpoint_present ? "loose checkpoint" : "missing");
    sourceCounts[src] = (sourceCounts[src] || 0) + 1;
  });
  var sources = Array.isArray(data && data.source_zips) ? data.source_zips : [];
  var rows = sources.map(function (src) {
    var name = src.name || "unknown";
    return "<tr><td>" + escapeHtml(name) + "</td><td>" + escapeHtml(String(src.artifacts || sourceCounts[name] || 0)) +
      "</td><td>" + escapeHtml(data.checkpoint_manifest && data.checkpoint_manifest.generated_at ? data.checkpoint_manifest.generated_at : "not generated") + "</td></tr>";
  });
  if (!rows.length && models.length) {
    rows = Object.keys(sourceCounts).sort().map(function (name) {
      return "<tr><td>" + escapeHtml(name) + "</td><td>" + escapeHtml(String(sourceCounts[name])) + "</td><td>runtime scan</td></tr>";
    });
  }
  body.innerHTML = rows.length ? rows.join("") : '<tr><td colspan="3">No checkpoint manifest found. Run scripts/extract_checkpoints.py.</td></tr>';
}

function fetchModelRegistry() {
  fetch("/api/models")
    .then(function (res) {
      if (!res.ok) throw new Error("model registry HTTP " + res.status);
      return res.json();
    })
    .then(function (data) {
      renderModelRegistryFromApi(data);
      renderCheckpointProvenance(data);
    })
    .catch(function (err) {
      renderCheckpointProvenance(null);
      console.warn("Model registry fetch failed", err);
    });
}

function initAPIEndpoints() {
  var eps = [
    { m: "POST", p: "/api/predict/degradation", d: "Na-ion UDE capacity fade prediction with mechanism contributions." },
    { m: "POST", p: "/api/simulate/bms", d: "Thermal graph pack simulation with EIS-informed cell risk." },
    { m: "POST", p: "/api/optimize/recycling", d: "Shrinking-core leaching plus Bayesian recovery priors." },
    { m: "POST", p: "/api/screen/cathode", d: "Composition-property scoring with Pareto candidates." },
    { m: "POST", p: "/api/byod/analyze", d: "Upload cycler data, fingerprint schema, extract tier-1 features, dQ/dV, M1-M14 rules, and checkpoint outputs when available." },
    { m: "POST", p: "/api/byod/analyze-full", d: "Strict upload analysis that requires trained PyTorch checkpoint inference." },
    { m: "POST", p: "/api/byod/compare", d: "Compare two cycler files for A/B cell, protocol, or formation studies." },
    { m: "POST", p: "/api/byod/batch", d: "Analyze a ZIP of cell files and return batch variance plus outlier flags." },
    { m: "POST", p: "/api/byod/webhook/cycle", d: "Cycle-level cycler integration hook returning continue, investigate, or stop." },
    { m: "GET", p: "/api/byod/session/{session_id}", d: "Return a canonical JSON view of an in-memory BYOD session." },
    { m: "GET", p: "/api/byod/session/{session_id}/export-json", d: "Export canonical BYOD JSON for downstream notebooks, QC systems, or archives." },
    { m: "GET", p: "/api/byod/session/{session_id}/export", d: "Export cycle summaries, extracted features, and M1-M14 readouts from an in-memory BYOD session." },
    { m: "POST", p: "/api/chat", d: "OpenRouter assistant with deterministic fallback and browser-supplied recent-turn context." },
    { m: "POST", p: "/predict/lifetime", d: "Compatibility alias for degradation prediction." },
    { m: "POST", p: "/alert/bms", d: "Compatibility alias for BMS alert output." },
    { m: "POST", p: "/optimize/recycling", d: "Compatibility alias for recycling optimization." },
    { m: "POST", p: "/cathode/screen", d: "Compatibility alias for cathode screening." },
    { m: "GET", p: "/api/models", d: "Honest M1-M14 registry with checkpoint presence, status gates, and parameter-count basis." },
    { m: "GET", p: "/health", d: "Server status and timestamp. No auth required." }
  ];
  var ct = document.getElementById("api-endpoints");
  if (!ct || ct.children.length) return;
  eps.forEach(function (ep) {
    var d = document.createElement("div");
    d.className = "switch-row";
    d.style.cursor = "pointer";
    d.onclick = function () { showAPI(ep); };
    d.innerHTML = '<div class="switch-label"><span class="name"><span class="tag" style="margin-right:6px">' + ep.m + "</span> " + ep.p + '</span><span class="desc">' + ep.d + "</span></div>";
    ct.appendChild(d);
  });
}

function showAPI(ep) {
  var examples = {
    "/api/predict/degradation": '{\n  "result": {\n    "capacity_end": 0.912,\n    "fade_pct": 0.088,\n    "mechanisms": {"p2o2": 0.041, "jt": 0.008, "sei_desolv": 0.024}\n  },\n  "provenance": {"model": "Na-ion UDE physics mirror"}\n}',
    "/api/simulate/bms": '{\n  "cells": 8,\n  "thermal_equation": "Cth dT/dt = q + sum(kij(Tj-Ti)) - h(T-Ta)",\n  "max_risk": 0.64,\n  "alerts": [{"t": 74, "cell": 3, "risk": 0.51}]\n}',
    "/api/optimize/recycling": '{\n  "recoveries": {"Mn": {"recovery_rate": 0.91}, "Fe": {"recovery_rate": 0.78}, "Na": {"recovery_rate": 0.82}},\n  "kinetics": "shrinking-core leaching",\n  "uncertainty": {"basis": "Monte Carlo feedstock assay"}\n}',
    "/api/byod/analyze": '{\n  "session_id": "uuid",\n  "schema": {"format": "neware", "score": 0.82},\n  "features": {"early_coulombic_efficiency": 0.992},\n  "feature_mask": [1,1,1],\n  "predictions": {"soh": 0.94, "confidence": 0.71, "inference_mode": "checkpoint_plus_rules"}\n}',
    "/api/byod/analyze-full": '{\n  "session_id": "uuid",\n  "checkpoint_inference": {"status": "ok"},\n  "predictions": {"inference_mode": "checkpoint_plus_rules"}\n}',
    "/api/byod/compare": '{\n  "comparison": {"delta": {"soh": 0.012}, "decision": "file_b looks healthier on SOH"},\n  "file_a_session_id": "uuid",\n  "file_b_session_id": "uuid"\n}',
    "/api/byod/batch": '{\n  "files_analyzed": 12,\n  "stats": {"soh_mean": 0.941, "soh_std": 0.018},\n  "outliers": [{"filename": "cell_07.csv", "reasons": ["SOH outside 2 sigma"]}]\n}',
    "/api/byod/webhook/cycle": '{\n  "status": "accepted",\n  "result": {"cell_id": "A17", "cycle": 42, "recommendation": "continue", "soh": 0.943}\n}',
    "/api/byod/session/{session_id}": '{\n  "format": "kineticsforge_canonical_v1",\n  "features": {"capacity_fade_rate_ah_per_cycle": -0.00014},\n  "predictions": {"inference_mode": "checkpoint_plus_rules"}\n}',
    "/api/byod/session/{session_id}/export-json": '{\n  "format": "kineticsforge_canonical_v1",\n  "cycle_summary": [],\n  "checkpoint_inference": {"status": "ok"}\n}',
    "/api/models": '{\n  "total": 14,\n  "checkpoint_manifest": {"files": 57, "generated_at": "..."},\n  "models": [{"id": "M11_ElectrolyteHealth", "checkpoint_present": true, "checkpoint_source_zip": "results (25).zip"}]\n}',
    "/api/chat": '{\n  "answer": "C6 is highlighted because its risk crossed the action threshold...",\n  "source": "openrouter",\n  "memory": "browser_recent_turns"\n}'
  };
  var r = examples[ep.p] || '{"status":"ok","endpoint":"' + ep.p + '"}';
  var c = document.getElementById("api-response");
  c.innerHTML = '<div class="cmd">$ curl -X ' + ep.m + ' http://localhost:8000' + ep.p + '</div>\n<div class="ok">' + escapeHtml(r).replace(/\n/g, "<br>").replace(/ /g, "&nbsp;") + "</div>";
}

// ── PDF Report Generator ────────────────────────────────────────────────
function generateReport() {
  var active = document.querySelector(".nav-links a.active");
  var panelName = active ? active.textContent.trim() : "Diagnostics";
  var now = new Date();
  var dateStr = now.toISOString().slice(0, 10) + " " + now.toTimeString().slice(0, 5);
  // Collect all visible canvases in the active section
  var sections = document.querySelectorAll(".section");
  var activeSection = null;
  sections.forEach(function (s) {
    if (s.style.display !== "none" && s.offsetParent !== null) activeSection = s;
  });
  if (!activeSection) {
    // Fallback: find by panel name
    var sectionMap = { "DIAGNOSTICS": "sec-diagnostics", "UPLOAD": "sec-upload", "BMS": "sec-bms", "MATERIALS": "sec-materials", "RECYCLING": "sec-recycling" };
    activeSection = document.getElementById(sectionMap[panelName.toUpperCase()] || "sec-diagnostics");
  }
  // Build a report in a new window
  var win = window.open("", "_blank", "width=800,height=1100");
  if (!win) { showToast("Pop-up blocked. Allow pop-ups for report generation.", "warn"); return; }
  var html = '<!DOCTYPE html><html><head><title>KineticsForge Report - ' + panelName + '</title>';
  html += '<style>';
  html += 'body{font-family:Inter,system-ui,sans-serif;color:#222;margin:2rem;line-height:1.6;}';
  html += 'h1{font-size:1.4rem;border-bottom:2px solid #cc0000;padding-bottom:0.5rem;margin-bottom:0.3rem;}';
  html += '.meta{color:#666;font-size:0.78rem;margin-bottom:1.5rem;}';
  html += '.metric{display:inline-block;text-align:center;margin:0.5rem 1rem 0.5rem 0;padding:0.7rem 1.2rem;border:1px solid #ddd;border-radius:6px;min-width:100px;}';
  html += '.metric .val{font-size:1.3rem;font-weight:700;color:#cc0000;font-family:monospace;}';
  html += '.metric .lbl{font-size:0.65rem;color:#888;text-transform:uppercase;margin-top:2px;}';
  html += '.decision{border-left:3px solid #cc0000;background:#fff5f5;padding:0.8rem;margin:1rem 0;font-size:0.82rem;}';
  html += 'img{max-width:100%;margin:0.8rem 0;border:1px solid #eee;border-radius:4px;}';
  html += '.footer{margin-top:2rem;padding-top:1rem;border-top:1px solid #ddd;font-size:0.68rem;color:#999;}';
  html += '.confidence-bar{height:8px;background:#eee;border-radius:4px;margin:4px 0 8px;overflow:hidden;}';
  html += '.confidence-fill{height:100%;border-radius:4px;}';
  html += '@media print{body{margin:1cm;}@page{margin:1.5cm;}}';
  html += '</style></head><body>';
  html += '<h1>KineticsForge — ' + escapeHtml(panelName) + ' Report</h1>';
  html += '<div class="meta">Generated: ' + dateStr + ' | Platform: KineticsForge v1.0 | Mode: simulation-backed</div>';
  // Metrics
  if (activeSection) {
    var stats = activeSection.querySelectorAll(".stat");
    if (stats.length) {
      stats.forEach(function (s) {
        var val = s.querySelector(".val");
        var lbl = s.querySelector(".lbl");
        if (val && lbl) {
          html += '<div class="metric"><div class="val">' + escapeHtml(val.textContent) + '</div><div class="lbl">' + escapeHtml(lbl.textContent) + '</div></div>';
        }
      });
      html += '<br>';
    }
    // Readout grids
    var readouts = activeSection.querySelectorAll(".readout");
    if (readouts.length) {
      readouts.forEach(function (r) {
        var k = r.querySelector(".k");
        var v = r.querySelector(".v");
        if (k && v) html += '<div class="metric"><div class="val">' + escapeHtml(v.textContent) + '</div><div class="lbl">' + escapeHtml(k.textContent) + '</div></div>';
      });
      html += '<br>';
    }
    // Decision box
    var decision = activeSection.querySelector(".decision-box");
    if (decision && decision.textContent.trim()) {
      html += '<div class="decision">' + decision.innerHTML + '</div>';
    }
    // Canvases → inline images
    var canvases = activeSection.querySelectorAll("canvas");
    canvases.forEach(function (cv) {
      try {
        var dataUrl = cv.toDataURL("image/png");
        if (dataUrl && dataUrl.length > 100) {
          html += '<img src="' + dataUrl + '" alt="Chart">';
        }
      } catch (e) { /* tainted canvas, skip */ }
    });
  }
  html += '<div class="footer">KineticsForge — Simulation-backed, real-data indexed, validation-gated.<br>All predictions carry uncertainty bounds and provenance metadata. Research-preview outputs are not certification.</div>';
  html += '</body></html>';
  win.document.write(html);
  win.document.close();
  setTimeout(function () { win.print(); }, 600);
}

// ── Confidence/Uncertainty Display ──────────────────────────────────────
function renderConfidence(containerId, confidence, detail) {
  var el = document.getElementById(containerId);
  if (!el) return;
  if (confidence == null || !Number.isFinite(confidence)) {
    el.innerHTML = "";
    return;
  }
  var pct = Math.round(clamp(confidence, 0, 1) * 100);
  var color = pct >= 75 ? "#22c55e" : pct >= 50 ? "#ff9f1a" : "#ff1a1a";
  var label = pct >= 75 ? "High confidence" : pct >= 50 ? "Moderate confidence" : "Low confidence";
  var html = '<div style="margin-top:0.5rem;padding:0.5rem 0.65rem;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:6px">';
  html += '<div style="display:flex;justify-content:space-between;align-items:center;font-size:0.7rem">';
  html += '<span style="color:#888">Model Confidence</span>';
  html += '<span style="color:' + color + ';font-weight:700;font-family:JetBrains Mono,monospace">' + pct + '% — ' + label + '</span>';
  html += '</div>';
  html += '<div style="height:6px;background:#1a1a1a;border-radius:3px;margin-top:5px;overflow:hidden">';
  html += '<div style="height:100%;width:' + pct + '%;background:' + color + ';border-radius:3px;transition:width 0.6s ease"></div>';
  html += '</div>';
  if (detail) {
    html += '<div style="font-size:0.64rem;color:#666;margin-top:4px">' + escapeHtml(detail) + '</div>';
  }
  html += '</div>';
  el.innerHTML = html;
}

function outsidePenalty(value, lo, hi, maxPenalty) {
  if (!Number.isFinite(value)) return maxPenalty * 0.5;
  if (value >= lo && value <= hi) return 0;
  var dist = value < lo ? (lo - value) : (value - hi);
  var span = Math.max(hi - lo, 1e-9);
  return clamp((dist / span) * maxPenalty, 0, maxPenalty);
}

function computeDiagConfidence(out) {
  if (!out || !out.cap || !out.cap.length) {
    return { confidence: 0.25, detail: "No diagnostic curve is available yet." };
  }
  var sei = Number(out.sei && out.sei[out.sei.length - 1]) || 0;
  var p2 = Number(out.p2 && out.p2[out.p2.length - 1]) || 0;
  var jt = Number(out.jt && out.jt[out.jt.length - 1]) || 0;
  var rate = Number(out.rate && out.rate[out.rate.length - 1]) || 0;
  var residual = Number(out.residual && out.residual[out.residual.length - 1]) || 0;
  var total = Math.max(1e-9, sei + p2 + jt + rate + residual);
  var residualFrac = clamp(residual / total, 0, 1);
  var topShare = Math.max(sei, p2, jt, rate, residual) / total;
  var fade = 1 - (Number(out.cap[out.cap.length - 1]) || 1);
  var bandWidth = 0.06;
  if (out.band && out.band.lo && out.band.hi && out.band.lo.length && out.band.hi.length) {
    var lo = Number(out.band.lo[out.band.lo.length - 1]);
    var hi = Number(out.band.hi[out.band.hi.length - 1]);
    if (Number.isFinite(lo) && Number.isFinite(hi)) bandWidth = Math.max(0, hi - lo);
  }
  var tempC = num("temp-slider", 45);
  var cRate = num("crate-slider", 1.0);
  var inputPenalty = outsidePenalty(tempC, 20, 55, 0.12) + outsidePenalty(cRate, 0.2, 2.0, 0.10);
  var conf = 0.78;
  conf -= residualFrac * 0.40;
  conf -= clamp((bandWidth - 0.02) / 0.20, 0, 0.22);
  conf -= fade < 0.02 ? 0.08 : 0;
  conf -= inputPenalty;
  conf += clamp((topShare - 0.22) * 0.14, 0, 0.10);
  conf = clamp(conf, 0.18, 0.90);
  var detail = "Residual share " + Math.round(residualFrac * 100) + "%, end-band width " + (bandWidth * 100).toFixed(1) + "%, stress penalty " + Math.round(inputPenalty * 100) + " pts.";
  return { confidence: conf, detail: detail };
}

function computeBmsConfidence(bms, frame) {
  if (!bms || !frame || !Array.isArray(frame.risks) || !frame.risks.length) {
    return { confidence: 0.25, detail: "No BMS frame is available yet." };
  }
  var gate = Number(bms.threshold);
  if (!Number.isFinite(gate)) gate = 0.42;
  var maxRisk = Number(frame.maxRisk);
  if (!Number.isFinite(maxRisk)) maxRisk = Math.max.apply(null, frame.risks);
  var distance = Math.abs(maxRisk - gate);
  var distanceScore = clamp(distance / 0.22, 0, 1);
  var spreadScore = clamp(stddev(frame.risks) / 0.16, 0, 1);
  var tempSpread = clamp(stddev(frame.temps || []) / 8.0, 0, 1);
  var slopePeak = Math.max.apply(null, (frame.slopes || [0]).map(function (x) { return Math.abs(Number(x) || 0); }));
  var slopeScore = clamp(slopePeak / 0.06, 0, 1);
  var ambient = Number(bms.ambient_C);
  var ambientPenalty = outsidePenalty(ambient, 20, 55, 0.12);
  var nearGatePenalty = distance < 0.035 ? 0.10 : 0;
  var conf = 0.42 + 0.24 * distanceScore + 0.13 * spreadScore + 0.10 * tempSpread + 0.08 * slopeScore - ambientPenalty - nearGatePenalty;
  var sweep = window.__kfBmsSweep;
  var sweepNote = "";
  if (sweep && sweep.scenarioKey && bms.scenarioKey && sweep.scenarioKey === bms.scenarioKey) {
    var sweepStd = Number(sweep.stdRisk);
    var alertRate = Number(sweep.alertRate);
    if (Number.isFinite(sweepStd)) conf -= clamp(sweepStd / 0.16, 0, 0.12);
    if (Number.isFinite(alertRate)) conf -= clamp(Math.abs(alertRate - 0.5) * 0.16, 0, 0.08);
    sweepNote = " Sweep alert rate " + (Number.isFinite(alertRate) ? (alertRate * 100).toFixed(0) : "--") + "%, sigma " + (Number.isFinite(sweepStd) ? sweepStd.toFixed(3) : "--") + ".";
  }
  conf = clamp(conf, 0.20, 0.88);
  var detail = "|risk-gate|=" + distance.toFixed(3) + ", pack risk std=" + stddev(frame.risks).toFixed(3) + ", Tmax spread " + stddev(frame.temps || []).toFixed(1) + " C." + sweepNote;
  return { confidence: conf, detail: detail };
}

function computeMaterialsConfidence(mat) {
  if (!mat) return { confidence: 0.25, detail: "No composition has been scored yet." };
  var stab = clamp(Number(mat.stability) || 0, 0, 1);
  var fade500 = clamp(Number(mat.fade500) || 0.5, 0, 1);
  var oxygenRisk = clamp(Number(mat.oxygenRisk) || 0.5, 0, 1);
  var chargeRisk = clamp(Number(mat.chargeRisk) || 0.5, 0, 1);
  var q0 = Number(mat.Q0) || 0;
  var v = Number(mat.avgVoltage) || 0;
  var comp = mat.comp || {};
  var compPenalty = outsidePenalty(Number(comp.Na), 0.85, 1.12, 0.10)
    + outsidePenalty(Number(comp.Mn), 0.18, 0.82, 0.08)
    + outsidePenalty(Number(comp.Fe), 0.12, 0.82, 0.08);
  var propertyPenalty = outsidePenalty(q0, 95, 210, 0.10) + outsidePenalty(v, 2.8, 4.3, 0.08);
  var quality = 0.36 * stab + 0.20 * (1 - fade500) + 0.22 * (1 - oxygenRisk) + 0.22 * (1 - chargeRisk);
  var conf = 0.30 + 0.52 * quality - compPenalty - propertyPenalty;
  conf = clamp(conf, 0.18, 0.86);
  var detail = "Stability " + stab.toFixed(2) + ", fade500 " + (fade500 * 100).toFixed(1) + "%, oxygen risk " + oxygenRisk.toFixed(2) + ", charge risk " + chargeRisk.toFixed(2) + ".";
  return { confidence: conf, detail: detail };
}

function computeRecyclingConfidence(rec) {
  if (!rec || !Number.isFinite(rec.totalRecovered)) {
    return { confidence: 0.25, detail: "No recycling run is available yet." };
  }
  var total = Math.max(1e-9, Number(rec.totalRecovered));
  var lo = Number(rec.lo);
  var hi = Number(rec.hi);
  var widthRatio = (Number.isFinite(lo) && Number.isFinite(hi)) ? Math.max(0, hi - lo) / total : 0.30;
  var purity = clamp(Number(rec.purity) || 0, 0, 1);
  var margin = Number(rec.marginProxy) || 0;
  var mn = Array.isArray(rec.targets) && rec.targets.length ? clamp(Number(rec.targets[0]) || 0, 0, 1) : 0;
  var acid = num("acid-slider", 2.0);
  var temp = num("leach-slider", 80);
  var processPenalty = outsidePenalty(acid, 1.0, 3.2, 0.08) + outsidePenalty(temp, 55, 92, 0.08);
  var mcEnabled = !!(document.getElementById("sw-mc") && document.getElementById("sw-mc").checked);
  var conf = 0.60 - 0.42 * clamp(widthRatio, 0, 0.8) + 0.12 * purity + 0.08 * mn - processPenalty - (mcEnabled ? 0 : 0.06);
  if (margin < 0) conf -= 0.03;
  conf = clamp(conf, 0.20, 0.85);
  var detail = "Recovery interval width " + (widthRatio * 100).toFixed(1) + "% of mean, purity " + (purity * 100).toFixed(1) + "%, Mn " + (mn * 100).toFixed(1) + "%.";
  return { confidence: conf, detail: detail };
}

function computeUploadConfidence(byod) {
  if (!byod || !byod.predictions) return { confidence: 0.25, detail: "No upload prediction is available yet." };
  var pred = byod.predictions || {};
  var modelConf = clamp(Number(pred.confidence) || 0, 0, 1);
  var mask = Array.isArray(byod.feature_mask) ? byod.feature_mask : [];
  var names = Array.isArray(byod.feature_names) ? byod.feature_names : [];
  var availability = names.length ? clamp(mask.filter(Boolean).length / names.length, 0, 1) : 0;
  var schemaScore = clamp(Number((byod.schema || {}).score) || 0, 0, 1);
  var warnings = Array.isArray(byod.warnings) ? byod.warnings.length : 0;
  var warningPenalty = clamp(warnings * 0.06, 0, 0.30);
  var mode = String(pred.inference_mode || "rules_only");
  var modeBonus = mode === "checkpoint_plus_rules" ? 0.06 : 0;
  var conf = 0.48 * modelConf + 0.24 * availability + 0.18 * schemaScore + 0.10 * (1 - warningPenalty) + modeBonus;
  conf -= warningPenalty;
  conf = clamp(conf, 0.15, 0.90);
  var detail = "Model " + Math.round(modelConf * 100) + "%, feature coverage " + Math.round(availability * 100) + "%, schema " + Math.round(schemaScore * 100) + "%, warnings " + warnings + ".";
  return { confidence: conf, detail: detail };
}

window.addEventListener("DOMContentLoaded", function () {
  setTimeout(function () {
    animC(document.getElementById("counter-data"), 5091293);
    animC(document.getElementById("counter-models"), 14);
    animC(document.getElementById("counter-cells"), 555);
    animC(document.getElementById("counter-endpoints"), 14);
  }, 300);
  initArchitecture();
  fetchModelRegistry();
  initAPIEndpoints();
  initAssistant();
  updateDiag();
  updateBMS();
  updateMat();
  updateRecycling();
  renderDecisionConsole();
});
