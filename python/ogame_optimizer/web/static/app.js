// OGame Fleet Optimizer - app.js v20260702e
(function() {
  let lastResult = null;
  let lastRequest = null;
  let refineCount = 0;
  let useSeedFleet = false;
  const HISTORY_KEY = "ogame_optimizer_history";
  const HISTORY_MAX = 6;
  let historyList = []; // array of saved result snapshots
  let activeTab = "counter"; // "counter" or "myfleet"

  const SHIP_KEYS = ["light_fighter","heavy_fighter","cruiser","battleship","battlecruiser","bomber","destroyer","deathstar","small_cargo","large_cargo","espionage_probe","pathfinder","recycler","reaper"];
  const DEFENSE_KEYS = ["rocket_launcher","light_laser","heavy_laser","gauss_cannon","ion_cannon","plasma_turret","small_shield_dome","large_shield_dome"];

  // Ship metadata: speed (base), fuel (deut/hour), shield (base)
  const SHIP_META = {
    "light_fighter":   {speed: 12500,  fuel: 20,   shield: 10,    cost: 4000},
    "heavy_fighter":   {speed: 10000,  fuel: 75,   shield: 25,    cost: 10000},
    "cruiser":         {speed: 15000,  fuel: 300,  shield: 50,    cost: 29000},
    "battleship":      {speed: 10000,  fuel: 500,  shield: 200,   cost: 60000},
    "battlecruiser":   {speed: 10000,  fuel: 250,  shield: 400,   cost: 85000},
    "bomber":          {speed: 4000,   fuel: 1000, shield: 500,   cost: 90000},
    "destroyer":       {speed: 5000,   fuel: 1000, shield: 500,   cost: 125000},
    "deathstar":       {speed: 100,    fuel: 1,    shield: 50000, cost: 10000000},
    "small_cargo":     {speed: 5000,   fuel: 10,   shield: 10,    cost: 4000},
    "large_cargo":     {speed: 7500,   fuel: 50,   shield: 25,    cost: 12000},
    "espionage_probe": {speed: 100000000, fuel: 1, shield: 0,     cost: 1000},
    "pathfinder":      {speed: 10000,  fuel: 50,   shield: 100,   cost: 22000},
    "recycler":        {speed: 2000,   fuel: 300,  shield: 10,    cost: 18000},
    "reaper":          {speed: 7000,   fuel: 100,  shield: 1000,  cost: 140000},
  };
  const DEFENSE_COST = {
    "rocket_launcher": 2000, "light_laser": 2000, "heavy_laser": 8000,
    "gauss_cannon": 37000, "ion_cannon": 8000, "plasma_turret": 130000,
    "small_shield_dome": 20000, "large_shield_dome": 100000,
  };
  function shipCost(key) {
    var m = SHIP_META[key];
    return m ? m.cost : (DEFENSE_COST[key] || 0);
  }
  function isSlowOrExpensive(key) {
    var m = SHIP_META[key];
    if (!m) return false;
    return m.speed < 6000 || m.fuel >= 500;
  }

  const form = document.getElementById("optimize-form");
  const btn = document.getElementById("optimize-btn");
  const spinner = document.getElementById("spinner");
  const results = document.getElementById("results");
  const errorBox = document.getElementById("error");
  const metrics = document.getElementById("metrics");
  const tbody = document.querySelector("#fleet-table tbody");
  const refineBtn = document.getElementById("refine-btn");

  function readNonZero(formData, keys) {
    const out = {};
    for (const k of keys) {
      const v = parseInt(formData.get(k) || "0", 10);
      if (v > 0) out[k] = v;
    }
    return out;
  }
  function readMyFleet() {
    var fleet = {};
    var inputs = document.querySelectorAll('input[name^="my_"]');
    for (var i = 0; i < inputs.length; i++) {
      var v = parseInt(inputs[i].value || "0", 10);
      if (v > 0) {
        var key = inputs[i].name.replace(/^my_/, "");
        fleet[key] = v;
      }
    }
    return fleet;
  }
  function fmtNum(n) {
    if (n === null || n === undefined) return "-";
    return Math.round(n).toLocaleString();
  }
  function fmtPct(p) { return (p * 100).toFixed(1) + "%"; }
  function winColorClass(p) {
    if (p >= 0.95) return "win-green";
    if (p >= 0.80) return "win-yellow";
    return "win-red";
  }
  function getExcludedShips() {
    const excluded = [];
    document.querySelectorAll('input[name="exclude"]:checked').forEach(function(cb) { excluded.push(cb.value); });
    return excluded.length > 0 ? excluded : null;
  }

  form.addEventListener("submit", async function(e) {
    e.preventDefault();
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
    results.classList.add("hidden");
    btn.disabled = true;
    if (refineBtn) refineBtn.disabled = true;
    spinner.classList.remove("hidden");

    try {
      const fd = new FormData(form);
      const modeEl = document.querySelector('input[name="mode"]:checked');
      const mode = modeEl ? modeEl.value : "attack";
      var gaTimeEl = document.getElementById("ga_time");
      var finalSimsEl = document.getElementById("final_sims");
      var payload = {
        enemy_fleet: { ships: readNonZero(fd, SHIP_KEYS) },
        enemy_defenses: { defenses: readNonZero(fd, DEFENSE_KEYS) },
        attacker_tech: { weapon: parseInt(fd.get("attacker_weapon")||"0"), shield: parseInt(fd.get("attacker_shield")||"0"), armor: parseInt(fd.get("attacker_armor")||"0") },
        defender_tech: { weapon: parseInt(fd.get("defender_weapon")||"0"), shield: parseInt(fd.get("defender_shield")||"0"), armor: parseInt(fd.get("defender_armor")||"0") },
        budget_multiplier: parseFloat(fd.get("budget_multiplier") || "1.0"),
        mode: mode,
        seed: fd.get("seed") ? parseInt(fd.get("seed")) : 42,
        ga_time_budget: parseFloat((gaTimeEl||{value:"5"}).value || "5"),
        final_sims: parseInt((finalSimsEl||{value:"500"}).value || "500"),
        exclude_ships: getExcludedShips(),
        seed_fleet: (useSeedFleet && lastResult) ? lastResult.recommended_fleet : null,
        debris_pct: parseFloat((document.getElementById('debris_pct')||{value:'0.30'}).value || '0.30'),
        deuterium_in_debris: document.getElementById('deut_in_debris') ? document.getElementById('deut_in_debris').checked : false,
        optimization_target: (document.getElementById('optimization_target')||{value:'maximize_profit'}).value || 'maximize_profit',
        min_gain_pct: parseFloat((document.getElementById('min_gain_pct')||{value:'0'}).value || '0'),
        hyperspace_tech: parseInt((document.getElementById('hyperspace_tech')||{value:'0'}).value || '0'),
        collector_class: document.getElementById('collector_class') ? document.getElementById('collector_class').checked : false,
        resource_weights: [
          parseFloat((document.getElementById('weight_m')||{value:'2.0'}).value || '2.0'),
          parseFloat((document.getElementById('weight_c')||{value:'1.0'}).value || '1.0'),
          parseFloat((document.getElementById('weight_d')||{value:'1.0'}).value || '1.0'),
        ],
        preference_beta: parseFloat((document.getElementById('preference_beta')||{value:'0.05'}).value || '0.05'),
      };

      // Check: 0.0x requires base fleet
      if (payload.budget_multiplier === 0 && activeTab !== 'myfleet') {
        throw new Error('0.0X requires "Start from My Fleet" tab. Switch tabs and enter your fleet.');
      }
      var myFleetCheck = readMyFleet();
      if (payload.budget_multiplier === 0 && activeTab === 'myfleet' && Object.keys(myFleetCheck).length === 0) {
        throw new Error('0.0X requires ships in the My Fleet section. Enter your fleet first.');
      }
      // Send base_fleet when on myfleet tab (not just 0.0x)
      if (activeTab === 'myfleet' && Object.keys(myFleetCheck).length > 0) {
        payload.base_fleet = myFleetCheck;
        console.log('DIAG SET payload.base_fleet with', Object.keys(myFleetCheck).length, 'ships');
      }
      console.log('POST /api/optimize payload:', payload);
      console.log("DIAG activeTab:", activeTab);
      console.log("DIAG base_fleet in payload:", payload.base_fleet);
      console.log('DIAG base_fleet ships:', payload.base_fleet ? Object.keys(payload.base_fleet).length : 0);
      var dbg2 = document.getElementById('debug-badge');
      if (dbg2) dbg2.textContent = 'TAB: ' + activeTab + ' | base_fleet: ' + (payload.base_fleet ? Object.keys(payload.base_fleet).length + ' ships WILL be sent' : 'NOT SET');
      lastRequest = payload;
      var resp = await fetch("/api/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        var err = await resp.json().catch(function() { return {detail: resp.statusText}; });
        var msg = err.detail || "Optimization failed";
        if (Array.isArray(err.detail)) {
          msg = err.detail.map(function(e) { return e.loc.join(".") + ": " + e.msg; }).join("; ");
        }
        throw new Error(msg);
      }

      var data = await resp.json();
      renderResults(data);
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.classList.remove("hidden");
      results.classList.remove("hidden");
    } finally {
      btn.disabled = false;
      spinner.classList.add("hidden");
      useSeedFleet = false;
    }
  });

  function renderResults(data) {
    metrics.innerHTML = "";
    var wp = data.win_probability;
    var ci = data.confidence_interval_95 || [0, 0];
    var rawLoss = data.raw_loss_mean || data.expected_loss_mean || 0;
    var effLoss = data.expected_loss_mean || 0;
    var rawPct = data.fleet_value > 0 ? (rawLoss / data.fleet_value) * 100 : 0;
    var effPct = data.fleet_lost_pct || (data.fleet_value > 0 ? (effLoss / data.fleet_value) * 100 : 0);
    var lostClass = rawPct < 5 ? "win-green" : (rawPct < 50 ? "win-yellow" : "win-red");
    var cards = [
      ["Win Probability", fmtPct(wp), winColorClass(wp)],
      ["Ships Lost", (data.ships_lost_count != null ? fmtNum(data.ships_lost_count) + " / " + fmtNum(data.ships_initial_count || 0) : fmtPct(rawPct/100)), lostClass],
      ["Fleet Value", fmtNum(data.fleet_value)],
      ["Ships Lost (raw)", fmtNum(rawLoss) + " (" + rawPct.toFixed(2) + "% of fleet value)"],
      ["Debris Metal", fmtNum(data.debris_metal)],
      ["Debris Crystal", fmtNum(data.debris_crystal)],
    ];
    if (data.debris_deuterium > 0) cards.push(["Debris Deuterium", fmtNum(data.debris_deuterium)]);
    if (data.debris_total > 0) cards.push(["Debris Total", fmtNum(data.debris_total), "win-green"]);
    var netP = data.net_profit || 0;
    var netPct = data.net_profit_pct || 0;
    var netClass = netP > 0 ? "win-green" : "win-red";
    var netSign = netP >= 0 ? "+" : "";
    var netLabel = netP >= 0 ? "Net Profit (if recycled)" : "NET LOSS (if recycled)";
    cards.push([netLabel, netSign + fmtNum(netP) + " (" + netSign + netPct.toFixed(1) + "%)", netClass]);
    if (data.min_gain_required != null && data.min_gain_required > 0) {
      var mgMet = data.min_gain_met;
      var mgRoi = data.actual_roi_pct || 0;
      var mgClass = mgMet ? "win-green" : "win-red";
      var mgLabel = mgMet ? "Min Gain Met" : "Min Gain NOT Met";
      cards.push([
        mgLabel + " (required: " + data.min_gain_required.toFixed(0) + "%)",
        "Actual ROI: " + (mgRoi >= 0 ? "+" : "") + mgRoi.toFixed(1) + "%",
        mgClass
      ]);
    }
    if (data.recyclers_needed > 0) {
      var rcT = data.recyclers_cost_total || 0;
      var recLabel = fmtNum(data.recyclers_needed) + " (cap " + fmtNum(data.recycler_capacity) + " each)";
      if (rcT > 0) {
        recLabel += "<br><span style='font-size:0.8em;color:#8b95a7'>Build: " + fmtNum(data.recyclers_cost_metal) + " M / " + fmtNum(data.recyclers_cost_crystal) + " C / " + fmtNum(data.recyclers_cost_deuterium) + " D</span>";
      }
      cards.push(["Recyclers Needed", recLabel]);
    }
    // base_fleet mode: show base + additions breakdown
    if (data.base_fleet && Object.keys(data.base_fleet).length > 0) {
      var addCost = data.fleet_value - (data.base_fleet_cost || 0);
      cards.push(["Existing Fleet", fmtNum(data.base_fleet_count || 0) + " ships (" + fmtNum(data.base_fleet_cost || 0) + " res)"]);
      cards.push(["Additions Cost", fmtNum(addCost) + " res"]);
    }
    cards.push(["Loss Stddev", fmtNum(data.expected_loss_stddev)]);
    cards.push(["95% CI", fmtNum(ci[0]) + " - " + fmtNum(ci[1])]);
    cards.push(["GA Improvement", fmtNum(data.ga_improvement_pct) + "%"]);
    if (data.resource_weights && data.resource_preference_pct != null) {
      var rw = data.resource_weights;
      var prefPct = data.resource_preference_pct;
      var prefSign = prefPct > 0.05 ? "+" : (prefPct < -0.05 ? "" : "");
      var prefClass = prefPct > 0.5 ? "win-green" : (prefPct < -0.5 ? "win-red" : "");
      var alphaStr = (data.preference_beta != null) ? (" @ β=" + data.preference_beta.toFixed(2)) : "";
      cards.push([
        "Resource Pref. (M:C:D " + rw[0].toFixed(1) + ":" + rw[1].toFixed(1) + ":" + rw[2].toFixed(1) + ")" + alphaStr,
        prefSign + prefPct.toFixed(2) + "% applied to loss",
        prefClass
      ]);
    }
    cards.push(["Simulations", data.sims_run_final]);
    cards.push(["Total Time", (data.time_elapsed_total || 0).toFixed(2) + "s"]);
    cards.push(["Seed", data.seed_used]);
    for (var i = 0; i < cards.length; i++) {
      var c = cards[i];
      var div = document.createElement("div");
      div.className = "metric" + (c[2] ? " " + c[2] : "");
      div.innerHTML = "<div class=label>" + c[0] + "</div><div class=value>" + c[1] + "</div>";
      metrics.appendChild(div);
    }

    tbody.innerHTML = "";
    var fleet = data.recommended_fleet || {};
    var keys = Object.keys(fleet).sort();
    var analysis = data.fleet_analysis || {};
    var hasAnalysis = Object.keys(analysis).length > 0;

    // Build fleet rows, attaching impact data when available
    var fleetRows = [];
    for (var j = 0; j < keys.length; j++) {
      var k = keys[j];
      var info = analysis[k] || {};
      fleetRows.push({
        key: k,
        count: fleet[k],
        impact_pct: info.impact_pct != null ? info.impact_pct : null,
        tag: info.tag || "",
        survival_pct: info.survival_pct != null ? info.survival_pct : null
      });
    }
    // Sort by impact descending (critical first, dead_weight last).
    // Ships without impact analysis sort to the bottom, preserving name order.
    if (hasAnalysis) {
      fleetRows.sort(function(a, b) {
        var av = a.impact_pct == null ? -Infinity : a.impact_pct;
        var bv = b.impact_pct == null ? -Infinity : b.impact_pct;
        if (bv !== av) return bv - av;
        return a.key < b.key ? -1 : (a.key > b.key ? 1 : 0);
      });
    }

    // Pre-compute fleet cost for percentage calculation
    var totalFleetCost = 0;
    var costPctMap = {};
    for (var pc = 0; pc < fleetRows.length; pc++) {
      var pcFr = fleetRows[pc];
      var pcCost = shipCost(pcFr.key);
      if (pcCost > 0) {
        costPctMap[pcFr.key] = pcCost * pcFr.count;
        totalFleetCost += pcCost * pcFr.count;
      }
    }
    for (var j = 0; j < fleetRows.length; j++) {
      var fr = fleetRows[j];
      var rowClass = "";
      if (hasAnalysis) {
        if (fr.tag === "critical") rowClass = "fleet-row-critical";
        else if (fr.tag === "important") rowClass = "fleet-row-important";
        else if (fr.tag === "dead_weight") rowClass = "fleet-row-deadweight";
        else if (fr.tag === "fodder") rowClass = "fleet-row-fodder";
      }
      var row = document.createElement("tr");
      if (rowClass) row.className = rowClass;
      var breakdownText = "";
      if (hasAnalysis && fr.loss_breakdown) {
        var parts = [];
        for (var bt in fr.loss_breakdown) {
          var d = fr.loss_breakdown[bt];
          if (Math.abs(d) >= 1) {
            parts.push((d > 0 ? "+" : "") + fmtNum(d) + " " + bt.replace(/_/g, " "));
          }
        }
        if (parts.length > 0) breakdownText = " title=\"Without this ship: " + parts.join(", ") + "\"";
      }
      var impactCell = hasAnalysis
        ? "<td class=\"value-col\"" + breakdownText + ">" + (fr.impact_pct != null ? fr.impact_pct.toFixed(1) + "%" : "-") + "</td>"
        : "<td class=\"value-col\">-</td>";
      var shipLabel = fr.key.replace(/_/g, " ");
      // In base_fleet mode, annotate ships the player already owns
      if (data.base_fleet && data.base_fleet[fr.key]) {
        var baseCnt = data.base_fleet[fr.key];
        var newCnt = fr.count - baseCnt;
        shipLabel += ' <span class="existing-badge" title="You already have ' + fmtNum(baseCnt) + '">existing: ' + fmtNum(baseCnt) + (newCnt > 0 ? ' +' + fmtNum(newCnt) + ' new' : '') + '</span>';
      }
      if (isSlowOrExpensive(fr.key)) shipLabel += ' <span style="color:#f87171;font-weight:bold" title="Slow or deuterium-expensive">*</span>';
      if (hasAnalysis) {
        var sv = (analysis[fr.key] || {}).survival_pct;
        if (sv != null && sv >= 95 && fr.count > 1) shipLabel += ' <span style="color:#4ade80" title="High survival rate">&#x26E8;</span>';
      }
      var survivalCell = "<td class=\"value-col\">" + (fr.survival_pct != null ? fmtNum(fr.count * fr.survival_pct / 100) + " <span style=\"color:#8b95a7;font-size:0.9em\">(" + fr.survival_pct.toFixed(1) + "%)</span>" : "-") + "</td>";
      var costPct = "";
      if (totalFleetCost > 0 && costPctMap[fr.key]) {
        costPct = "<td class=\"value-col\">" + (costPctMap[fr.key] / totalFleetCost * 100).toFixed(1) + "%</td>";
      } else {
        costPct = "<td class=\"value-col\">-</td>";
      }
      row.innerHTML = "<td>" + shipLabel + "</td><td>" + fmtNum(fr.count) + "</td>" + costPct + impactCell + survivalCell;
      tbody.appendChild(row);
    }
    if (fleetRows.length === 0) tbody.innerHTML = "<tr><td colspan=5>(empty fleet)</td></tr>";

    // ---- Defender fleet table (right column) ----
    var defTbody = document.querySelector("#defender-fleet-table tbody");
    var defSummary = document.getElementById("defender-fleet-summary");
    if (defTbody) {
      defTbody.innerHTML = "";
      var defAnalysis = data.defender_fleet_analysis || {};
      var defKeys = Object.keys(defAnalysis);
      if (defKeys.length === 0) {
        defTbody.innerHTML = "<tr><td colspan=4 style=\"color:#6c7891\">(no enemy ships)</td></tr>";
        if (defSummary) defSummary.textContent = "";
      } else {
        // Sort by initial count desc so biggest threats come first
        defKeys.sort(function(a, b) {
          var ca = (defAnalysis[a] && defAnalysis[a].count) || 0;
          var cb = (defAnalysis[b] && defAnalysis[b].count) || 0;
          if (cb !== ca) return cb - ca;
          return a < b ? -1 : (a > b ? 1 : 0);
        });
        var totalCount = 0, totalSurv = 0;
        var totalDefCost = 0;
        var defCostMap = {};
        for (var d = 0; d < defKeys.length; d++) {
          var dk0 = defKeys[d];
          var dc = shipCost(dk0) * ((defAnalysis[dk0] || {}).count || 0);
          totalDefCost += dc;
          defCostMap[dk0] = dc;
        }
        for (var d = 0; d < defKeys.length; d++) {
          var dk = defKeys[d];
          var info = defAnalysis[dk];
          var cnt = info.count || 0;
          var survCnt = info.surviving_count != null ? info.surviving_count : 0;
          var survPct = info.survival_pct;
          totalCount += cnt;
          totalSurv += survCnt;
          var defRow = document.createElement("tr");
          var destroyedClass = (survPct != null && survPct < 5) ? " class=\"defender-destroyed\"" : "";
          var survCell = survPct != null
            ? fmtNum(survCnt) + " <span style=\"color:#8b95a7;font-size:0.9em\">(" + survPct.toFixed(1) + "%)</span>"
            : "-";
          var defCostPct = (totalDefCost > 0 && defCostMap[dk]) ? "<td class=\"value-col\">" + (defCostMap[dk] / totalDefCost * 100).toFixed(1) + "%</td>" : "<td class=\"value-col\">-</td>";
          defRow.innerHTML = "<td" + destroyedClass + ">" + dk.replace(/_/g, " ") + "</td>"
                            + "<td" + destroyedClass + ">" + fmtNum(cnt) + "</td>"
                            + defCostPct
                            + "<td class=\"value-col\">" + survCell + "</td>";
          defTbody.appendChild(defRow);
        }
        if (defSummary) {
          var totalDestroyed = totalCount - totalSurv;
          var totalPct = totalCount > 0 ? (totalSurv / totalCount * 100).toFixed(1) : "0.0";
          defSummary.textContent = "Total: " + fmtNum(totalCount)
                                 + " ships, " + fmtNum(totalSurv) + " survive (" + totalPct + "%)"
                                 + " — " + fmtNum(totalDestroyed) + " destroyed by you";
        }
      }
    }

    // Legend: only show when impact analysis is present
    var legend = document.getElementById("fleet-legend");
    if (legend) {
      if (hasAnalysis) legend.classList.remove("hidden");
      else legend.classList.add("hidden");
    }

    renderEnemyDefenses();
    results.classList.remove("hidden");
    lastResult = data;
    pushHistory(data);
    // Auto-scroll to results on first run (or when results were hidden)
    if (!historyList || historyList.length <= 1) {
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (refineBtn) refineBtn.disabled = false;
  }

  // ---- Clear refine when budget multiplier changes ----
  var multSelect = document.querySelector('select[name="budget_multiplier"]');
  if (multSelect) {
    multSelect.addEventListener('change', function() {
      if (lastResult) {
        lastResult = null;
        refineCount = 0;
        if (refineBtn) refineBtn.disabled = true;
        var info = document.getElementById('refine-info');
        if (info) { info.textContent = 'Budget changed - starting fresh optimization'; info.classList.remove('hidden'); }
      }
    });
  }

  // ---- Refine button ----
  if (refineBtn) {
    refineBtn.addEventListener("click", function() {
      if (!lastResult) return;
      refineCount++;
      useSeedFleet = true;
      var info = document.getElementById("refine-info");
      if (info) {
        info.textContent = "Refinement #" + refineCount + " - iterating from previous best fleet";
        info.classList.remove("hidden");
      }
      form.dispatchEvent(new Event("submit", {cancelable: true}));
    });
  }

// ---- Paste parser ----
var SHIP_NAME_MAP = {
  "light fighter": "light_fighter", "heavy fighter": "heavy_fighter",
  "cruiser": "cruiser", "battleship": "battleship", "battlecruiser": "battlecruiser",
  "bomber": "bomber", "destroyer": "destroyer", "deathstar": "deathstar",
  "small cargo": "small_cargo", "large cargo": "large_cargo", "espionage probe": "espionage_probe",
  "pathfinder": "pathfinder", "recycler": "recycler", "reaper": "reaper",
  "solar satellite": "solar_satellite", "crawler": "crawler",
};
var DEFENSE_NAME_MAP = {
  "rocket launcher": "rocket_launcher", "light laser": "light_laser", "heavy laser": "heavy_laser",
  "gauss cannon": "gauss_cannon", "ion cannon": "ion_cannon", "plasma turret": "plasma_turret",
  "small shield dome": "small_shield_dome", "large shield dome": "large_shield_dome",
};
var UNSUPPORTED = ["interceptor", "ion bomber", "anti-ballistic missiles", "anti-ballistic missile", "interplanetary missiles", "interplanetary missile"];

function parseOGameReport(text, nameMap, unsupportedList) {
  var result = { parsed: {}, unknown: [], unsupported: [] };
  var lines = text.split(/\r?\n/);
  for (var i = 0; i < lines.length; i++) {
    var raw = lines[i].trim();
    if (!raw) continue;
    if (i + 1 < lines.length) {
      var numRaw = lines[i + 1].trim().replace(/,/g, "");
      var count = parseInt(numRaw, 10);
      if (isNaN(count) || count < 0) continue;
      var key = raw.toLowerCase();
      if (key in nameMap) { result.parsed[nameMap[key]] = count; i++; }
      else if (unsupportedList.indexOf(key) >= 0) { result.unsupported.push(raw + " (" + count + ")"); i++; }
      else { result.unknown.push(raw + " (" + count + ")"); i++; }
    }
  }
  return result;
}

function fillForm(parsed, prefix) {
  for (var k in parsed) {
    var el = document.querySelector('input[name="' + prefix + k + '"]');
    if (el) el.value = parsed[k];
  }
}

var parseBtn = document.getElementById("parse-btn");
if (parseBtn) {
  parseBtn.addEventListener("click", function() {
    var sb = document.getElementById("parse-status");
    sb.className = "parse-status"; sb.textContent = "";
    var ft = document.getElementById("paste-fleet").value;
    var dt = document.getElementById("paste-defenses").value;
    var msgs = [];
    // Handle combined spy reports pasted into Fleet textarea
    // Split at "Defence" / "Defenses" / "Defense" section header
    var fleetText = ft;
    var defText = dt;
    var combinedRegex = /
\s*(defen[cs]es?)\s*
/i;
    var ftMatch = ft.match(combinedRegex);
    if (ftMatch && !dt.trim()) {
      // User pasted combined report into Fleet field only
      var splitIdx = ft.indexOf(ftMatch[0]);
      fleetText = ft.substring(0, splitIdx);
      defText = ft.substring(splitIdx + ftMatch[0].length);
    }
    if (fleetText.trim()) {
      var r = parseOGameReport(fleetText, SHIP_NAME_MAP, UNSUPPORTED);
      fillForm(r.parsed, "");
      msgs.push("Fleet: " + Object.keys(r.parsed).length + " types parsed");
      if (r.unsupported.length) msgs.push("Skipped (unsupported): " + r.unsupported.join(", "));
      if (r.unknown.length) msgs.push("Unknown: " + r.unknown.join(", "));
    }
    if (defText.trim()) {
      var r2 = parseOGameReport(defText, DEFENSE_NAME_MAP, ["anti-ballistic missiles", "anti-ballistic missile", "interplanetary missiles", "interplanetary missile"]);
      fillForm(r2.parsed, "");
      msgs.push("Defenses: " + Object.keys(r2.parsed).length + " types parsed");
      if (r2.unsupported.length) msgs.push("Skipped (unsupported): " + r2.unsupported.join(", "));
      if (r2.unknown.length) msgs.push("Unknown defenses: " + r2.unknown.join(", "));
    }
    if (!msgs.length) { sb.textContent = "Nothing to parse. Paste data first."; sb.classList.add("parse-warning"); }
    else { sb.innerHTML = msgs.map(function(m) { return "<div>" + m + "</div>"; }).join(""); sb.classList.add("parse-ok"); }
  });
}

  // Deathstar and Bomber are default-excluded via the `checked` attribute
  // in the HTML so they are visible to the user on first paint. Do not
  // toggle them in JS — it would silently override what the user sees.

  // Resource priority: reset-to-1:1:1 button (so users can quickly
  // disable the metal preference if they don't want it)
  var resetBtn = document.getElementById("reset-weights");
  if (resetBtn) {
    resetBtn.addEventListener("click", function() {
      var wm = document.getElementById("weight_m");
      var wc = document.getElementById("weight_c");
      var wd = document.getElementById("weight_d");
      var beta = document.getElementById("preference_beta");
      var betaLabel = document.getElementById("preference_beta_value");
      if (wm) wm.value = "1.0";
      if (wc) wc.value = "1.0";
      if (wd) wd.value = "1.0";
      if (beta) beta.value = "0.05";
      if (betaLabel) betaLabel.textContent = "0.05";
    });
  }

  // Live-update the β value display as the slider moves
  var betaSlider = document.getElementById("preference_beta");
  var betaLabel = document.getElementById("preference_beta_value");
  if (betaSlider && betaLabel) {
    betaSlider.addEventListener("input", function() {
      betaLabel.textContent = parseFloat(betaSlider.value).toFixed(2);
    });
  }

  // ---- Tab switching ----
  var tabBtns = document.querySelectorAll(".tab-btn");
  var myFleetSection = document.getElementById("my-fleet-section");
  var budgetHintCounter = document.getElementById("budget-hint-counter");
  var budgetHintMyfleet = document.getElementById("budget-hint-myfleet");
  var multSelect = document.querySelector('select[name="budget_multiplier"]');
  function switchTab(tabName) {
    activeTab = tabName;
    console.log('DIAG switchTab called:', tabName);
    var dbg = document.getElementById('debug-badge');
    if (dbg) dbg.textContent = 'TAB: ' + tabName + ' | base_fleet: ' + (activeTab === 'myfleet' ? 'will be sent' : 'not set');
    tabBtns.forEach(function(b) { b.classList.toggle("active", b.dataset.tab === tabName); });
    if (myFleetSection) myFleetSection.classList.toggle("hidden", tabName !== "myfleet");
    if (budgetHintCounter) budgetHintCounter.classList.toggle("hidden", tabName === "myfleet");
    if (budgetHintMyfleet) budgetHintMyfleet.classList.toggle("hidden", tabName !== "myfleet");
    // Default multiplier per tab
    if (multSelect) {
      multSelect.value = tabName === "myfleet" ? "0.1" : "1.0";
    }
    // Reset refine state on tab switch
    if (lastResult) {
      lastResult = null;
      refineCount = 0;
      if (refineBtn) refineBtn.disabled = true;
    }
  }
  tabBtns.forEach(function(b) {
    b.addEventListener("click", function() { switchTab(b.dataset.tab); });
  });

  // ---- Clean fleet button (clears enemy inputs) ----
  var cleanBtn = document.getElementById("clean-fleet-btn");
  if (cleanBtn) {
    cleanBtn.addEventListener("click", function() {
      SHIP_KEYS.forEach(function(k) {
        var el = document.querySelector('input[name="' + k + '"]');
        if (el) el.value = "0";
      });
      DEFENSE_KEYS.forEach(function(k) {
        var el = document.querySelector('input[name="' + k + '"]');
        if (el) el.value = "0";
      });
      var pf = document.getElementById("paste-fleet");
      var pd = document.getElementById("paste-defenses");
      if (pf) pf.value = "";
      if (pd) pd.value = "";
      var sb = document.getElementById("parse-status");
      if (sb) { sb.textContent = "Enemy fleet and defenses cleared."; sb.className = "parse-status parse-ok"; }
    });
  }

  // ---- Parse my fleet button ----
  var parseMyBtn = document.getElementById("parse-my-fleet-btn");
  if (parseMyBtn) {
    parseMyBtn.addEventListener("click", function() {
      var sb = document.getElementById("my-fleet-status");
      if (sb) { sb.className = "parse-status"; sb.textContent = ""; }
      var text = document.getElementById("paste-my-fleet").value;
      if (!text.trim()) {
        if (sb) { sb.textContent = "Nothing to parse."; sb.classList.add("parse-warning"); }
        return;
      }
      var r = parseOGameReport(text, SHIP_NAME_MAP, UNSUPPORTED);
      // Fill my_* inputs
      for (var k in r.parsed) {
        var el = document.querySelector('input[name="my_' + k + '"]');
        if (el) el.value = r.parsed[k];
      }
      var msgs = ["Fleet: " + Object.keys(r.parsed).length + " types parsed"];
      if (r.unsupported.length) msgs.push("Skipped (unsupported): " + r.unsupported.join(", "));
      if (r.unknown.length) msgs.push("Unknown: " + r.unknown.join(", "));
      if (sb) { sb.innerHTML = msgs.map(function(m) { return "<div>" + m + "</div>"; }).join(""); sb.classList.add("parse-ok"); }
    });
  }

  // ---- Clear my fleet button ----
  var clearMyBtn = document.getElementById("clear-my-fleet-btn");
  if (clearMyBtn) {
    clearMyBtn.addEventListener("click", function() {
      var inputs = document.querySelectorAll('input[name^="my_"]');
      inputs.forEach(function(el) { el.value = "0"; });
      var ta = document.getElementById("paste-my-fleet");
      if (ta) ta.value = "";
      var sb = document.getElementById("my-fleet-status");
      if (sb) { sb.textContent = ""; sb.className = "parse-status"; }
    });
  }

  // ---- Copy table to clipboard (for OGame boards, Discord, etc.) ----

  function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function() {
        showToast("Copied to clipboard!");
      }).catch(function() { fallbackCopy(text); });
    } else {
      fallbackCopy(text);
    }
  }
  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); showToast("Copied!"); }
    catch (e) { showToast("Copy failed - select and Ctrl+C"); }
    document.body.removeChild(ta);
  }
  var toastTimer = null;
  function showToast(msg) {
    var t = document.getElementById("copy-toast");
    if (!t) return;
    t.textContent = msg;
    t.classList.add("copy-toast-show");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function() { t.classList.remove("copy-toast-show"); }, 2000);
  }

  function padR(str, len) { str = String(str); while (str.length < len) str += " "; return str; }
  function padL(str, len) { str = String(str); while (str.length < len) str = " " + str; return str; }

  function buildTextTable(headers, dataRows) {
    var allRows = [headers].concat(dataRows);
    var widths = [];
    for (var c = 0; c < headers.length; c++) {
      var w = 0;
      for (var r = 0; r < allRows.length; r++) {
        if (allRows[r][c] && String(allRows[r][c]).length > w) w = String(allRows[r][c]).length;
      }
      widths.push(w);
    }
    var lines = [];
    for (var r = 0; r < allRows.length; r++) {
      var cells = [];
      for (var c = 0; c < allRows[r].length; c++) {
        var val = allRows[r][c] || "";
        cells.push(c === 0 ? padR(val, widths[c]) : padL(val, widths[c]));
      }
      lines.push(cells.join(" | "));
    }
    return lines.join("\n");
  }

  function buildHtmlTable(headers, dataRows, title) {
    var h = '<table style="border-collapse:collapse;font-family:monospace;font-size:13px">';
    if (title) h += '<caption style="caption-side:top;font-weight:bold;padding:4px;text-align:left">' + title + '</caption>';
    h += "<thead><tr>";
    for (var i = 0; i < headers.length; i++) h += '<th style="border:1px solid #555;padding:3px 8px;background:#2a2a4a;color:#e0e0e0">' + headers[i] + "</th>";
    h += "</tr></thead><tbody>";
    for (var r = 0; r < dataRows.length; r++) {
      h += "<tr>";
      for (var c = 0; c < dataRows[r].length; c++) {
        var align = c === 0 ? "left" : "right";
        h += '<td style="border:1px solid #555;padding:3px 8px;text-align:' + align + '">' + dataRows[r][c] + "</td>";
      }
      h += "</tr>";
    }
    h += "</tbody></table>";
    return h;
  }

  function copyFleetTable(format) {
    if (!lastResult) return;
    var fleet = lastResult.recommended_fleet || {};
    var analysis = lastResult.fleet_analysis || {};
    var rows = [];
    var totalCost = 0;
    for (var k in fleet) {
      if (!fleet[k] || fleet[k] <= 0) continue;
      totalCost += shipCost(k) * fleet[k];
    }
    for (var k in fleet) {
      if (!fleet[k] || fleet[k] <= 0) continue;
      var info = analysis[k] || {};
      var impact = info.impact_pct != null ? (info.impact_pct > 0 ? "+" : "") + info.impact_pct.toFixed(1) + "%" : "-";
      var survPct = info.survival_pct;
      var survCount = survPct != null ? Math.round(fleet[k] * survPct / 100) : null;
      var surv = survPct != null ? fmtNum(survCount) + " (" + survPct.toFixed(1) + "%)" : "-";
      var costPct = totalCost > 0 ? (shipCost(k) * fleet[k] / totalCost * 100).toFixed(1) + "%" : "-";
      rows.push({
        sort: info.impact_pct != null ? info.impact_pct : -Infinity,
        cells: [k.replace(/_/g, " "), fmtNum(fleet[k]), costPct, impact, surv]
      });
    }
    rows.sort(function(a, b) { return b.sort - a.sort; });
    var headers = ["Ship", "Count", "Cost %", "Impact %", "Surviving"];
    var data = rows.map(function(r) { return r.cells; });
    var text = format === "html"
      ? buildHtmlTable(headers, data, "Recommended Fleet")
      : buildTextTable(headers, data);
    copyToClipboard(text);
  }

  function copyDefenderTable(format) {
    if (!lastResult) return;
    var defAnalysis = lastResult.defender_fleet_analysis || {};
    var keys = Object.keys(defAnalysis);
    if (keys.length === 0) return;
    keys.sort(function(a, b) {
      var ca = (defAnalysis[a] && defAnalysis[a].count) || 0;
      var cb = (defAnalysis[b] && defAnalysis[b].count) || 0;
      return cb - ca;
    });
    var totalDefCost = 0;
    for (var d = 0; d < keys.length; d++) totalDefCost += shipCost(keys[d]) * ((defAnalysis[keys[d]] || {}).count || 0);
    var rows = [];
    for (var d = 0; d < keys.length; d++) {
      var info = defAnalysis[keys[d]];
      var cnt = (info && info.count) || 0;
      var survPct = info && info.survival_pct;
      var survCount = info && info.surviving_count != null ? info.surviving_count : 0;
      var surv = survPct != null ? fmtNum(survCount) + " (" + survPct.toFixed(1) + "%)" : "-";
      var costPct = totalDefCost > 0 ? (shipCost(keys[d]) * cnt / totalDefCost * 100).toFixed(1) + "%" : "-";
      rows.push([keys[d].replace(/_/g, " "), fmtNum(cnt), costPct, surv]);
    }
    var headers = ["Ship", "Count", "Cost %", "Surviving"];
    var text = format === "html"
      ? buildHtmlTable(headers, rows, "Defender Fleet (Enemy)")
      : buildTextTable(headers, rows);
    copyToClipboard(text);
  }


  // ---- Copy fleet in OGame paste format (Ship / Count per line) ----
  function copyFleetOGame() {
    if (!lastResult) return;
    var fleet = lastResult.recommended_fleet || {};
    var lines = [];
    var keys = Object.keys(fleet).filter(function(k) { return fleet[k] > 0; }).sort();
    for (var i = 0; i < keys.length; i++) {
      var name = KEY_TO_DISPLAY ? (KEY_TO_DISPLAY[keys[i]] || keys[i]) : keys[i];
      name = name.charAt(0).toUpperCase() + name.slice(1);
      lines.push(name + "\n" + fleet[keys[i]].toLocaleString());
    }
    copyToClipboard(lines.join("\n"));
  }

  function copyDefenderOGame() {
    if (!lastResult) return;
    var defAnalysis = lastResult.defender_fleet_analysis || {};
    var lines = [];
    // Enemy ships
    var shipKeys = Object.keys(defAnalysis).sort();
    for (var i = 0; i < shipKeys.length; i++) {
      var info = defAnalysis[shipKeys[i]];
      var cnt = info.count || 0;
      if (cnt > 0) {
        var name = (KEY_TO_DISPLAY ? (KEY_TO_DISPLAY[shipKeys[i]] || shipKeys[i]) : shipKeys[i]);
        name = name.charAt(0).toUpperCase() + name.slice(1);
        lines.push(name + "\n" + cnt.toLocaleString());
      }
    }
    // Enemy defenses (from lastRequest)
    if (lastRequest && lastRequest.enemy_defenses && lastRequest.enemy_defenses.defenses) {
      var defenses = lastRequest.enemy_defenses.defenses;
      var defKeys = Object.keys(defenses).filter(function(k) { return defenses[k] > 0; }).sort();
      for (var d = 0; d < defKeys.length; d++) {
        var dname = (DEF_KEY_TO_DISPLAY ? (DEF_KEY_TO_DISPLAY[defKeys[d]] || defKeys[d]) : defKeys[d]);
        dname = dname.charAt(0).toUpperCase() + dname.slice(1);
        lines.push(dname + "\n" + defenses[defKeys[d]].toLocaleString());
      }
    }
    if (lines.length > 0) copyToClipboard(lines.join("\n"));
  }

  // ---- Render enemy defenses summary in results ----
  function renderEnemyDefenses() {
    var container = document.getElementById("enemy-defenses-summary");
    if (!container) return;
    if (!lastRequest || !lastRequest.enemy_defenses || !lastRequest.enemy_defenses.defenses) {
      container.innerHTML = "";
      return;
    }
    var defenses = lastRequest.enemy_defenses.defenses;
    var keys = Object.keys(defenses).filter(function(k) { return defenses[k] > 0; }).sort();
    if (keys.length === 0) {
      container.innerHTML = "";
      return;
    }
    var parts = [];
    for (var i = 0; i < keys.length; i++) {
      var name = (DEF_KEY_TO_DISPLAY ? (DEF_KEY_TO_DISPLAY[keys[i]] || keys[i]) : keys[i]);
      name = name.charAt(0).toUpperCase() + name.slice(1);
      parts.push(name + ": " + defenses[keys[i]].toLocaleString());
    }
    container.innerHTML = '<span class="defenses-summary-label">Enemy Defenses:</span> ' + parts.join(" | ");
  }

  // ============================================================
  // History: store last 6 results in localStorage, render tabs + graph
  // ============================================================
  function loadHistory() {
    try {
      var raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) { return []; }
  }
  function saveHistory(list) {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)); } catch (e) {}
  }
  function snapshotResult(data, idx) {
    return {
      ts: Date.now(),
      label: idx === 0 ? 'latest' : '#' + idx,
      recommended_fleet: data.recommended_fleet,
      recommended_additions: data.recommended_additions,
      base_fleet: data.base_fleet,
      base_fleet_cost: data.base_fleet_cost,
      base_fleet_count: data.base_fleet_count,
      fleet_value: data.fleet_value,
      fleet_lost_pct: data.fleet_lost_pct,
      ships_lost_count: data.ships_lost_count,
      ships_initial_count: data.ships_initial_count,
      raw_loss_mean: data.raw_loss_mean,
      expected_loss_mean: data.expected_loss_mean,
      expected_loss_stddev: data.expected_loss_stddev,
      win_probability: data.win_probability,
      confidence_interval_95: data.confidence_interval_95,
      sims_run_final: data.sims_run_final,
      debris_metal: data.debris_metal,
      debris_crystal: data.debris_crystal,
      debris_deuterium: data.debris_deuterium,
      debris_total: data.debris_total,
      net_profit: data.net_profit,
      net_profit_pct: data.net_profit_pct,
      recyclers_needed: data.recyclers_needed,
      recyclers_cost_metal: data.recyclers_cost_metal,
      recyclers_cost_crystal: data.recyclers_cost_crystal,
      recyclers_cost_deuterium: data.recyclers_cost_deuterium,
      recyclers_cost_total: data.recyclers_cost_total,
      recycler_capacity: data.recycler_capacity,
      fleet_analysis: data.fleet_analysis,
      defender_fleet_analysis: data.defender_fleet_analysis,
      mode: data.mode,
      seed_used: data.seed_used,
      time_elapsed_total: data.time_elapsed_total,
      min_gain_required: data.min_gain_required,
      min_gain_met: data.min_gain_met,
      actual_roi_pct: data.actual_roi_pct,
    };
  }
  function pushHistory(data) {
    var snap = snapshotResult(data, 0);
    historyList = loadHistory();
    historyList.unshift(snap);
    if (historyList.length > HISTORY_MAX) historyList = historyList.slice(0, HISTORY_MAX);
    for (var i = 0; i < historyList.length; i++) {
      historyList[i].label = i === 0 ? 'latest' : '#' + i;
    }
    saveHistory(historyList);
    renderHistoryTabs();
    renderHistoryGraph();
  }
  function renderHistoryTabs() {
    var bar = document.getElementById('history-bar');
    var tabs = document.getElementById('history-tabs');
    if (!bar || !tabs) return;
    if (historyList.length <= 1) {
      bar.classList.add('hidden');
      return;
    }
    bar.classList.remove('hidden');
    tabs.innerHTML = '';
    for (var i = 0; i < historyList.length; i++) {
      (function(idx) {
        var snap = historyList[idx];
        var wp = snap.win_probability || 0;
        var cls = wp >= 0.95 ? 'win-green' : (wp >= 0.8 ? 'win-yellow' : 'win-red');
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'history-tab' + (idx === 0 ? ' active' : '');
        var npSign = snap.net_profit >= 0 ? '+' : '';
        var npM = Math.round(snap.net_profit / 1e6);
        btn.innerHTML = '<span class="' + cls + '">' + Math.round(wp * 100) + '%</span> ' + npSign + npM + 'M';
        btn.title = 'Sim #' + idx + ' - ' + new Date(snap.ts).toLocaleTimeString();
        btn.addEventListener('click', function() {
          loadHistoryEntry(idx);
        });
        tabs.appendChild(btn);
      })(i);
    }
  }
  function loadHistoryEntry(idx) {
    var snap = historyList[idx];
    if (!snap) return;
    lastResult = snap;
    renderResults(snap);
    renderHistoryTabs();
    var r = document.getElementById('results');
    if (r) r.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  function renderHistoryGraph() {
    var container = document.getElementById('history-graph');
    var svg = document.getElementById('history-svg');
    if (!container || !svg) return;
    if (historyList.length < 2) {
      container.classList.add('hidden');
      return;
    }
    container.classList.remove('hidden');
    // Oldest first so line goes left-to-right with newest rightmost
    var pts = historyList.slice().reverse();
    var n = pts.length;
    var w = 600, h = 140;
    var padX = 30, padY = 10;
    var plotW = w - padX * 2, plotH = h - padY * 2;
    function range(arr) {
      var mn = Infinity, mx = -Infinity;
      for (var i = 0; i < arr.length; i++) {
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
      }
      if (mn === mx) { mn -= 1; mx += 1; }
      return [mn, mx];
    }
    var netProfits = pts.map(function(p) { return p.net_profit_pct || 0; });
    var rawLosses = pts.map(function(p) { return (p.raw_loss_mean || 0) / 1e6; });
    var shipsLost = pts.map(function(p) {
      if (p.ships_initial_count && p.ships_initial_count > 0) {
        return (p.ships_lost_count / p.ships_initial_count) * 100;
      }
      return p.fleet_lost_pct || 0;
    });
    var r1 = range(netProfits), r2 = range(rawLosses), r3 = range(shipsLost);
    function normalize(v, r) { return padY + plotH - ((v - r[0]) / (r[1] - r[0])) * plotH; }
    function xPos(i) { return padX + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW); }
    function makePath(arr, r) {
      var pts2 = [];
      for (var i = 0; i < arr.length; i++) {
        pts2.push(xPos(i) + ',' + normalize(arr[i], r));
      }
      return 'M' + pts2.join(' L');
    }
    var html = '';
    html += '<line x1="' + padX + '" y1="' + padY + '" x2="' + padX + '" y2="' + (h - padY) + '" stroke="#3a3a5a" stroke-width="1"/>';
    html += '<line x1="' + padX + '" y1="' + (h - padY) + '" x2="' + (w - padX) + '" y2="' + (h - padY) + '" stroke="#3a3a5a" stroke-width="1"/>';
    html += '<path d="' + makePath(netProfits, r1) + '" stroke="#4ade80" stroke-width="2" fill="none"/>';
    html += '<path d="' + makePath(rawLosses, r2) + '" stroke="#f87171" stroke-width="2" fill="none"/>';
    html += '<path d="' + makePath(shipsLost, r3) + '" stroke="#60a5fa" stroke-width="2" fill="none"/>';
    for (var i = 0; i < n; i++) {
      var xp = xPos(i);
      html += '<circle cx="' + xp + '" cy="' + normalize(netProfits[i], r1) + '" r="3" fill="#4ade80"/>';
      html += '<circle cx="' + xp + '" cy="' + normalize(rawLosses[i], r2) + '" r="3" fill="#f87171"/>';
      html += '<circle cx="' + xp + '" cy="' + normalize(shipsLost[i], r3) + '" r="3" fill="#60a5fa"/>';
    }
    svg.innerHTML = html;
  }
  // Load history on startup
  historyList = loadHistory();
  renderHistoryTabs();
  renderHistoryGraph();

  // Expose copy functions to global scope for inline onclick handlers
  window.copyFleetTable = copyFleetTable;
  window.copyDefenderTable = copyDefenderTable;
  window.copyFleetOGame = copyFleetOGame;
  window.copyDefenderOGame = copyDefenderOGame;
  window.switchTab = switchTab;

  // ============================================================
  // Fleet Presets: localStorage save/load + TXT/XML export/import
  // ============================================================

  // Reverse name map: ship_key -> display name (e.g. "light_fighter" -> "light fighter")
  var KEY_TO_DISPLAY = {};
  for (var dn in SHIP_NAME_MAP) { KEY_TO_DISPLAY[SHIP_NAME_MAP[dn]] = dn; }
  var DEF_KEY_TO_DISPLAY = {};
  for (var dn2 in DEFENSE_NAME_MAP) { DEF_KEY_TO_DISPLAY[DEFENSE_NAME_MAP[dn2]] = dn2; }

  function getPresets(storageKey) {
    try {
      var raw = localStorage.getItem(storageKey);
      return raw ? JSON.parse(raw) : {};
    } catch (e) { return {}; }
  }

  function savePresetToStorage(storageKey, name, data) {
    var presets = getPresets(storageKey);
    presets[name] = data;
    localStorage.setItem(storageKey, JSON.stringify(presets));
  }

    function deletePresetFromStorage(storageKey, name) {
    var presets = getPresets(storageKey);
    delete presets[name];
    localStorage.setItem(storageKey, JSON.stringify(presets));
  }

  function populatePresetDropdown(selectEl, storageKey) {
    var presets = getPresets(storageKey);
    var keys = Object.keys(presets).sort();
    selectEl.innerHTML = '<option value="">-- Saved (' + keys.length + ') --</option>';
    for (var i = 0; i < keys.length; i++) {
      var name = keys[i];
      var data = presets[name];
      var types = data.ships ? Object.keys(data.ships).length : 0;
      var opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name + " (" + types + " types)";
      selectEl.appendChild(opt);
    }
  }

  // --- Export formats ---
  function exportFleetAsTxt(ships) {
    var lines = [];
    var keys = Object.keys(ships).filter(function(k) { return ships[k] > 0; }).sort();
    for (var i = 0; i < keys.length; i++) {
      var name = KEY_TO_DISPLAY[keys[i]] || keys[i];
      lines.push(name.charAt(0).toUpperCase() + name.slice(1) + "\n" + ships[keys[i]].toLocaleString());
    }
    return lines.join("\n");
  }

  function exportFleetAsXml(name, ships) {
    var keys = Object.keys(ships).filter(function(k) { return ships[k] > 0; }).sort();
    var lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                 '<fleetPreset name="' + (name || "preset").replace(/"/g, "&quot;") + '" timestamp="' + new Date().toISOString() + '">',
                 "  <ships>"];
    for (var i = 0; i < keys.length; i++) {
      lines.push('    <ship key="' + keys[i] + '" count="' + ships[keys[i]] + '"/>');
    }
    lines.push("  </ships>");
    lines.push("</fleetPreset>");
    return lines.join("\n");
  }

  // --- Import with auto-detect (TXT or XML) ---
  function importFleetAuto(text, nameMap) {
    var trimmed = text.trim();
    if (!trimmed) return {};
    // Auto-detect XML
    if (trimmed.indexOf("<fleetPreset") >= 0 || trimmed.indexOf("<ship ") >= 0) {
      return importFleetFromXml(trimmed);
    }
    // Fallback to TXT (reuse existing parser)
    var r = parseOGameReport(trimmed, nameMap, UNSUPPORTED);
    return r.parsed;
  }

  function importFleetFromXml(text) {
    var ships = {};
    var re = /<ship\s+key="([^"]+)"\s+count="(\d+)"\s*\/>/g;
    var m;
    while ((m = re.exec(text)) !== null) {
      var key = m[1];
      var count = parseInt(m[2], 10);
      if (count > 0) ships[key] = count;
    }
    return ships;
  }

  // --- Fill form inputs from fleet dict ---
  function fillFormInputs(ships, prefix) {
    // Zero out existing inputs first
    var pattern = prefix ? 'input[name^="' + prefix + '"]' : 'input[name]:not([name^="my_"]):not([name^="exclude"])';
    var inputs = document.querySelectorAll(pattern);
    for (var i = 0; i < inputs.length; i++) {
      var name = inputs[i].name;
      if (prefix) name = name.replace(prefix, "");
      if (name in SHIP_NAME_MAP || Object.values(SHIP_NAME_MAP).indexOf(name) >= 0) {
        inputs[i].value = "0";
      }
    }
    // Fill from ships
    for (var k in ships) {
      var sel = prefix ? 'input[name="' + prefix + k + '"]' : 'input[name="' + k + '"]';
      var el = document.querySelector(sel);
      if (el) el.value = ships[k];
    }
  }

  // --- Read fleet from form inputs ---
  function readFormFleet(prefix) {
    var fleet = {};
    var keys = prefix ? Object.values(SHIP_NAME_MAP) : SHIP_KEYS;
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i];
      var name = prefix ? prefix + k : k;
      var el = document.querySelector('input[name="' + name + '"]');
      if (el) {
        var v = parseInt(el.value || "0", 10);
        if (v > 0) fleet[k] = v;
      }
    }
    return fleet;
  }

  // ============================================================
  // Wire up preset controls for both My Fleet and Enemy Fleet
  // ============================================================

  function setupPresetControls(config) {
    var storageKey = config.storageKey;
    var prefix = config.prefix; // "my_" or ""
    var selectEl = document.getElementById(config.selectId);
    var loadBtn = document.getElementById(config.loadId);
    var saveBtn = document.getElementById(config.saveId);
    var delBtn = document.getElementById(config.delId);
    var exportTxtBtn = document.getElementById(config.exportTxtId);
    var exportXmlBtn = document.getElementById(config.exportXmlId);
    var importToggleBtn = document.getElementById(config.importToggleId);
    var importSection = document.getElementById(config.importSectionId);
    var importTextarea = document.getElementById(config.importTextareaId);
    var importConfirm = document.getElementById(config.importConfirmId);
    var importCancel = document.getElementById(config.importCancelId);
    var importStatus = document.getElementById(config.importStatusId);

    if (!selectEl) return;
    populatePresetDropdown(selectEl, storageKey);

    if (loadBtn) {
      loadBtn.addEventListener("click", function() {
        var name = selectEl.value;
        if (!name) { showToast("Select a preset first"); return; }
        var presets = getPresets(storageKey);
        var data = presets[name];
        if (!data || !data.ships) { showToast("Preset not found"); return; }
        fillFormInputs(data.ships, prefix);
        showToast("Loaded: " + name);
      });
    }

    if (saveBtn) {
      saveBtn.addEventListener("click", function() {
        var name = prompt("Name this preset:");
        if (!name) return;
        name = name.trim();
        if (!name) return;
        var fleet = readFormFleet(prefix);
        if (Object.keys(fleet).length === 0) {
          showToast("No ships to save"); return;
        }
        var data = { ships: fleet, saved_at: new Date().toISOString() };
        savePresetToStorage(storageKey, name, data);
        populatePresetDropdown(selectEl, storageKey);
        selectEl.value = name;
        showToast("Saved: " + name);
      });
    }

    if (delBtn) {
      delBtn.addEventListener("click", function() {
        var name = selectEl.value;
        if (!name) { showToast("Select a preset first"); return; }
        if (!confirm("Delete preset \"" + name + "\"?")) return;
        deletePresetFromStorage(storageKey, name);
        populatePresetDropdown(selectEl, storageKey);
        showToast("Deleted: " + name);
      });
    }

    function doExport(format) {
      var fleet = readFormFleet(prefix);
      if (Object.keys(fleet).length === 0) {
        showToast("Nothing to export"); return;
      }
      var name = selectEl.value || "fleet";
      var text = format === "xml" ? exportFleetAsXml(name, fleet) : exportFleetAsTxt(fleet);
      copyToClipboard(text);
    }

    if (exportTxtBtn) exportTxtBtn.addEventListener("click", function() { doExport("txt"); });
    if (exportXmlBtn) exportXmlBtn.addEventListener("click", function() { doExport("xml"); });

    if (importToggleBtn) {
      importToggleBtn.addEventListener("click", function() {
        if (importSection) importSection.classList.toggle("hidden");
        if (importTextarea && !importSection.classList.contains("hidden")) {
          importTextarea.value = "";
          importTextarea.focus();
        }
        if (importStatus) { importStatus.textContent = ""; importStatus.className = "parse-status"; }
      });
    }

    if (importCancel) {
      importCancel.addEventListener("click", function() {
        if (importSection) importSection.classList.add("hidden");
        if (importTextarea) importTextarea.value = "";
        if (importStatus) { importStatus.textContent = ""; importStatus.className = "parse-status"; }
      });
    }

    if (importConfirm) {
      importConfirm.addEventListener("click", function() {
        var text = importTextarea ? importTextarea.value : "";
        if (!text.trim()) {
          if (importStatus) { importStatus.textContent = "Nothing to import."; importStatus.className = "parse-status parse-warning"; }
          return;
        }
        var nameMap = prefix ? SHIP_NAME_MAP : SHIP_NAME_MAP;
        var parsed = importFleetAuto(text, nameMap);
        var count = Object.keys(parsed).length;
        if (count === 0) {
          if (importStatus) { importStatus.textContent = "No ships found in pasted text."; importStatus.className = "parse-status parse-warning"; }
          return;
        }
        fillFormInputs(parsed, prefix);
        if (importStatus) { importStatus.textContent = "Imported " + count + " types."; importStatus.className = "parse-status parse-ok"; }
        if (importSection) importSection.classList.add("hidden");
        showToast("Imported " + count + " types");
      });
    }
  }

  // Wire up My Fleet presets
  setupPresetControls({
    storageKey: "ogame_optimizer_presets_my_fleet",
    prefix: "my_",
    selectId: "my-preset-select",
    loadId: "my-preset-load",
    saveId: "my-preset-save",
    delId: "my-preset-delete",
    exportTxtId: "my-preset-export-txt",
    exportXmlId: "my-preset-export-xml",
    importToggleId: "my-preset-import-toggle",
    importSectionId: "my-import-section",
    importTextareaId: "my-import-textarea",
    importConfirmId: "my-import-confirm",
    importCancelId: "my-import-cancel",
    importStatusId: "my-import-status",
  });

  // Wire up Enemy Fleet presets
  setupPresetControls({
    storageKey: "ogame_optimizer_presets_enemy_fleet",
    prefix: "",
    selectId: "enemy-preset-select",
    loadId: "enemy-preset-load",
    saveId: "enemy-preset-save",
    delId: "enemy-preset-delete",
    exportTxtId: "enemy-preset-export-txt",
    exportXmlId: "enemy-preset-export-xml",
    importToggleId: "enemy-preset-import-toggle",
    importSectionId: "enemy-import-section",
    importTextareaId: "enemy-import-textarea",
    importConfirmId: "enemy-import-confirm",
    importCancelId: "enemy-import-cancel",
    importStatusId: "enemy-import-status",
  });

})();
