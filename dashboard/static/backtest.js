// Dashboard backtesting: prediksi out-of-sample vs harga aktual.

// Library grafik dimuat dari CDN; tampilkan pesan jelas jika gagal (misalnya offline).
if (typeof LightweightCharts === "undefined") {
  document.querySelector("main").insertAdjacentHTML(
    "afterbegin",
    '<div class="banner">Library grafik (unpkg.com/lightweight-charts) gagal dimuat. Periksa koneksi internet lalu muat ulang halaman.</div>'
  );
  throw new Error("LightweightCharts tidak tersedia");
}

const PAGE_SIZE = 100;
const DEFAULT_WINDOW_DAYS = 3;
const DAY_SECONDS = 86400;

const state = {
  runs: [],
  runId: null,
  model: null,
  period: null,
  summary: null,
  predictions: [],
  byCandleTime: new Map(),
  bars: [],
  windowStart: null,
  windowEnd: null,
  selected: null,
  priceLines: [],
  page: 0,
  signalFilter: "ALL",
  resultFilter: "ALL",
  equity: null,
};

const chartParts = createCandleChart(document.getElementById("chart"));
const legend = document.getElementById("legend");
attachLegend(chartParts, legend, function () { return state.bars[state.bars.length - 1]; });

// ------------------------------------------------------------------ helper

function element(id) {
  return document.getElementById(id);
}

function dateInputValue(epochSeconds) {
  return new Date(epochSeconds * 1000).toISOString().slice(0, 10);
}

function dateInputToEpoch(value, endOfDay) {
  if (!value) {
    return null;
  }
  let seconds = Math.floor(new Date(value + "T00:00:00Z").getTime() / 1000);
  if (endOfDay) {
    seconds += DAY_SECONDS;
  }
  return seconds;
}

function isoFromEpoch(seconds) {
  return new Date(seconds * 1000).toISOString();
}

function baseUrl() {
  return "/api/backtest/" + encodeURIComponent(state.runId) + "/" +
    encodeURIComponent(state.model) + "/" + encodeURIComponent(state.period);
}

function metricCard(label, value, sub, color) {
  let style = "";
  if (color) {
    style = ' style="color:' + color + '"';
  }
  return '<div class="metric"><div class="label">' + label + '</div><div class="value num"' + style + ">" +
    value + '</div><div class="sub">' + (sub || "&nbsp;") + "</div></div>";
}

function showOverlay(text) {
  const overlay = element("chartOverlay");
  if (text === null) {
    overlay.hidden = true;
    return;
  }
  overlay.hidden = false;
  element("chartOverlayText").textContent = text;
}

function entryReference(row) {
  if (row.signal === "BUY") {
    return row.ask_reference;
  }
  return row.bid_reference;
}

function levels(row) {
  if (row.signal === "BUY") {
    return { entry: row.ask_reference, takeProfit: row.close + row.barrier, stopLoss: row.close - row.barrier };
  }
  return { entry: row.bid_reference, takeProfit: row.close - row.barrier, stopLoss: row.close + row.barrier };
}

// ------------------------------------------------------------------ pilihan run

function fillSelect(select, values, selected) {
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    if (value === selected) {
      option.selected = true;
    }
    select.appendChild(option);
  }
}

function currentRun() {
  for (const run of state.runs) {
    if (run.run_id === state.runId) {
      return run;
    }
  }
  return null;
}

function refreshSelectors() {
  const runIds = [];
  for (const run of state.runs) {
    runIds.push(run.run_id);
  }
  fillSelect(element("runSelect"), runIds, state.runId);

  const run = currentRun();
  const models = Object.keys(run.models);
  if (!models.includes(state.model)) {
    if (models.includes("xgboost")) {
      state.model = "xgboost";
    } else {
      state.model = models[0];
    }
  }
  fillSelect(element("modelSelect"), models, state.model);

  const periods = run.models[state.model];
  if (!periods.includes(state.period)) {
    if (periods.includes("final_test")) {
      state.period = "final_test";
    } else {
      state.period = periods[periods.length - 1];
    }
  }
  fillSelect(element("periodSelect"), periods, state.period);
}

async function loadRuns() {
  const data = await fetchJson("/api/backtest/runs");
  state.runs = data.runs;
  if (state.runs.length === 0) {
    element("emptyBanner").hidden = false;
    element("emptyBanner").innerHTML =
      "Belum ada hasil backtest di <code>" + escapeHtml(data.runs_root) + "</code>. Jalankan " +
      "<code>uv run python -m technical.run walk-forward --models xgboost</code> (fold 2021–2025) dan/atau " +
      "<code>uv run python -m technical.run final-test --model xgboost --confirm</code> (uji 2026), lalu muat ulang halaman.";
    element("metrics").style.display = "none";
    showOverlay("Belum ada data backtest.");
    return;
  }
  state.runId = state.runs[0].run_id;
  refreshSelectors();
  await loadPeriod();
}

// ------------------------------------------------------------------ periode

function describePeriod(period, horizonMinutes) {
  if (period === "final_test") {
    return "Uji out-of-sample 2026: model dilatih dengan data sampai 2025-06-30, kalibrasi Juli–September 2025, threshold Oktober–Desember 2025. " +
      "Horizon " + horizonMinutes + " menit.";
  }
  const year = period.replace("validation_", "");
  return "Fold walk-forward " + year + ": model dilatih dengan data sebelum " + (Number(year) - 1) +
    "-07-01, kalibrasi Juli–September dan threshold Oktober–Desember " + (Number(year) - 1) + ", lalu diuji di " + year +
    ". Horizon " + horizonMinutes + " menit.";
}

async function loadSummary() {
  const lot = Number(element("lotInput").value) || 0.01;
  const leverage = Number(element("leverageInput").value) || 100;
  const summary = await fetchJson(baseUrl() + "/summary?lot=" + lot + "&leverage=" + leverage);
  state.summary = summary;
  renderMetrics(summary);
  return summary;
}

async function loadPeriod() {
  showOverlay("Memuat data periode…");
  state.selected = null;
  renderDetail(null);
  try {
    const summary = await loadSummary();
    const predictionData = await fetchJson(baseUrl() + "/predictions");
    state.predictions = [];
    state.byCandleTime = new Map();
    for (const values of predictionData.rows) {
      const row = {};
      for (let index = 0; index < predictionData.columns.length; index += 1) {
        row[predictionData.columns[index]] = values[index];
      }
      row.model_version = predictionData.model_version;
      row.candleTime = toEpochSeconds(row.candle_open_time);
      state.predictions.push(row);
      state.byCandleTime.set(row.candleTime, row);
    }

    const pill = element("periodPill");
    pill.className = "pill ok";
    pill.querySelector("span").textContent = state.period + " · " + (summary.model_version || state.model);
    element("periodNote").textContent = describePeriod(state.period, summary.horizon_minutes);

    const first = toEpochSeconds(summary.first_prediction);
    const last = toEpochSeconds(summary.last_prediction);
    element("startDate").min = dateInputValue(first);
    element("startDate").max = dateInputValue(last);
    element("endDate").min = dateInputValue(first);
    element("endDate").max = dateInputValue(last);
    const windowStart = Math.max(first, last - DEFAULT_WINDOW_DAYS * DAY_SECONDS);
    element("startDate").value = dateInputValue(windowStart);
    element("endDate").value = dateInputValue(last);

    state.page = 0;
    await loadCandles(windowStart, last + DAY_SECONDS);
    renderTable();
  } catch (error) {
    showOverlay("Gagal memuat periode: " + error.message);
  }
}

async function loadCandles(startSeconds, endSeconds) {
  showOverlay("Memuat candle…");
  const url = baseUrl() + "/candles?start=" + encodeURIComponent(isoFromEpoch(startSeconds)) +
    "&end=" + encodeURIComponent(isoFromEpoch(endSeconds));
  const data = await fetchJson(url);
  state.bars = data.bars;
  state.windowStart = startSeconds;
  state.windowEnd = endSeconds;
  chartParts.candles.setData(data.bars);
  const volumes = [];
  for (const bar of data.bars) {
    volumes.push(volumePoint(bar));
  }
  chartParts.volume.setData(volumes);
  setLegendFromBar(legend, data.bars[data.bars.length - 1]);
  applyMarkers();
  if (data.bars.length === 0) {
    showOverlay("Tidak ada candle pada rentang ini.");
  } else {
    showOverlay(null);
    // Tampilkan ±200 candle terakhir agar marker terbaca; geser/zoom untuk melihat sisanya.
    const lastIndex = data.bars.length - 1;
    chartParts.chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, lastIndex - 200), to: lastIndex + 5 });
  }
  if (element("rangeOnly").checked) {
    state.page = 0;
    renderTable();
  }
}

function applyMarkers() {
  const markers = [];
  for (const row of state.predictions) {
    if (row.candleTime < state.windowStart || row.candleTime >= state.windowEnd) {
      continue;
    }
    const isSelected = state.selected && state.selected.candleTime === row.candleTime;
    if (row.signal === "BUY") {
      markers.push({
        time: row.candleTime, position: "belowBar", shape: "arrowUp",
        color: isSelected ? COLORS.gold : COLORS.buy, size: isSelected ? 2 : 1,
        text: isSelected ? "BUY" : "",
      });
    } else {
      markers.push({
        time: row.candleTime, position: "aboveBar", shape: "arrowDown",
        color: isSelected ? COLORS.gold : COLORS.sell, size: isSelected ? 2 : 1,
        text: isSelected ? "SELL" : "",
      });
    }
  }
  if (state.selected) {
    const endTime = toEpochSeconds(state.selected.label_end_time) - 300;
    if (endTime >= state.windowStart && endTime < state.windowEnd) {
      markers.push({ time: endTime, position: "inBar", shape: "circle", color: COLORS.gold, text: "t+H" });
    }
  }
  markers.sort(function (a, b) { return a.time - b.time; });
  chartParts.candles.setMarkers(markers);
}

// ------------------------------------------------------------------ pilih sinyal

function clearPriceLines() {
  for (const line of state.priceLines) {
    chartParts.candles.removePriceLine(line);
  }
  state.priceLines = [];
}

function addPriceLine(price, color, title) {
  const line = chartParts.candles.createPriceLine({
    price: price, color: color, lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: title,
  });
  state.priceLines.push(line);
}

async function selectPrediction(row) {
  state.selected = row;
  const horizonSeconds = (state.summary ? state.summary.horizon_minutes : 30) * 60;
  if (row.candleTime < state.windowStart || row.candleTime >= state.windowEnd) {
    const start = row.candleTime - 2 * DAY_SECONDS;
    const end = row.candleTime + 2 * DAY_SECONDS;
    element("startDate").value = dateInputValue(start);
    element("endDate").value = dateInputValue(end - DAY_SECONDS);
    await loadCandles(start, end);
  }
  applyMarkers();
  clearPriceLines();
  const level = levels(row);
  addPriceLine(level.entry, COLORS.gold, "Entry " + (row.signal === "BUY" ? "ask" : "bid"));
  if (Number.isFinite(row.barrier)) {
    addPriceLine(level.takeProfit, COLORS.buy, "TP");
    addPriceLine(level.stopLoss, COLORS.sell, "SL");
  }
  chartParts.chart.timeScale().setVisibleRange({
    from: row.candleTime - 4 * 3600,
    to: row.candleTime + horizonSeconds + 4 * 3600,
  });
  renderDetail(row);
  highlightRow();
}

function renderDetail(row) {
  const panel = element("detailPanel");
  if (!row) {
    panel.innerHTML = '<p class="muted">Pilih sinyal di grafik atau di tabel.</p>';
    clearPriceLines();
    return;
  }
  const level = levels(row);
  let move = null;
  if (Number.isFinite(row.future_close)) {
    move = row.future_close - row.close;
  }
  let moveClass = "";
  if (move !== null) {
    moveClass = move >= 0 ? "pos" : "neg";
  }
  panel.innerHTML =
    '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">' +
    '<span class="signal-badge big-signal ' + row.signal + '">' + row.signal + "</span>" +
    '<span class="result ' + row.prediction_result + '">' + row.prediction_result + "</span>" +
    '<span class="result ' + row.direction_result + '">arah ' + row.direction_result + "</span></div>" +
    '<dl class="kv">' +
    "<dt>Waktu sinyal</dt><dd class='num'>" + formatTime(row.prediction_time) + "</dd>" +
    "<dt>Akhir horizon</dt><dd class='num'>" + formatTime(row.label_end_time) + "</dd>" +
    "<dt>Entry (" + (row.signal === "BUY" ? "Ask" : "Bid") + ")</dt><dd class='num'>" + formatPrice(level.entry) + "</dd>" +
    "<dt>Spread</dt><dd class='num'>" + formatPrice(row.spread) + "</dd>" +
    "<dt>TP / SL (bid)</dt><dd class='num'>" + formatPrice(level.takeProfit) + " / " + formatPrice(level.stopLoss) + "</dd>" +
    "<dt>Close t+H (bid)</dt><dd class='num'>" + formatPrice(row.future_close) + "</dd>" +
    "<dt>Perubahan</dt><dd class='num " + moveClass + "'>" + formatPrice(move) + " (" + formatPercent(row.actual_return, 3) + ")</dd>" +
    "<dt>Label aktual</dt><dd>" + escapeHtml(row.actual_label || "–") + " <span class='muted'>(" + escapeHtml(row.label_status) + ")</span></dd>" +
    "<dt>P(BUY/NT/SELL)</dt><dd class='num'>" + formatPercent(row.probability_buy) + " / " +
    formatPercent(row.probability_no_trade) + " / " + formatPercent(row.probability_sell) + "</dd>" +
    "<dt>Alasan</dt><dd class='muted'>" + escapeHtml(row.reason_code) + "</dd>" +
    "<dt>Model</dt><dd class='muted'>" + escapeHtml(row.model_version) + "</dd>" +
    "</dl>";
}

chartParts.chart.subscribeClick(function (param) {
  if (!param || !param.time) {
    return;
  }
  const row = state.byCandleTime.get(param.time);
  if (row) {
    selectPrediction(row);
  }
});

// ------------------------------------------------------------------ tabel

function filteredRows() {
  const rows = [];
  const rangeOnly = element("rangeOnly").checked;
  for (const row of state.predictions) {
    if (state.signalFilter !== "ALL" && row.signal !== state.signalFilter) {
      continue;
    }
    if (state.resultFilter !== "ALL" && row.prediction_result !== state.resultFilter) {
      continue;
    }
    if (rangeOnly && (row.candleTime < state.windowStart || row.candleTime >= state.windowEnd)) {
      continue;
    }
    rows.push(row);
  }
  return rows;
}

function renderTable() {
  const rows = filteredRows();
  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  if (state.page >= pageCount) {
    state.page = pageCount - 1;
  }
  const start = state.page * PAGE_SIZE;
  const pageRows = rows.slice(start, start + PAGE_SIZE);
  element("tableCount").textContent = rows.length + " sinyal BUY/SELL";
  element("pageInfo").textContent = "Halaman " + (state.page + 1) + " dari " + pageCount;
  element("prevPage").disabled = state.page === 0;
  element("nextPage").disabled = state.page >= pageCount - 1;

  const body = element("tableBody");
  if (pageRows.length === 0) {
    body.innerHTML = '<tr><td colspan="12" class="empty">Tidak ada sinyal untuk filter ini.</td></tr>';
    return;
  }
  let html = "";
  for (let index = 0; index < pageRows.length; index += 1) {
    const row = pageRows[index];
    let returnClass = "";
    if (Number.isFinite(row.actual_return)) {
      returnClass = row.actual_return >= 0 ? "pos" : "neg";
    }
    html += '<tr data-index="' + (start + index) + '" data-time="' + row.candleTime + '">' +
      "<td class='num'>" + formatTime(row.prediction_time) + "</td>" +
      "<td><span class='signal-badge " + row.signal + "'>" + row.signal + "</span></td>" +
      "<td class='num'>" + formatPercent(row.confidence) + "</td>" +
      "<td class='num'>" + formatPrice(entryReference(row)) + "</td>" +
      "<td class='num'>" + formatPrice(row.spread) + "</td>" +
      "<td class='num'>" + formatPrice(row.barrier) + "</td>" +
      "<td class='num'>" + formatPrice(row.future_close) + "</td>" +
      "<td class='num " + returnClass + "'>" + formatPercent(row.actual_return, 3) + "</td>" +
      "<td>" + escapeHtml(row.actual_label || "–") + "</td>" +
      "<td><span class='result " + row.prediction_result + "'>" + row.prediction_result + "</span></td>" +
      "<td><span class='result " + row.direction_result + "'>" + row.direction_result + "</span></td>" +
      "<td class='muted'>" + escapeHtml(row.model_version) + "</td></tr>";
  }
  body.innerHTML = html;
  highlightRow();
}

function highlightRow() {
  const rows = element("tableBody").querySelectorAll("tr[data-time]");
  for (const tableRow of rows) {
    const selected = state.selected && Number(tableRow.dataset.time) === state.selected.candleTime;
    tableRow.classList.toggle("selected", Boolean(selected));
  }
}

element("tableBody").addEventListener("click", function (event) {
  const tableRow = event.target.closest("tr[data-index]");
  if (!tableRow) {
    return;
  }
  const rows = filteredRows();
  const row = rows[Number(tableRow.dataset.index)];
  if (row) {
    selectPrediction(row);
    element("chart").scrollIntoView({ behavior: "smooth", block: "center" });
  }
});

function bindSegmented(containerId, onChange) {
  const container = element(containerId);
  container.addEventListener("click", function (event) {
    const button = event.target.closest("button[data-value]");
    if (!button) {
      return;
    }
    for (const other of container.querySelectorAll("button")) {
      other.classList.toggle("active", other === button);
    }
    onChange(button.dataset.value);
  });
}

bindSegmented("signalFilter", function (value) {
  state.signalFilter = value;
  state.page = 0;
  renderTable();
});
bindSegmented("resultFilter", function (value) {
  state.resultFilter = value;
  state.page = 0;
  renderTable();
});
element("rangeOnly").addEventListener("change", function () {
  state.page = 0;
  renderTable();
});
element("prevPage").addEventListener("click", function () {
  state.page -= 1;
  renderTable();
});
element("nextPage").addEventListener("click", function () {
  state.page += 1;
  renderTable();
});

// ------------------------------------------------------------------ metrik

function renderMetrics(summary) {
  element("metrics").style.display = "";
  const classification = summary.classification;
  const counts = summary.signal_counts;

  // 1. Klasifikasi
  let cards = "";
  if (classification) {
    cards += metricCard("Accuracy", formatPercent(classification.accuracy), summary.evaluated_rows + " candle berlabel");
    cards += metricCard("Macro F1 (3 kelas)", classification.macro_f1.toFixed(3), "directional F1 " + classification.directional_macro_f1.toFixed(3));
    cards += metricCard("Coverage sinyal", formatPercent(classification.signal_coverage), "BUY+SELL / semua candle");
  } else {
    cards += metricCard("Accuracy", "–", "belum ada label lengkap");
  }
  cards += metricCard("Jumlah prediksi", summary.rows.toLocaleString("en-US"),
    "BUY " + counts.BUY + " · SELL " + counts.SELL + " · NT " + counts.NO_TRADE);
  element("classificationCards").innerHTML = cards;

  if (classification) {
    let table = '<table style="margin-top:12px"><thead><tr><th>Kelas</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Support</th></tr></thead><tbody>';
    for (const name of ["BUY", "SELL", "NO_TRADE"]) {
      const metric = classification.per_class[name];
      table += "<tr style='cursor:default'><td><span class='signal-badge " + name + "'>" + name + "</span></td>" +
        "<td class='num'>" + formatPercent(metric.precision) + "</td>" +
        "<td class='num'>" + formatPercent(metric.recall) + "</td>" +
        "<td class='num'>" + metric.f1.toFixed(3) + "</td>" +
        "<td class='num'>" + metric.support + "</td></tr>";
    }
    table += "</tbody></table>";
    element("perClass").innerHTML = table;
    element("confusion").innerHTML = confusionTable(classification.confusion_matrix);
  } else {
    element("perClass").innerHTML = "";
    element("confusion").innerHTML = "";
  }

  // 2. Arah dan hasil sinyal
  const direction = summary.direction;
  const outcomes = summary.prediction_results;
  const decided = outcomes.CORRECT + outcomes.INCORRECT + outcomes.NEUTRAL;
  let directionCards = "";
  directionCards += metricCard("Hit rate arah", formatPercent(direction.ALL.hit_rate), direction.ALL.signals + " sinyal selesai");
  directionCards += metricCard("Precision sinyal", formatPercent(decided > 0 ? outcomes.CORRECT / decided : null),
    "CORRECT / (C+I+N)");
  directionCards += metricCard("Arah BUY", formatPercent(direction.BUY.hit_rate), direction.BUY.signals + " sinyal", COLORS.buy);
  directionCards += metricCard("Arah SELL", formatPercent(direction.SELL.hit_rate), direction.SELL.signals + " sinyal", COLORS.sell);
  element("directionCards").innerHTML = directionCards;

  const total = decided + outcomes.PENDING + outcomes.UNAVAILABLE;
  let split = '<div class="split">';
  const parts = [
    ["CORRECT", COLORS.buy], ["INCORRECT", COLORS.sell], ["NEUTRAL", COLORS.neutral],
    ["PENDING", COLORS.pending], ["UNAVAILABLE", "#607b86"],
  ];
  for (const part of parts) {
    const width = total > 0 ? (outcomes[part[0]] / total) * 100 : 0;
    split += '<i style="width:' + width + "%;background:" + part[1] + '"></i>';
  }
  split += '</div><div class="legend-row">';
  for (const part of parts) {
    split += '<span style="--c:' + part[1] + '">' + part[0] + " " + outcomes[part[0]] + "</span>";
  }
  split += "</div>";
  element("outcomeSplit").innerHTML = split;
  element("confidenceHist").innerHTML = histogram(summary.confidence_histogram);

  // 3. Simulasi
  const simulation = summary.simulation;
  let simulationCards = "";
  if (simulation.trades > 0) {
    const pnlColor = simulation.net_pnl >= 0 ? COLORS.buy : COLORS.sell;
    simulationCards += metricCard("Net P&L (USD)", formatPrice(simulation.net_pnl), simulation.trades + " transaksi", pnlColor);
    simulationCards += metricCard("Win rate", formatPercent(simulation.win_rate), "P&L > 0 setelah spread");
    simulationCards += metricCard("Profit factor", simulation.profit_factor === null ? "–" : simulation.profit_factor.toFixed(2), "gross profit / gross loss");
    simulationCards += metricCard("Max drawdown", formatPrice(simulation.max_drawdown), "USD, dari puncak equity", COLORS.sell);
  } else {
    simulationCards += metricCard("Transaksi", "0", "belum ada sinyal dengan horizon lengkap");
  }
  element("simulationCards").innerHTML = simulationCards;
  renderEquity(simulation.equity_curve || []);
  if (simulation.trades > 0) {
    const instrument = simulation.instrument;
    const reasons = simulation.exit_reasons;
    element("simulationNote").textContent =
      "Lot " + instrument.lot + " × kontrak " + instrument.contract_size + " oz, leverage 1:" + instrument.leverage +
      ". Keluar: TP " + (reasons.TP || 0) + ", SL " + (reasons.SL || 0) + ", waktu habis " + (reasons.TIME || 0) +
      ", dua sisi " + (reasons.BOTH_SAME_CANDLE || 0) + " (dianggap SL). Total biaya spread " +
      formatPrice(simulation.total_spread_cost) + " (harga). Simulasi tingkat candle, tanpa slippage/komisi/swap.";
  } else {
    element("simulationNote").textContent = "";
  }
}

function confusionTable(confusion) {
  const names = confusion.rows_actual_cols_predicted;
  const values = confusion.values;
  let maximum = 1;
  for (const row of values) {
    for (const value of row) {
      maximum = Math.max(maximum, value);
    }
  }
  let html = '<table class="cm"><tr><th>aktual ↓ / prediksi →</th>';
  for (const name of names) {
    html += "<th>" + name + "</th>";
  }
  html += "</tr>";
  for (let rowIndex = 0; rowIndex < names.length; rowIndex += 1) {
    html += "<tr><th>" + names[rowIndex] + "</th>";
    for (let columnIndex = 0; columnIndex < names.length; columnIndex += 1) {
      const value = values[rowIndex][columnIndex];
      const alpha = 0.08 + 0.6 * (value / maximum);
      let color = "119, 170, 255";
      if (rowIndex === columnIndex) {
        color = "38, 198, 160";
      } else if (names[rowIndex] !== "NO_TRADE" && names[columnIndex] !== "NO_TRADE") {
        color = "255, 107, 122";
      }
      html += '<td style="background:rgba(' + color + "," + alpha.toFixed(2) + ')">' + value + "</td>";
    }
    html += "</tr>";
  }
  html += "</table>";
  return html;
}

function histogram(data) {
  const outcomes = [["CORRECT", COLORS.buy], ["INCORRECT", COLORS.sell], ["NEUTRAL", COLORS.neutral]];
  let maximum = 1;
  for (let index = 0; index < data.CORRECT.length; index += 1) {
    const total = data.CORRECT[index] + data.INCORRECT[index] + data.NEUTRAL[index];
    maximum = Math.max(maximum, total);
  }
  let html = '<div class="hist">';
  for (let index = 0; index < data.CORRECT.length; index += 1) {
    html += '<div class="bar" title="' + formatPercent(data.edges[index], 0) + "–" + formatPercent(data.edges[index + 1], 0) +
      ": C " + data.CORRECT[index] + " · I " + data.INCORRECT[index] + " · N " + data.NEUTRAL[index] + '">';
    for (const outcome of outcomes) {
      const height = (data[outcome[0]][index] / maximum) * 120;
      html += '<i style="height:' + height + "px;background:" + outcome[1] + '"></i>';
    }
    html += "</div>";
  }
  html += '</div><div class="hist-axis"><span>' + formatPercent(data.edges[0], 0) + "</span><span>confidence kelas terpilih</span><span>" +
    formatPercent(data.edges[data.edges.length - 1], 0) + "</span></div>";
  return html;
}

function renderEquity(points) {
  const container = element("equityChart");
  if (!state.equity) {
    const chart = LightweightCharts.createChart(container, {
      autoSize: true,
      layout: { background: { color: "transparent" }, textColor: COLORS.text, fontSize: 10 },
      grid: { vertLines: { visible: false }, horzLines: { color: COLORS.grid } },
      rightPriceScale: { borderColor: COLORS.border },
      timeScale: { borderColor: COLORS.border },
      handleScroll: false,
      handleScale: false,
    });
    const series = chart.addAreaSeries({ lineWidth: 2, priceLineVisible: false });
    state.equity = { chart: chart, series: series };
  }
  let positive = true;
  if (points.length > 0 && points[points.length - 1].value < 0) {
    positive = false;
  }
  state.equity.series.applyOptions({
    lineColor: positive ? COLORS.buy : COLORS.sell,
    topColor: positive ? "rgba(38,198,160,0.28)" : "rgba(255,107,122,0.28)",
    bottomColor: "rgba(0,0,0,0)",
  });
  state.equity.series.setData(points);
  if (points.length > 0) {
    state.equity.chart.timeScale().fitContent();
  }
}

// ------------------------------------------------------------------ event

element("runSelect").addEventListener("change", function (event) {
  state.runId = event.target.value;
  refreshSelectors();
  loadPeriod();
});
element("modelSelect").addEventListener("change", function (event) {
  state.model = event.target.value;
  refreshSelectors();
  loadPeriod();
});
element("periodSelect").addEventListener("change", function (event) {
  state.period = event.target.value;
  loadPeriod();
});
element("applyRange").addEventListener("click", function () {
  const start = dateInputToEpoch(element("startDate").value, false);
  const end = dateInputToEpoch(element("endDate").value, true);
  if (start === null || end === null || start >= end) {
    showOverlay("Rentang tanggal tidak valid.");
    return;
  }
  loadCandles(start, end);
});
for (const id of ["lotInput", "leverageInput"]) {
  element(id).addEventListener("change", function () {
    if (state.runId) {
      loadSummary();
    }
  });
}

loadRuns().catch(function (error) {
  showOverlay("Gagal memuat daftar run: " + error.message);
});
