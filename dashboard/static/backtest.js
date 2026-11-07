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
const CANDLE_SECONDS = 300;

// Alasan posisi ditutup, sesuai technical/simulation.py.
const EXIT_TEXT = {
  TP: "TP tersentuh lebih dulu",
  SL: "SL tersentuh lebih dulu",
  TIME: "Waktu habis (close t+H)",
  BOTH_SAME_CANDLE: "TP & SL 1 candle → SL",
};
const EXIT_SHORT = { TP: "TP", SL: "SL", TIME: "WAKTU", BOTH_SAME_CANDLE: "2 SISI→SL" };
const EXIT_COLORS = { TP: COLORS.buy, SL: COLORS.sell, BOTH_SAME_CANDLE: "#c2505c", TIME: COLORS.neutral };
const EXIT_ORDER = ["TP", "SL", "BOTH_SAME_CANDLE", "TIME"];

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

function hasExit(row) {
  return Boolean(row.exit_reason);
}

function entrySide(row) {
  return row.signal === "BUY" ? "Ask" : "Bid";
}

function exitSide(row) {
  // BUY ditutup dengan menjual di Bid; SELL ditutup dengan membeli kembali di Ask.
  return row.signal === "BUY" ? "Bid" : "Ask";
}

function spreadShares(row) {
  // Pecahan (bukan persen): spread / harga buka dan spread / barrier (jarak TP).
  const entry = entryReference(row);
  return {
    toPrice: Number.isFinite(entry) && entry > 0 ? row.spread / entry : null,
    toBarrier: Number.isFinite(row.barrier) && row.barrier > 0 ? row.spread / row.barrier : null,
  };
}

function formatClock(value) {
  if (!value) {
    return "–";
  }
  const date = new Date(value);
  const pad = function (number) { return String(number).padStart(2, "0"); };
  return pad(date.getUTCHours()) + ":" + pad(date.getUTCMinutes());
}

function signedPrice(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "–";
  }
  return (value > 0 ? "+" : "") + formatPrice(value);
}

function signedPercent(value, digits) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "–";
  }
  return (value > 0 ? "+" : "") + formatPercent(value, digits);
}

function closeCandleText(row) {
  return formatClock(row.exit_candle_open_time) + "–" + formatClock(row.exit_time) + " UTC (t+" + row.exit_step + ")";
}

function closeTimeText(row) {
  // TP/SL tersentuh di DALAM candle; menit persisnya tidak diketahui dari data M5.
  if (row.exit_reason === "TIME") {
    return formatClock(row.exit_time) + " UTC (close candle)";
  }
  return "≤ " + formatClock(row.exit_time) + " UTC (di dalam candle)";
}

function lotValue() {
  return Number(element("lotInput").value) || 0.01;
}

function instrumentQuery() {
  const leverage = Number(element("leverageInput").value) || 100;
  return "lot=" + lotValue() + "&leverage=" + leverage;
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
    element("tradePanel").style.display = "none";
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
  const summary = await fetchJson(baseUrl() + "/summary?" + instrumentQuery());
  state.summary = summary;
  renderMetrics(summary);
  return summary;
}

async function loadPredictions() {
  // Baris BUY/SELL beserta waktu buka/tutup, P&L (sesuai lot), dan spread per transaksi.
  const predictionData = await fetchJson(baseUrl() + "/predictions?" + instrumentQuery());
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
}

async function refreshInstrument() {
  // Lot/leverage hanya mengubah skala P&L; daftar sinyal dan pilihan tetap sama.
  await loadSummary();
  await loadPredictions();
  renderTable();
  if (state.selected) {
    const row = state.byCandleTime.get(state.selected.candleTime);
    if (row) {
      state.selected = row;
      renderDetail(row);
    }
  }
}

async function loadPeriod() {
  showOverlay("Memuat data periode…");
  state.selected = null;
  renderDetail(null);
  try {
    const summary = await loadSummary();
    await loadPredictions();

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
        text: isSelected ? "BUKA BUY" : "",
      });
    } else {
      markers.push({
        time: row.candleTime, position: "aboveBar", shape: "arrowDown",
        color: isSelected ? COLORS.gold : COLORS.sell, size: isSelected ? 2 : 1,
        text: isSelected ? "BUKA SELL" : "",
      });
    }
  }
  if (state.selected) {
    // Marker tutup diletakkan pada candle tempat posisi ditutup (TP/SL/waktu habis).
    const selected = state.selected;
    let closeTime = toEpochSeconds(selected.label_end_time) - CANDLE_SECONDS;
    let closeLabel = "t+H";
    let closeColor = COLORS.gold;
    if (hasExit(selected)) {
      closeTime = toEpochSeconds(selected.exit_candle_open_time);
      closeLabel = "TUTUP " + EXIT_SHORT[selected.exit_reason];
      closeColor = EXIT_COLORS[selected.exit_reason];
    }
    if (closeTime >= state.windowStart && closeTime < state.windowEnd) {
      markers.push({ time: closeTime, position: "inBar", shape: "circle", color: closeColor, text: closeLabel });
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
  addPriceLine(level.entry, COLORS.gold, "Buka " + entrySide(row).toLowerCase());
  if (Number.isFinite(row.barrier)) {
    addPriceLine(level.takeProfit, COLORS.buy, "TP");
    addPriceLine(level.stopLoss, COLORS.sell, "SL");
  }
  // TP/SL: posisi tutup di garis TP/SL (bid; SELL membeli kembali di bid + spread).
  // Garis tutup terpisah hanya perlu untuk waktu habis.
  if (row.exit_reason === "TIME" && Number.isFinite(row.exit_price)) {
    addPriceLine(row.exit_price, COLORS.neutral, "Tutup " + exitSide(row).toLowerCase());
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
  const shares = spreadShares(row);
  let move = null;
  if (Number.isFinite(row.future_close)) {
    move = row.future_close - row.close;
  }
  let moveClass = "";
  if (move !== null) {
    moveClass = move >= 0 ? "pos" : "neg";
  }

  let badges = '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">' +
    '<span class="signal-badge big-signal ' + row.signal + '">' + row.signal + "</span>" +
    '<span class="result ' + row.prediction_result + '">' + row.prediction_result + "</span>";
  if (hasExit(row)) {
    badges += '<span class="result ' + row.exit_reason + '">tutup ' + EXIT_SHORT[row.exit_reason] + "</span>";
  }
  badges += "</div>";

  let html = badges;
  html += kvSection("Buka posisi", [
    ["Waktu buka", formatTime(row.prediction_time) + " UTC"],
    ["Candle keputusan", formatClock(row.candle_open_time) + "–" + formatClock(row.prediction_time) + " UTC"],
    ["Harga buka (" + entrySide(row) + ")", formatPrice(level.entry)],
    ["Confidence", formatPercent(row.confidence)],
  ]);
  html += kvSection("Target (harga bid)", [
    ["TP / SL", formatPrice(level.takeProfit) + " / " + formatPrice(level.stopLoss)],
    ["Barrier", formatPrice(row.barrier)],
    ["= k·ATR + spread", formatPrice(row.barrier - row.spread) + " + " + formatPrice(row.spread)],
  ]);

  if (hasExit(row)) {
    const pnlClass = row.pnl >= 0 ? "pos" : "neg";
    html += kvSection("Tutup posisi", [
      ["Candle tutup", closeCandleText(row)],
      ["Waktu tutup", closeTimeText(row)],
      ["Alasan", EXIT_TEXT[row.exit_reason]],
      ["Lama posisi", row.holding_minutes + " menit"],
      ["Harga tutup (" + exitSide(row) + ")", formatPrice(row.exit_price)],
    ]);
    html += kvSection("Hasil transaksi", [
      ["Gerak kotor", signedPrice(row.gross_move)],
      ["Biaya spread", "−" + formatPrice(row.spread)],
      ["Gerak bersih", "<span class='" + pnlClass + "'>" + signedPrice(row.net_move) + " (" + signedPercent(row.net_return, 3) + ")</span>"],
      ["P&amp;L (lot " + lotValue() + ")", "<span class='" + pnlClass + "'>" + signedPrice(row.pnl) + " USD</span>"],
    ]);
  } else {
    let status = "Horizon melewati gap pasar/data; tidak bisa dinilai";
    if (row.label_status === "PENDING") {
      status = "Belum selesai; menunggu " + formatClock(row.label_end_time) + " UTC";
    }
    html += kvSection("Tutup posisi", [["Status", escapeHtml(status)]]);
  }

  html += kvSection("Spread", [
    ["Spread buka", formatPrice(row.spread)],
    ["Spread / harga buka", formatPercent(shares.toPrice, 4)],
    ["Spread / barrier (jarak TP)", formatPercent(shares.toBarrier, 1)],
    ["Spread tutup (simulasi)", formatPrice(row.spread) + " (= buka)"],
    ["Spread candle tutup", hasExit(row) ? formatPrice(row.exit_candle_spread) + " (tercatat)" : "–"],
  ]);
  html += kvSection("Penilaian", [
    ["Label aktual", escapeHtml(row.actual_label || "–") + " <span class='muted'>(" + escapeHtml(row.label_status) + ")</span>"],
    ["Close t+H (bid)", formatPrice(row.future_close)],
    ["Perubahan t+H", "<span class='" + moveClass + "'>" + formatPrice(move) + " (" + formatPercent(row.actual_return, 3) + ")</span>"],
    ["Arah t+H", "<span class='result " + row.direction_result + "'>" + escapeHtml(row.direction_result) + "</span>"],
    ["P(BUY/NT/SELL)", formatPercent(row.probability_buy) + " / " + formatPercent(row.probability_no_trade) + " / " + formatPercent(row.probability_sell)],
    ["Alasan model", "<span class='muted'>" + escapeHtml(row.reason_code) + "</span>"],
    ["Model", "<span class='muted'>" + escapeHtml(row.model_version) + "</span>"],
  ]);
  panel.innerHTML = html;
}

function kvSection(title, items) {
  // items: [label, nilai HTML]; nilai dari data teks sudah di-escape oleh pemanggil.
  let html = '<h3 class="kv-title">' + title + '</h3><dl class="kv">';
  for (const item of items) {
    html += "<dt>" + item[0] + "</dt><dd class='num'>" + item[1] + "</dd>";
  }
  return html + "</dl>";
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
    const level = levels(row);
    const shares = spreadShares(row);

    let closeCell = "<span class='muted'>" + escapeHtml(row.prediction_result) + "</span>";
    let exitCell = "<span class='muted'>–</span>";
    let exitPriceCell = "–";
    let pnlCell = "–";
    let pnlClass = "";
    if (hasExit(row)) {
      closeCell = formatTime(row.exit_time) +
        "<span class='cell-sub'>t+" + row.exit_step + " · " + row.holding_minutes + " menit</span>";
      exitCell = "<span class='result " + row.exit_reason + "'>" + EXIT_SHORT[row.exit_reason] + "</span>";
      exitPriceCell = formatPrice(row.exit_price) + "<span class='cell-sub'>" + exitSide(row) + "</span>";
      pnlCell = signedPrice(row.pnl) + "<span class='cell-sub'>" + signedPercent(row.net_return, 3) + "</span>";
      pnlClass = row.pnl >= 0 ? "pos" : "neg";
    }

    html += '<tr data-index="' + (start + index) + '" data-time="' + row.candleTime + '">' +
      "<td class='num'>" + formatTime(row.prediction_time) + "</td>" +
      "<td><span class='signal-badge " + row.signal + "'>" + row.signal + "</span></td>" +
      "<td class='num'>" + formatPercent(row.confidence) + "</td>" +
      "<td class='num'>" + formatPrice(level.entry) + "<span class='cell-sub'>" + entrySide(row) + "</span></td>" +
      "<td class='num'>" + formatPrice(row.spread) + "<span class='cell-sub'>" +
      formatPercent(shares.toPrice, 4) + " · " + formatPercent(shares.toBarrier, 1) + "</span></td>" +
      "<td class='num'>" + formatPrice(level.takeProfit) + " / " + formatPrice(level.stopLoss) + "</td>" +
      "<td class='num'>" + closeCell + "</td>" +
      "<td>" + exitCell + "</td>" +
      "<td class='num'>" + exitPriceCell + "</td>" +
      "<td class='num " + pnlClass + "'>" + pnlCell + "</td>" +
      "<td><span class='result " + row.prediction_result + "'>" + row.prediction_result + "</span>" +
      "<span class='cell-sub'>label " + escapeHtml(row.actual_label || "–") + "</span></td>" +
      "<td><span class='result " + row.direction_result + "'>" + row.direction_result + "</span></td></tr>";
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
  element("tradePanel").style.display = "";
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
    element("simulationNote").textContent =
      "Lot " + instrument.lot + " × kontrak " + instrument.contract_size + " oz, leverage 1:" + instrument.leverage +
      ". Rincian alasan tutup dan biaya spread ada di panel 4. Simulasi tingkat candle, tanpa slippage/komisi/swap.";
  } else {
    element("simulationNote").textContent = "";
  }
  renderTradePanel(summary);
}

function renderTradePanel(summary) {
  const bars = summary.horizon_bars;
  const minutes = summary.horizon_minutes;
  const k = summary.barrier_atr_multiple;
  const rules = [
    "<b>Kapan model menilai:</b> setiap candle M5 yang sudah tutup (setiap 5 menit). Hanya BUY/SELL yang membuka posisi; NO_TRADE tidak membuka apa pun.",
    "<b>Buka:</b> di close candle keputusan. BUY dibeli di <b>Ask</b> = close (bid) + spread; SELL dijual di <b>Bid</b> = close.",
    "<b>Target:</b> barrier = " + k + " × ATR14 + spread. TP dan SL = close ± barrier pada harga bid.",
    "<b>Tutup:</b> posisi diawasi " + bars + " candle (" + minutes + " menit). Posisi tutup saat high/low pertama kali menyentuh TP atau SL. " +
      "Jika keduanya tersentuh di candle yang sama, dianggap SL. Jika tidak ada yang tersentuh, tutup di close candle ke-" + bars + " (waktu habis).",
    "<b>Spread saat tutup:</b> spread di titik sentuh tidak tercatat, jadi dianggap sama dengan spread saat buka. " +
      "BUY membayar spread saat buka, SELL saat tutup (beli kembali di Ask).",
    "<b>Hasil bersih:</b> TP = +" + k + "·ATR; SL = −(" + k + "·ATR + 2 × spread); waktu habis = selisih harga di close candle ke-" + bars + " dikurangi spread.",
    "<b>Posisi tumpang tindih diizinkan:</b> sinyal baru tetap membuka posisi walaupun posisi sebelumnya belum tutup.",
  ];
  let rulesHtml = "";
  for (const rule of rules) {
    rulesHtml += "<li>" + rule + "</li>";
  }
  element("tradeRules").innerHTML = rulesHtml;

  const simulation = summary.simulation;
  if (!simulation.trades) {
    element("exitSplit").innerHTML = '<p class="muted">Belum ada transaksi dengan horizon lengkap.</p>';
    element("spreadCards").innerHTML = "";
    element("spreadNote").textContent = "";
    return;
  }

  const counts = simulation.exit_reasons;
  const shares = simulation.exit_reason_share;
  let split = '<div class="split">';
  for (const reason of EXIT_ORDER) {
    split += '<i style="width:' + ((shares[reason] || 0) * 100) + "%;background:" + EXIT_COLORS[reason] + '"></i>';
  }
  split += '</div><div class="legend-row">';
  for (const reason of EXIT_ORDER) {
    split += '<span style="--c:' + EXIT_COLORS[reason] + '">' + EXIT_SHORT[reason] + " " + (counts[reason] || 0) +
      " (" + formatPercent(shares[reason] || 0) + ")</span>";
  }
  split += "</div>";
  element("exitSplit").innerHTML = split;

  const spread = simulation.spread;
  let cards = "";
  cards += metricCard("Spread rata-rata", formatPrice(spread.average),
    "median " + formatPrice(spread.median) + " · maks " + formatPrice(spread.maximum) + " (harga)");
  cards += metricCard("Spread / harga buka", formatPercent(spread.average_to_price, 4), "biaya relatif terhadap harga");
  cards += metricCard("Spread / jarak TP", formatPercent(spread.average_to_barrier, 1), "porsi barrier untuk menutup spread");
  cards += metricCard("Total biaya spread", formatPrice(spread.total_cost_usd) + " USD",
    simulation.trades + " transaksi · lot " + simulation.instrument.lot, COLORS.sell);
  cards += metricCard("P&amp;L kotor → bersih", signedPrice(spread.gross_pnl_before_spread) + " → " + signedPrice(spread.net_pnl_after_spread),
    "USD, sebelum → sesudah spread");
  cards += metricCard("Spread candle tutup", formatPrice(spread.average_exit_candle_spread),
    "rata-rata tercatat; simulasi memakai spread buka");
  cards += metricCard("Lama posisi rata-rata", simulation.average_holding_minutes.toFixed(1) + " mnt", "maksimal " + minutes + " menit");
  cards += metricCard("Posisi tumpang tindih", String(simulation.overlapping_trades),
    formatPercent(simulation.overlapping_share) + " transaksi dibuka saat posisi lain masih terbuka");
  element("spreadCards").innerHTML = cards;
  element("spreadNote").textContent =
    "Persentase spread adalah rata-rata per transaksi. Spread berasal dari candle MT5 akun demo dan belum tentu sama dengan akun live. " +
    "Total biaya spread = jumlah spread × lot × ukuran kontrak.";
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
      refreshInstrument().catch(function (error) {
        showOverlay("Gagal memuat ulang simulasi: " + error.message);
      });
    }
  });
}

loadRuns().catch(function (error) {
  showOverlay("Gagal memuat daftar run: " + error.message);
});
