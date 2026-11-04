// Dashboard live: WebSocket /ws/live, fallback grafik historis PostgreSQL.

// Library grafik dimuat dari CDN; tampilkan pesan jelas jika gagal (misalnya offline).
if (typeof LightweightCharts === "undefined") {
  document.querySelector("main").insertAdjacentHTML(
    "afterbegin",
    '<div class="banner">Library grafik (unpkg.com/lightweight-charts) gagal dimuat. Periksa koneksi internet lalu muat ulang halaman.</div>'
  );
  throw new Error("LightweightCharts tidak tersedia");
}

const state = {
  socket: null,
  retryDelay: 1000,
  bars: [],
  source: null, // "mt5" | "postgres" | null
  historyRequested: false,
  status: null,
  predictor: null,
};

const chartParts = createCandleChart(document.getElementById("chart"));
const legend = document.getElementById("legend");
attachLegend(chartParts, legend, function () { return state.bars[state.bars.length - 1]; });

function setPill(element, kind, text) {
  element.className = "pill " + kind;
  element.querySelector("span").textContent = text;
}

function setBanner(text, kind) {
  const banner = document.getElementById("sourceBanner");
  if (!text) {
    banner.hidden = true;
    return;
  }
  banner.hidden = false;
  banner.className = "banner" + (kind ? " " + kind : "");
  banner.textContent = text;
}

function showOverlay(text) {
  const overlay = document.getElementById("chartOverlay");
  if (text === null) {
    overlay.hidden = true;
    return;
  }
  overlay.hidden = false;
  document.getElementById("chartOverlayText").textContent = text;
}

// ---------------------------------------------------------------- grafik

function setAllBars(bars, source) {
  state.bars = bars;
  state.source = source;
  chartParts.candles.setData(bars);
  const volumes = [];
  for (const bar of bars) {
    volumes.push(volumePoint(bar));
  }
  chartParts.volume.setData(volumes);
  setLegendFromBar(legend, bars[bars.length - 1]);
  if (bars.length > 0) {
    showOverlay(null);
    chartParts.chart.timeScale().scrollToRealTime();
  }
  let label = "–";
  if (source === "mt5") {
    label = "sumber: MT5 live";
  } else if (source === "postgres") {
    label = "sumber: PostgreSQL historis (bukan live)";
  }
  document.getElementById("chartSource").textContent = label;
}

function updateBars(bars) {
  for (const bar of bars) {
    const last = state.bars[state.bars.length - 1];
    if (last && bar.time < last.time) {
      continue;
    }
    if (last && bar.time === last.time) {
      state.bars[state.bars.length - 1] = bar;
    } else {
      state.bars.push(bar);
    }
    chartParts.candles.update(bar);
    chartParts.volume.update(volumePoint(bar));
  }
  setLegendFromBar(legend, state.bars[state.bars.length - 1]);
}

async function loadHistoryFallback() {
  if (state.historyRequested) {
    return;
  }
  state.historyRequested = true;
  try {
    const data = await fetchJson("/api/live/history?limit=500");
    if (state.source !== "mt5") {
      if (data.bars.length === 0) {
        showOverlay("PostgreSQL terhubung, tetapi belum ada candle M5 untuk XAUUSD.");
      } else {
        setAllBars(data.bars, "postgres");
      }
    }
  } catch (error) {
    if (state.source !== "mt5") {
      showOverlay("MT5 tidak terhubung dan data historis tidak tersedia: " + error.message);
    }
  }
}

// ---------------------------------------------------------------- panel

function renderTick(tick) {
  if (!tick) {
    document.getElementById("bid").textContent = "–";
    document.getElementById("ask").textContent = "–";
    document.getElementById("spread").textContent = "–";
    document.getElementById("spreadPoints").textContent = "–";
    document.getElementById("tickTime").textContent = "–";
    return;
  }
  document.getElementById("bid").textContent = formatPrice(tick.bid);
  document.getElementById("ask").textContent = formatPrice(tick.ask);
  document.getElementById("spread").textContent = formatPrice(tick.spread);
  document.getElementById("spreadPoints").textContent = Math.round(tick.spread * 100) + " point (0,01)";
  document.getElementById("tickTime").textContent = formatTime(tick.time);
}

function renderStatus(status) {
  state.status = status;
  document.getElementById("symbol").textContent = status.symbol;
  document.getElementById("timeframe").textContent = "Timeframe " + status.timeframe;
  const statusText = document.getElementById("mt5Status");
  const reasonText = document.getElementById("mt5Reason");
  const pill = document.getElementById("mt5Pill");
  if (status.connected) {
    statusText.textContent = "Terhubung";
    statusText.style.color = COLORS.buy;
    reasonText.textContent = "update " + formatTime(status.last_update);
    setPill(pill, "ok", "MT5 live");
    setBanner(null);
  } else {
    statusText.textContent = "Tidak terhubung";
    statusText.style.color = COLORS.sell;
    reasonText.textContent = status.reason || "–";
    setPill(pill, "bad", "MT5 terputus");
    setBanner(
      "MetaTrader 5 tidak terhubung (" + (status.reason || "alasan tidak diketahui") +
      "). Harga Bid/Ask live tidak ditampilkan; grafik memakai data historis PostgreSQL jika tersedia."
    );
  }
}

function probabilityRows(probabilities) {
  const order = [["BUY", COLORS.buy], ["NO_TRADE", COLORS.neutral], ["SELL", COLORS.sell]];
  let html = "";
  for (const item of order) {
    const name = item[0];
    const value = probabilities ? probabilities[name] : null;
    const width = value === null || value === undefined ? 0 : value * 100;
    html += '<div class="prob-row"><span>' + name + '</span><div class="prob-bar"><i style="width:' +
      width.toFixed(1) + "%;background:" + item[1] + '"></i></div><span class="num">' +
      formatPercent(value) + "</span></div>";
  }
  return html;
}

function renderPrediction(predictor, predictions) {
  state.predictor = predictor;
  const panel = document.getElementById("predictionPanel");
  if (predictor && predictor.horizon_minutes) {
    document.getElementById("predHorizon").textContent = "horizon " + predictor.horizon_minutes + " menit";
  }
  if (!predictor || !predictor.available) {
    panel.innerHTML =
      '<span class="result UNAVAILABLE">UNAVAILABLE</span>' +
      '<p class="note">' + escapeHtml(predictor ? predictor.reason : "Model belum dimuat") + "</p>" +
      probabilityRows(null);
    return;
  }
  if (!predictions || predictions.length === 0) {
    panel.innerHTML =
      '<span class="result PENDING">MENUNGGU CANDLE</span>' +
      '<p class="note">Model ' + escapeHtml(predictor.model_version) +
      " siap. Prediksi pertama dibuat ketika candle M5 berikutnya selesai dan MT5 terhubung.</p>" +
      probabilityRows(null);
    return;
  }
  const latest = predictions[predictions.length - 1];
  let headline = '<span class="signal-badge big-signal ' + latest.signal + '">' + latest.signal + "</span>";
  if (latest.signal === "NO_TRADE") {
    headline += '<p class="note">Model tidak memberi sinyal transaksi untuk candle ini.</p>';
  }
  let reference = formatPrice(latest.bid_reference) + " (bid)";
  if (latest.signal === "BUY") {
    reference = formatPrice(latest.ask_reference) + " (ask)";
  }
  panel.innerHTML =
    headline + probabilityRows(latest.probabilities) +
    '<dl class="kv">' +
    "<dt>Harga referensi</dt><dd class='num'>" + reference + "</dd>" +
    "<dt>Spread</dt><dd class='num'>" + formatPrice(latest.spread) + "</dd>" +
    "<dt>Barrier (±)</dt><dd class='num'>" + formatPrice(latest.barrier) + "</dd>" +
    "<dt>Waktu prediksi</dt><dd class='num'>" + formatTime(latest.prediction_time) + "</dd>" +
    "<dt>Target evaluasi</dt><dd class='num'>" + formatTime(latest.target_time) + "</dd>" +
    "<dt>Status</dt><dd><span class='result " + latest.status + "'>" + latest.status + "</span></dd>" +
    "<dt>Model</dt><dd class='muted'>" + escapeHtml(latest.model_version) + "</dd>" +
    "</dl>";
}

function renderHistory(predictions) {
  const body = document.getElementById("liveHistory");
  if (!predictions || predictions.length === 0) {
    body.innerHTML = '<tr><td colspan="4" class="muted">Belum ada prediksi.</td></tr>';
    return;
  }
  let html = "";
  for (let index = predictions.length - 1; index >= 0; index -= 1) {
    const item = predictions[index];
    let status = item.status;
    let statusClass = item.status;
    if (item.status === "EVALUATED" && item.signal !== "NO_TRADE") {
      if (item.actual_label === item.signal) {
        status = "BENAR";
        statusClass = "CORRECT";
      } else if (item.actual_label === "NO_TRADE") {
        status = "NETRAL";
        statusClass = "NEUTRAL";
      } else {
        status = "SALAH";
        statusClass = "INCORRECT";
      }
    }
    html += "<tr><td class='num'>" + formatTime(item.prediction_time) + "</td>" +
      "<td><span class='signal-badge " + item.signal + "'>" + item.signal + "</span></td>" +
      "<td class='num'>" + formatPercent(item.confidence) + "</td>" +
      "<td><span class='result " + statusClass + "'>" + status + "</span></td></tr>";
  }
  body.innerHTML = html;
}

// ---------------------------------------------------------------- websocket

function handleMessage(message) {
  renderStatus(message.status);
  renderTick(message.status.connected ? message.tick : null);
  renderPrediction(message.predictor, message.predictions);
  renderHistory(message.predictions);

  const hasBars = message.bars && message.bars.length > 0;
  if (message.status.connected && hasBars) {
    if (message.type === "snapshot" || state.source !== "mt5") {
      setAllBars(message.bars, "mt5");
    } else {
      updateBars(message.bars);
    }
  } else if (!message.status.connected) {
    loadHistoryFallback();
  }
}

function connect() {
  const protocol = window.location.protocol === "https:" ? "wss://" : "ws://";
  const socket = new WebSocket(protocol + window.location.host + "/ws/live");
  state.socket = socket;
  setPill(document.getElementById("socketPill"), "warn", "Menghubungkan…");

  socket.onopen = function () {
    state.retryDelay = 1000;
    setPill(document.getElementById("socketPill"), "ok", "Server terhubung");
  };
  socket.onmessage = function (event) {
    try {
      handleMessage(JSON.parse(event.data));
    } catch (error) {
      console.error("Pesan WebSocket tidak valid", error);
    }
  };
  socket.onclose = function () {
    setPill(document.getElementById("socketPill"), "bad", "Server terputus");
    setPill(document.getElementById("mt5Pill"), "", "MT5: –");
    setBanner("Koneksi ke server dashboard terputus. Mencoba lagi dalam " +
      Math.round(state.retryDelay / 1000) + " detik…");
    setTimeout(connect, state.retryDelay);
    state.retryDelay = Math.min(state.retryDelay * 2, 30000);
  };
  socket.onerror = function () {
    socket.close();
  };
}

connect();
