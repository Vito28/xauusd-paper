// Utilitas bersama dashboard live dan backtest.

const COLORS = {
  buy: "#26c6a0",
  sell: "#ff6b7a",
  neutral: "#b8c4cc",
  pending: "#f4bd55",
  gold: "#f4bd55",
  text: "#91aab4",
  grid: "rgba(65, 97, 110, 0.18)",
  border: "#223d49",
};

function formatPrice(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "–";
  }
  if (digits === undefined) {
    digits = 2;
  }
  return Number(value).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatPercent(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "–";
  }
  if (digits === undefined) {
    digits = 1;
  }
  return (value * 100).toFixed(digits) + "%";
}

function formatTime(value) {
  if (!value) {
    return "–";
  }
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  const pad = function (number) { return String(number).padStart(2, "0"); };
  return date.getUTCFullYear() + "-" + pad(date.getUTCMonth() + 1) + "-" + pad(date.getUTCDate()) +
    " " + pad(date.getUTCHours()) + ":" + pad(date.getUTCMinutes());
}

function toEpochSeconds(isoText) {
  return Math.floor(new Date(isoText).getTime() / 1000);
}

function escapeHtml(text) {
  const element = document.createElement("span");
  element.textContent = text === null || text === undefined ? "" : String(text);
  return element.innerHTML;
}

// Grafik candlestick + volume dengan gaya gelap seperti terminal trading.
function createCandleChart(container) {
  const chart = LightweightCharts.createChart(container, {
    autoSize: true,
    layout: { background: { color: "transparent" }, textColor: COLORS.text, fontSize: 11 },
    grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: COLORS.border },
    timeScale: { borderColor: COLORS.border, timeVisible: true, secondsVisible: false, rightOffset: 6 },
    localization: { priceFormatter: function (price) { return formatPrice(price, 2); } },
  });
  const candles = chart.addCandlestickSeries({
    upColor: COLORS.buy,
    downColor: COLORS.sell,
    wickUpColor: COLORS.buy,
    wickDownColor: COLORS.sell,
    borderVisible: false,
  });
  const volume = chart.addHistogramSeries({
    priceScaleId: "",
    priceFormat: { type: "volume" },
    color: "rgba(119, 170, 255, 0.35)",
  });
  volume.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
  candles.priceScale().applyOptions({ scaleMargins: { top: 0.08, bottom: 0.22 } });
  return { chart: chart, candles: candles, volume: volume };
}

function volumePoint(bar) {
  let color = "rgba(255, 107, 122, 0.35)";
  if (bar.close >= bar.open) {
    color = "rgba(38, 198, 160, 0.35)";
  }
  return { time: bar.time, value: bar.volume || 0, color: color };
}

function setLegendFromBar(legend, bar) {
  if (!bar) {
    legend.innerHTML = "";
    return;
  }
  legend.innerHTML =
    "<b>" + formatTime(bar.time) + " UTC</b> &nbsp; O <b>" + formatPrice(bar.open) +
    "</b> H <b>" + formatPrice(bar.high) + "</b> L <b>" + formatPrice(bar.low) +
    "</b> C <b>" + formatPrice(bar.close) + "</b>";
}

function attachLegend(chartParts, legend, getLastBar) {
  chartParts.chart.subscribeCrosshairMove(function (param) {
    if (!param || !param.time || !param.seriesData) {
      setLegendFromBar(legend, getLastBar());
      return;
    }
    const data = param.seriesData.get(chartParts.candles);
    if (data) {
      setLegendFromBar(legend, Object.assign({ time: param.time }, data));
    }
  });
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch (error) {
      // respons bukan JSON
    }
    throw new Error(detail);
  }
  return response.json();
}
