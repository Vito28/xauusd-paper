"""Prediksi 10.000 deret harga dengan LSTM murni NumPy.

Mendukung training bertahap: jalankan sebagian epoch, simpan checkpoint, lalu
lanjutkan epoch sisanya tanpa kehilangan bobot maupun state optimizer Adam.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_SIZE = 10_000


def create_dataset(path: Path, seed: int = 42) -> tuple[list[str], np.ndarray]:
    """Buat tepat 10.000 harga harian dengan tren, musim, dan noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(DATASET_SIZE, dtype=float)
    prices = (
        100_000
        + 15 * t
        + 5_000 * np.sin(2 * np.pi * t / 365.25)  # pola tahunan
        + 1_500 * np.sin(2 * np.pi * t / 30.5)    # pola bulanan
        + 600 * np.sin(2 * np.pi * t / 7)         # pola mingguan
        + rng.normal(0, 900, size=DATASET_SIZE)
    )
    prices = np.round(prices, 2)
    start = date(2000, 1, 1)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(DATASET_SIZE)]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tanggal", "harga"])
        writer.writerows(zip(dates, prices))
    return dates, prices


def load_dataset(path: Path) -> tuple[list[str], np.ndarray]:
    dates, prices = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dates.append(row["tanggal"])
            prices.append(float(row["harga"]))
    if len(prices) != DATASET_SIZE:
        raise ValueError(f"Dataset harus berisi {DATASET_SIZE:,} data, ditemukan {len(prices):,}")
    return dates, np.asarray(prices, dtype=float)


class MinMaxScaler:
    def fit(self, values: np.ndarray) -> "MinMaxScaler":
        self.minimum = float(np.min(values))
        self.maximum = float(np.max(values))
        self.span = self.maximum - self.minimum or 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.minimum) / self.span

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.span + self.minimum


def make_sequences(values: np.ndarray, window: int, start: int, stop: int):
    """Target berada pada indeks [start, stop); input adalah window sebelumnya."""
    x = np.asarray([values[i - window : i] for i in range(start, stop)])
    y = np.asarray([values[i] for i in range(start, stop)])
    return x[:, :, None], y


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


class LSTMRegressor:
    """LSTM satu lapis dan dense output, lengkap dengan BPTT dan Adam."""

    def __init__(self, hidden_size: int = 20, seed: int = 42):
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        fan_in = hidden_size + 1
        scale = math.sqrt(1.0 / fan_in)
        self.params = {
            name: rng.normal(0, scale, (hidden_size, fan_in))
            for name in ("Wf", "Wi", "Wo", "Wg")
        }
        self.params.update(
            {
                "bf": np.ones(hidden_size),  # forget bias membantu memori awal
                "bi": np.zeros(hidden_size),
                "bo": np.zeros(hidden_size),
                "bg": np.zeros(hidden_size),
                "Wy": rng.normal(0, math.sqrt(1 / hidden_size), (1, hidden_size)),
                "by": np.zeros(1),
            }
        )
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.step = 0

    def forward(self, sequence: np.ndarray):
        """Hitung LSTM maju persis sesuai persamaan matematisnya.

        z_t = [h_(t-1), x_t]
        f_t = sigmoid(W_f z_t + b_f)              forget gate
        i_t = sigmoid(W_i z_t + b_i)              input gate
        o_t = sigmoid(W_o z_t + b_o)              output gate
        g_t = tanh(W_g z_t + b_g)                 kandidat memori
        c_t = f_t * c_(t-1) + i_t * g_t           cell state
        h_t = o_t * tanh(c_t)                      hidden state
        y_hat = W_y h_T + b_y                      prediksi
        """
        p, h_size = self.params, self.hidden_size
        h = np.zeros(h_size)
        c = np.zeros(h_size)
        cache = []
        for x_t in sequence:
            # 1. Gabungkan hidden state lama dan input harga saat ini.
            z = np.concatenate((h, np.atleast_1d(x_t)))

            # 2. Empat transformasi affine dan fungsi aktivasi gate.
            f = sigmoid(p["Wf"] @ z + p["bf"])
            i = sigmoid(p["Wi"] @ z + p["bi"])
            o = sigmoid(p["Wo"] @ z + p["bo"])
            g = np.tanh(p["Wg"] @ z + p["bg"])

            # 3. Memori lama yang dipertahankan + informasi baru yang ditulis.
            c_new = f * c + i * g

            # 4. Bagian memori yang dikeluarkan menjadi hidden state.
            h_new = o * np.tanh(c_new)
            cache.append((z, f, i, o, g, c, c_new, h_new))
            h, c = h_new, c_new
        prediction = float((p["Wy"] @ h + p["by"])[0])
        return prediction, cache

    def loss_and_gradients(self, x_batch: np.ndarray, y_batch: np.ndarray):
        """MSE dan BPTT manual, divektorisasi pada dimensi batch.

        Vektorisasi tidak mengubah matematika LSTM. Setiap baris tetap sebuah
        sequence independen; operasi banyak sequence dihitung sekaligus agar
        10.000 data dapat dilatih dengan cepat.
        """
        p = self.params
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        batch_size = len(x_batch)
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros_like(h)
        cache = []

        # Forward: bentuk z adalah [batch, hidden+1].
        for t in range(x_batch.shape[1]):
            z = np.concatenate((h, x_batch[:, t, :]), axis=1)
            f = sigmoid(z @ p["Wf"].T + p["bf"])
            i = sigmoid(z @ p["Wi"].T + p["bi"])
            o = sigmoid(z @ p["Wo"].T + p["bo"])
            g = np.tanh(z @ p["Wg"].T + p["bg"])
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            cache.append((z, f, i, o, g, c, c_new))
            h, c = h_new, c_new

        prediction = (h @ p["Wy"].T + p["by"]).reshape(-1)
        error = prediction - y_batch
        loss = float(np.mean(error**2))

        # L = mean((y_hat-y)^2), maka dL/dy_hat = 2*error/N.
        dy = (2.0 * error / batch_size)[:, None]
        grads["Wy"] = dy.T @ h
        grads["by"] = np.sum(dy, axis=0)
        dh_next = dy @ p["Wy"]
        dc_next = np.zeros_like(h)

        # Backward dari timestep terakhir menuju pertama (BPTT).
        for z, f, i, o, g, c_prev, c_now in reversed(cache):
            tanh_c = np.tanh(c_now)
            do = dh_next * tanh_c
            dc = dc_next + dh_next * o * (1.0 - tanh_c**2)
            df, di, dg = dc * c_prev, dc * g, dc * i
            daf = df * f * (1.0 - f)
            dai = di * i * (1.0 - i)
            dao = do * o * (1.0 - o)
            dag = dg * (1.0 - g**2)
            dz = np.zeros_like(z)
            for gate, da in (("f", daf), ("i", dai), ("o", dao), ("g", dag)):
                grads[f"W{gate}"] += da.T @ z
                grads[f"b{gate}"] += np.sum(da, axis=0)
                dz += da @ p[f"W{gate}"]
            dh_next = dz[:, : self.hidden_size]
            dc_next = dc * f
        return loss, grads

    def trace(self, sequence: np.ndarray) -> list[dict[str, float]]:
        """Keluarkan jejak numerik agar isi LSTM dapat dilihat per timestep."""
        _, cache = self.forward(sequence)
        rows = []
        for step, (z, f, i, o, g, c_prev, c_now, h_now) in enumerate(cache, 1):
            # Nilai neuron pertama dipakai untuk contoh substitusi persamaan.
            af0 = float(self.params["Wf"][0] @ z + self.params["bf"][0])
            rows.append(
                {
                    "timestep": step,
                    "x_scaled": float(z[-1]),
                    "a_forget_neuron_0": af0,
                    "forget_neuron_0": float(f[0]),
                    "input_neuron_0": float(i[0]),
                    "output_neuron_0": float(o[0]),
                    "candidate_neuron_0": float(g[0]),
                    "cell_neuron_0": float(c_now[0]),
                    "hidden_neuron_0": float(h_now[0]),
                    "forget_mean": float(np.mean(f)),
                    "input_mean": float(np.mean(i)),
                    "output_mean": float(np.mean(o)),
                    "cell_mean": float(np.mean(c_now)),
                    "hidden_mean": float(np.mean(h_now)),
                    "cell_previous_neuron_0": float(c_prev[0]),
                }
            )
        return rows

    def adam_update(self, grads: dict[str, np.ndarray], lr: float, clip: float = 1.0):
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        if norm > clip:
            grads = {k: g * clip / (norm + 1e-8) for k, g in grads.items()}
        self.step += 1
        for key, grad in grads.items():
            self.m[key] = 0.9 * self.m[key] + 0.1 * grad
            self.v[key] = 0.999 * self.v[key] + 0.001 * grad**2
            m_hat = self.m[key] / (1.0 - 0.9**self.step)
            v_hat = self.v[key] / (1.0 - 0.999**self.step)
            self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def fit(self, x, y, epochs=50, batch_size=128, lr=0.008, epoch_offset=0):
        history = []
        for local_epoch in range(1, epochs + 1):
            global_epoch = epoch_offset + local_epoch
            # Seed per epoch membuat training 100 epoch identik dengan 50+50.
            rng = np.random.default_rng(123 + global_epoch)
            order = rng.permutation(len(x))
            losses = []
            for pos in range(0, len(x), batch_size):
                idx = order[pos : pos + batch_size]
                loss, grads = self.loss_and_gradients(x[idx], y[idx])
                self.adam_update(grads, lr)
                losses.append(loss)
            epoch_loss = float(np.mean(losses))
            history.append(epoch_loss)
            if local_epoch == 1 or global_epoch % 10 == 0 or local_epoch == epochs:
                print(
                    f"Epoch total {global_epoch:4d} "
                    f"(sesi {local_epoch:3d}/{epochs}) - MSE scaled: {epoch_loss:.6f}"
                )
        return history

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        outputs = []
        p = self.params
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size]
            h = np.zeros((len(batch), self.hidden_size))
            c = np.zeros_like(h)
            for t in range(batch.shape[1]):
                z = np.concatenate((h, batch[:, t, :]), axis=1)
                f = sigmoid(z @ p["Wf"].T + p["bf"])
                i = sigmoid(z @ p["Wi"].T + p["bi"])
                o = sigmoid(z @ p["Wo"].T + p["bo"])
                g = np.tanh(z @ p["Wg"].T + p["bg"])
                c = f * c + i * g
                h = o * np.tanh(c)
            outputs.extend((h @ p["Wy"].T + p["by"]).reshape(-1))
        return np.asarray(outputs)

    def save(self, path: Path):
        np.savez(path, hidden_size=self.hidden_size, **self.params)

    def save_checkpoint(self, path: Path, completed_epochs: int, window: int, history):
        content = {
            "hidden_size": self.hidden_size,
            "completed_epochs": completed_epochs,
            "window": window,
            "adam_step": self.step,
            "history": np.asarray(history),
        }
        content.update({f"param_{k}": v for k, v in self.params.items()})
        content.update({f"adam_m_{k}": v for k, v in self.m.items()})
        content.update({f"adam_v_{k}": v for k, v in self.v.items()})
        np.savez(path, **content)

    @classmethod
    def load_checkpoint(cls, path: Path):
        data = np.load(path)
        model = cls(hidden_size=int(data["hidden_size"]))
        for key in model.params:
            model.params[key] = data[f"param_{key}"].copy()
            model.m[key] = data[f"adam_m_{key}"].copy()
            model.v[key] = data[f"adam_v_{key}"].copy()
        model.step = int(data["adam_step"])
        return (
            model,
            int(data["completed_epochs"]),
            int(data["window"]),
            data["history"].astype(float).tolist(),
        )


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(np.mean(np.abs(error / actual)) * 100),
    }


def save_predictions(path: Path, dates, actual, predicted):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tanggal", "harga_aktual", "harga_prediksi", "error"])
        for date, act, pred in zip(dates, actual, predicted):
            writer.writerow([date, f"{act:.2f}", f"{pred:.2f}", f"{act-pred:.2f}"])


def save_history(path: Path, history: list[float]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "mse_scaled"])
        writer.writerows((i, f"{loss:.8f}") for i, loss in enumerate(history, 1))


def save_trace(path: Path, rows: list[dict[str, float]]):
    """Simpan semua nilai internal untuk audit/perhitungan ulang."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_process_html(path, trace, history, dates, actual, predicted, scaler):
    """Buat visualisasi interaktif mandiri tanpa library plotting."""
    payload = {
        "trace": trace,
        "loss": [{"epoch": i + 1, "value": v} for i, v in enumerate(history)],
        "prediction": [
            {"date": d[:7], "actual": round(float(a), 2), "predicted": round(float(p), 2)}
            for d, a, p in zip(dates, actual, predicted)
        ],
        "minimum": scaler.minimum,
        "maximum": scaler.maximum,
    }
    html = r'''<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visualisasi Proses LSTM dari Nol</title>
<style>
:root{font-family:system-ui,sans-serif;color:#172033;background:#f6f8fc}body{max-width:1100px;margin:auto;padding:24px}h1,h2{color:#102454}.panel{background:#fff;border:1px solid #dbe2ef;border-radius:12px;padding:18px;margin:16px 0}.flow{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.node{padding:10px 14px;border:1px solid #7690bd;border-radius:8px;background:#eef4ff}.arrow{font-size:22px;color:#65718a}.gates{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.gate{border-left:5px solid #517bc3;padding:9px;background:#f5f8ff}.track{height:9px;background:#e3e8f0;border-radius:8px;margin-top:7px}.fill{height:100%;background:#3868b0;border-radius:8px}.formula{font-family:ui-monospace,monospace;background:#111827;color:#e5edf8;padding:13px;border-radius:8px;overflow:auto;white-space:pre-wrap}svg{width:100%;height:260px}label{font-weight:600}input{width:100%}.muted{color:#5c677d;font-size:.92rem}.legend span{display:inline-block;margin-right:18px}.dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px}@media(max-width:600px){body{padding:12px}.arrow{transform:rotate(90deg)}}
</style></head><body>
<h1>Visualisasi Proses LSTM dari Nol</h1>
<p>Ikuti data dari harga asli hingga prediksi. Slider menampilkan isi internal LSTM pada satu sequence test.</p>
<section class="panel"><h2>1. Alur pemrosesan</h2><div class="flow">
<div class="node">10.000 harga</div><div class="arrow">→</div><div class="node">Split 80/20</div><div class="arrow">→</div><div class="node">Min–Max scaling</div><div class="arrow">→</div><div class="node">Sliding window</div><div class="arrow">→</div><div class="node">LSTM 20 neuron</div><div class="arrow">→</div><div class="node">Inverse scaling</div><div class="arrow">→</div><div class="node">Prediksi Rp</div>
</div><p class="muted" id="scaleText"></p></section>
<section class="panel"><h2>2. Isi LSTM per timestep</h2>
<label for="step">Timestep: <output id="stepOut">1</output> dari <span id="stepMax"></span></label><input id="step" type="range" min="1" value="1">
<div class="gates" id="gates"></div><h3>Substitusi angka pada neuron pertama</h3><div class="formula" id="calculation"></div>
<svg id="stateChart" role="img" aria-label="Perubahan rata-rata gate dan state sepanjang sequence"></svg></section>
<section class="panel"><h2>3. Training: MSE mengecil</h2><svg id="lossChart" role="img" aria-label="Kurva loss training"></svg></section>
<section class="panel"><h2>4. Hasil pada data test</h2><div class="legend"><span><i class="dot" style="background:#245cb0"></i>Aktual</span><span><i class="dot" style="background:#d34a4a"></i>Prediksi</span></div><svg id="predChart" role="img" aria-label="Harga aktual dan prediksi"></svg></section>
<script>
const data=__PAYLOAD__;const fmt=n=>Number(n).toLocaleString('id-ID',{maximumFractionDigits:4});
document.getElementById('scaleText').textContent=`Scaling hanya memakai train: (x − ${fmt(data.minimum)}) / (${fmt(data.maximum)} − ${fmt(data.minimum)}).`;
const step=document.getElementById('step'),stepOut=document.getElementById('stepOut'),gates=document.getElementById('gates'),calculation=document.getElementById('calculation');
step.max=data.trace.length;document.getElementById('stepMax').textContent=data.trace.length;
const gateDefs=[['forget_mean','Forget gate f̄'],['input_mean','Input gate ī'],['output_mean','Output gate ō'],['cell_mean','Cell state c̄'],['hidden_mean','Hidden state h̄']];
function updateStep(){const r=data.trace[+step.value-1];stepOut.value=r.timestep;gates.innerHTML=gateDefs.map(([k,n])=>`<div class="gate"><strong>${n}</strong><div>${fmt(r[k])}</div><div class="track"><div class="fill" style="width:${Math.min(100,Math.abs(r[k])*100)}%"></div></div></div>`).join('');calculation.textContent=`z_t = concat(h_(t-1), x_t),   x_t = ${fmt(r.x_scaled)}\na_f[0] = W_f[0] · z_t + b_f[0] = ${fmt(r.a_forget_neuron_0)}\nf_t[0] = sigmoid(${fmt(r.a_forget_neuron_0)}) = ${fmt(r.forget_neuron_0)}\nc_t[0] = f_t[0] × c_(t-1)[0] + i_t[0] × g_t[0]\n       = ${fmt(r.forget_neuron_0)} × ${fmt(r.cell_previous_neuron_0)} + ${fmt(r.input_neuron_0)} × ${fmt(r.candidate_neuron_0)}\n       = ${fmt(r.cell_neuron_0)}\nh_t[0] = o_t[0] × tanh(c_t[0]) = ${fmt(r.hidden_neuron_0)}`}
step.addEventListener('input',updateStep);updateStep();
function lineChart(id,series,yKeys,colors,xLabel,yLabel){const svg=document.getElementById(id),W=900,H=250,m={l:70,r:20,t:15,b:42};svg.setAttribute('viewBox',`0 0 ${W} ${H}`);const ys=series.flatMap(d=>yKeys.map(k=>d[k])),ymin=Math.min(...ys),ymax=Math.max(...ys),X=i=>m.l+i/(series.length-1||1)*(W-m.l-m.r),Y=v=>H-m.b-(v-ymin)/(ymax-ymin||1)*(H-m.t-m.b);let s=`<rect x="${m.l}" y="${m.t}" width="${W-m.l-m.r}" height="${H-m.t-m.b}" fill="none" stroke="#cbd5e1"/><text x="${W/2}" y="${H-5}" text-anchor="middle">${xLabel}</text><text x="15" y="${H/2}" transform="rotate(-90 15 ${H/2})" text-anchor="middle">${yLabel}</text><text x="${m.l-8}" y="${m.t+5}" text-anchor="end">${fmt(ymax)}</text><text x="${m.l-8}" y="${H-m.b}" text-anchor="end">${fmt(ymin)}</text>`;yKeys.forEach((k,j)=>{const pts=series.map((d,i)=>`${X(i)},${Y(d[k])}`).join(' ');s+=`<polyline points="${pts}" fill="none" stroke="${colors[j]}" stroke-width="2"/>`});svg.innerHTML=s}
lineChart('stateChart',data.trace,['forget_mean','input_mean','output_mean','cell_mean','hidden_mean'],['#245cb0','#e89526','#228b61','#8b5bb3','#d34a4a'],'Timestep','Nilai rata-rata');
lineChart('lossChart',data.loss,['value'],['#245cb0'],'Epoch','MSE scaled');
lineChart('predChart',data.prediction,['actual','predicted'],['#245cb0','#d34a4a'],'Hari test','Harga (Rp)');
</script></body></html>'''.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    path.write_text(html, encoding="utf-8")


def save_svg(path: Path, prices, split, test_pred, future_pred):
    width, height, pad = 1000, 520, 65
    all_values = np.concatenate((prices, future_pred))
    lo, hi = float(all_values.min()), float(all_values.max())
    x = lambda idx: pad + idx / (len(all_values) - 1) * (width - 2 * pad)
    y = lambda val: height - pad - (val - lo) / (hi - lo) * (height - 2 * pad)
    points = lambda vals, start: " ".join(
        f"{x(start+i):.1f},{y(v):.1f}" for i, v in enumerate(vals)
    )
    split_x = x(split)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Prediksi Harga dengan LSTM</text>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#555"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#555"/>
<line x1="{split_x:.1f}" y1="{pad}" x2="{split_x:.1f}" y2="{height-pad}" stroke="#999" stroke-dasharray="5,5"/>
<polyline points="{points(prices, 0)}" fill="none" stroke="#2563eb" stroke-width="2"/><polyline points="{points(test_pred, split)}" fill="none" stroke="#dc2626" stroke-width="2.5"/><polyline points="{points(np.r_[prices[-1], future_pred], len(prices)-1)}" fill="none" stroke="#16a34a" stroke-width="2.5" stroke-dasharray="6,4"/>
<text x="{pad}" y="{height-20}" font-family="sans-serif" font-size="13">biru: aktual | merah: prediksi test | hijau: forecast 10 bulan</text><text x="8" y="{pad}" font-family="sans-serif" font-size="12">{hi:,.0f}</text><text x="8" y="{height-pad}" font-family="sans-serif" font-size="12">{lo:,.0f}</text></svg>'''
    path.write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="LSTM harga end-to-end (NumPy)")
    parser.add_argument("--epochs", type=int, default=50, help="epoch yang dijalankan pada sesi ini")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--future", type=int, default=10)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--resume", action="store_true", help="lanjutkan dari checkpoint sebelumnya")
    args = parser.parse_args()

    data_path = ROOT / "dataset_harga_10000.csv"
    if args.regenerate or not data_path.exists():
        dates, prices = create_dataset(data_path)
        print(f"Dataset dibuat: {data_path.name} ({len(prices)} baris)")
    else:
        try:
            dates, prices = load_dataset(data_path)
        except ValueError:
            dates, prices = create_dataset(data_path)
            print(f"Dataset lama diganti: {data_path.name} ({len(prices)} baris)")

    split = int(len(prices) * 0.8)
    scaler = MinMaxScaler().fit(prices[:split])  # cegah data leakage
    scaled = scaler.transform(prices)
    x_train, y_train = make_sequences(scaled, args.window, args.window, split)
    x_test, y_test = make_sequences(scaled, args.window, split, len(prices))
    print(f"Train: {len(x_train)} sequence | Test: {len(x_test)} sequence")

    checkpoint_path = ROOT / "checkpoint_lstm.npz"
    if args.resume:
        if not checkpoint_path.exists():
            parser.error("--resume dipilih, tetapi checkpoint_lstm.npz belum ada")
        model, completed_epochs, saved_window, old_history = LSTMRegressor.load_checkpoint(checkpoint_path)
        if saved_window != args.window:
            parser.error(
                f"checkpoint memakai window={saved_window}; jalankan dengan --window {saved_window}"
            )
        print(f"Checkpoint dimuat: {completed_epochs} epoch telah selesai")
    else:
        model = LSTMRegressor(hidden_size=20, seed=42)
        completed_epochs, old_history = 0, []
        print("Training baru dimulai dari epoch 1")

    new_history = model.fit(
        x_train, y_train, epochs=args.epochs, batch_size=128, lr=0.008,
        epoch_offset=completed_epochs,
    )
    completed_epochs += args.epochs
    history = old_history + new_history
    model.save_checkpoint(checkpoint_path, completed_epochs, args.window, history)
    print(f"Checkpoint disimpan: total {completed_epochs} epoch")
    test_pred = scaler.inverse_transform(model.predict(x_test))
    actual = scaler.inverse_transform(y_test)
    scores = metrics(actual, test_pred)
    trace = model.trace(x_test[0])

    rolling = list(scaled[-args.window :])
    future_scaled = []
    for _ in range(args.future):
        next_value = model.predict(np.asarray(rolling[-args.window:])[None, :, None])[0]
        future_scaled.append(next_value)
        rolling.append(next_value)
    future_pred = scaler.inverse_transform(future_scaled)
    last_date = date.fromisoformat(dates[-1])
    future_dates = [(last_date + timedelta(days=i + 1)).isoformat() for i in range(args.future)]

    save_predictions(ROOT / "hasil_prediksi_test_10000.csv", dates[split:], actual, test_pred)
    save_predictions(ROOT / "prediksi_masa_depan_10000.csv", future_dates, np.full(args.future, np.nan), future_pred)
    save_history(ROOT / "riwayat_training_10000.csv", history)
    save_trace(ROOT / "jejak_internal_lstm_10000.csv", trace)
    save_process_html(ROOT / "visualisasi_proses_10000.html", trace, history, dates[split:], actual, test_pred, scaler)
    save_svg(ROOT / "grafik_prediksi_10000.svg", prices, split, test_pred, future_pred)
    model.save(ROOT / "model_lstm_10000.npz")

    print(f"\nEvaluasi data test ({len(actual):,} hari terakhir)")
    for name, value in scores.items():
        suffix = "%" if name == "MAPE" else ""
        print(f"{name:>4}: {value:,.2f}{suffix}")
    print("\nPrediksi masa depan")
    for prediction_date, price in zip(future_dates, future_pred):
        print(f"{prediction_date}: Rp {price:,.2f}")
    print("\nArtefak tersimpan: CSV dataset/hasil/history/trace, model, SVG, dan HTML interaktif")


if __name__ == "__main__":
    main()
