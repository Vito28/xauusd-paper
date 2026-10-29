# Market — profile dan collector

Dua file utama memakai section `# %%` seperti notebook/Colab. Penanda section
memisahkan kelompok fungsi; pemanggilan proses tetap eksplisit melalui fungsi
atau CLI. Import modul tidak membaca runtime settings, membuka terminal MT5,
atau melakukan transaksi PostgreSQL.

```text
market/
  profile.py        profile broker, loader/saver TOML, dan GOLD_* runtime settings
  collector.py      seluruh workflow acquisition, validation, coverage, dan CLI
  contracts.py      record bersama untuk collector dan repository database
  dashboard.py      browser/API read-only; template berada di static/
  cli.py            penghubung untuk console script/perintah lama
  __init__.py       identitas package
```

`contracts.py` menjaga record data tetap independen dari collector dan database.
Dashboard tetap menjadi modul API dengan dokumentasi OpenAPI. `cli.py` hanya
mengekspor app dari `collector.py`; tidak menyimpan implementasi collection.

## Profile dan penulisan konfigurasi

Profile tersedia: `example_demo` dan `example_live` sebagai template,
`metaquotes_demo` untuk MetaQuotes-Demo, serta `finex_live` untuk
FinexBisnisSolusi-Real. Profile hanya berisi identitas broker/server/simbol
dan policy collection; credential berasal dari environment dan terminal MT5.
Pilih profile yang sesuai dengan environment dan account terminal aktif.

Section `profile.py`: import → schema profile broker → runtime settings →
penulisan konfigurasi. `load_broker_profile()` membaca TOML;
`save_broker_profile()` memvalidasi lalu menulis satu profile TOML.
Nama profile harus berupa stem file. File yang sudah ada tidak diganti kecuali
`overwrite=True` diberikan secara eksplisit. Runtime credential tetap melalui
environment/terminal; profile menyimpan identitas source dan policy non-secret.

Contoh membuat profile dari template yang tersedia:

```python
from market.profile import BrokerProfile, load_broker_profile, save_broker_profile

fields = load_broker_profile("example_demo").model_dump()
fields["source"].update(
    source_key="broker-baru-demo-xauusd",
    broker_name="Broker Baru",
    server_name="BrokerBaru-Demo",
    requested_symbol="XAUUSD",
)
profile = BrokerProfile.model_validate(fields)
path = save_broker_profile("broker_baru_demo", profile)
```

Contoh tersebut hanya menulis `configs/brokers/broker_baru_demo.toml`.
Pendaftaran source dan ingestion merupakan operasi CLI yang terpisah.
Seluruh profile TOML yang sudah ada tetap memakai format sebelumnya.

## Section collector

| Section | Proses | Komentar di source |
|---|---|---|
| 01 | Import dependensi | Import tidak mengaktifkan MT5/database |
| 02 | Resolusi exact symbol | `SYMBOL-*` |
| 03 | MT5 provider, preflight, metadata, fetch bars/ticks | `MT5-*` |
| 04 | Partition bulanan bars dan harian ticks dalam UTC | `PARTITION-*` |
| 05 | Normalisasi, fingerprint, validasi OHLC/quote dan rejection | `VALIDATE-*` |
| 06 | Derived aggregation M5 dan candidate gaps | `DERIVED-*` |
| 07 | Ingestion, resume, retry, transaksi, audit, checkpoint | `INGEST-*` |
| 08 | Coverage requested/actual, status run, invariant report | `REPORT-*` |
| 09 | Logging JSON dan redaksi credential | `LOG-*` |
| 10 | Register, inspect, backfill, sync, reports | `CLI-*` |

Alur utama: profile → MT5 preflight → partition → fetch → normalisasi/validasi →
PostgreSQL observations/rejections/checkpoint → coverage/status/validation report.
Data, rejection dan checkpoint disimpan atomik per partition; harga tidak
di-fill atau diinterpolasi. Laporan membaca PostgreSQL tanpa membuka MT5.

## Menjalankan

```powershell
uv run python -m market.collector --help
uv run python -m market.collector mt5-info --profile example_demo
uv run python -m market.collector ingest-bars --help
uv run python -m market.collector coverage --profile example_demo
```

Console script `uv run market ...` dan perintah lama `python -m market.cli ...`
tetap mengarah ke app yang sama. Dashboard tetap dijalankan melalui
`uv run python -m market.dashboard`.

## Verifikasi

```powershell
uv run pytest tests/market tests/database -m "not integration" -q
uv run ruff check market database tests/market tests/database
uv run python scripts/generate_code_reference.py
```

Unit tests memakai fake provider/session dan tidak memanggil terminal/database
nyata. Test integrasi PostgreSQL tetap membutuhkan database disposable yang
ditentukan melalui `GOLD_TEST_DATABASE_URL`.
