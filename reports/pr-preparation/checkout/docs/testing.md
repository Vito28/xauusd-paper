# Strategi Testing

## Unit tests default

`uv run pytest` tidak memerlukan terminal MT5 atau PostgreSQL. Provider preflight
memakai fake API, sedangkan validasi, partition, fingerprint, gap, aggregation,
logging, dan static safety guard diuji sebagai fungsi murni.

Guardrail AST memastikan production source tidak memiliki fungsi eksekusi
trading, call trading API, atau persistence CSV/Parquet/pickle/SQLite. Test juga
memastikan seluruh file Python memiliki module docstring.

## PostgreSQL integration

Gunakan database disposable terpisah:

```powershell
$env:GOLD_ENVIRONMENT = "demo"
$env:GOLD_DATABASE_URL = "postgresql+psycopg://user:pass@localhost/gold_market_test"
$env:GOLD_TEST_DATABASE_URL = $env:GOLD_DATABASE_URL
uv run alembic upgrade head
uv run pytest -m integration
```

Jangan gunakan `gold_market_demo` atau `gold_market_live` sebagai test database.
Integration test memverifikasi schema dan view hasil migration, dan dapat
ditambah untuk repository concurrency/idempotency pada environment CI.

## Real terminal validation

Test otomatis tidak mengklaim koneksi terminal nyata berhasil. Pada Windows
yang memiliki terminal dan account broker, jalankan berurutan:

```powershell
uv run python -m market.collector mt5-info --profile example_demo
uv run python -m market.collector symbol-info --profile example_demo
uv run python -m market.collector source-register --profile example_demo
```

Verifikasi output environment, company, server, actual symbol, dan warning
history sebelum memulai range backfill besar.
