# Database — schema dan storage

Implementasi persistence dipusatkan ke dua file berdasarkan tanggung jawab.
Keduanya memakai section `# %%` dan komentar proses seperti notebook.

```text
database/
  schema.py          struktur tabel, kolom, relasi, constraints, dan ORM metadata
  storage.py         engine/session/transaksi dan seluruh operasi repository
  __init__.py        dokumentasi package
  migrations/        runtime, template, dan riwayat revisi Alembic
```

## Schema

`schema.py` mendefinisikan struktur record PostgreSQL tanpa membuka koneksi atau
menjalankan DDL saat import. `Base.metadata` digunakan oleh Alembic.

| Section | Isi |
|---|---|
| 01 | `Base`: metadata root |
| 02 | `DataSource`: provenance broker/server/environment/symbol |
| 03 | `SymbolSpecVersion`: versi spesifikasi symbol menurut payload hash |
| 04 | `IngestionRun`: status, rentang waktu, counter dan error collection |
| 05 | `Checkpoint`: status resume per partition |
| 06 | `BarObservationModel`: observasi OHLC append-only |
| 07 | `TickObservationModel`: observasi tick dengan hash/occurrence |
| 08 | `Rejection`: audit payload invalid dan reason code |

## Storage

`storage.py` mendefinisikan operasi koneksi dan penyimpanan. Repository
menerima `Session` dari caller; commit/rollback dilakukan pada batas transaksi
collector, bukan diam-diam di setiap repository.

| Section | Isi | Penanda komentar |
|---|---|---|
| 01 | Import/dependensi | Import tidak melakukan runtime I/O |
| 02 | `get_engine`, `Database`: lazy engine, session, transaksi dan cleanup | `CONNECT-*`, `TRANSACTION-*` |
| 03 | `SourceRepository`: register source, verifikasi identity, versi symbol | `SOURCE-*` |
| 04 | `MarketDataRepository`: partition dan bulk insert bars/ticks | `STORE-*` |
| 05 | `IngestionRepository`: start/finish run dan upsert checkpoint | `CHECKPOINT-*` |
| 06 | `RejectionRepository`: penyimpanan record invalid | `AUDIT-*` |

Alur: normalized records → repository → transaksi caller → PostgreSQL.
Insert bars/ticks tetap memakai batch maksimal 1.000 baris dan conflict handling
yang sama. Rejection mempertahankan INSERT SQLAlchemy generik melalui alias
`generic_insert`; source/observations/checkpoint memakai INSERT PostgreSQL.

Contoh import baru:

```python
from database.schema import Base, IngestionRun
from database.storage import Database, MarketDataRepository, get_engine
```

`market.collector`, dashboard, analytics, tests dan Alembic memakai jalur baru.
Fingerprint symbol tetap diimpor saat `save_symbol_spec()` dipanggil agar
storage dan collector dapat diimpor dari urutan mana pun.

## Migrasi dan verifikasi

`migrations/` mempertahankan file revisi Alembic secara terpisah. Struktur tabel,
constraints, schema SQL, revision ID dan hubungan antar-revisi dipertahankan;
penggabungan source Python tidak membutuhkan migrasi database baru.
`migrations/env.py` menggunakan `database.schema.Base.metadata`.

```powershell
uv run alembic history
uv run pytest -m "not integration" -q
uv run ruff check database market analytics tests
```

Test PostgreSQL online tetap menggunakan `GOLD_TEST_DATABASE_URL` ke database
disposable; unit tests dan pemeriksaan import tidak membuka database nyata.
