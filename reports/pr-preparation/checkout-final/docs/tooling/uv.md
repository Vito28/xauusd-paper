# Tooling Khusus `uv`

## Mengapa `uv`

`uv` adalah satu-satunya project/dependency manager untuk collector ini. Ia
mengelola versi Python, virtual environment, resolusi dependency, lock file,
dev dependency, dan command execution. Konsistensi ini penting agar penelitian
dapat direproduksi pada mesin Windows lain.

File authoritative:

- `pyproject.toml`: metadata, direct dependency, script, konfigurasi test/lint.
- `uv.lock`: resolusi lengkap dan versi transitive yang reproducible.
- `.python-version`: versi Python yang dipilih untuk workspace.

Jangan menambah atau memelihara `requirements.txt` paralel.

## Bootstrap

```powershell
uv sync --dev
```

`--dev` memasang pytest, coverage, dan Ruff selain runtime dependency. Untuk
environment deployment yang hanya menjalankan collector:

```powershell
uv sync --no-dev --frozen
```

`--frozen` menolak perubahan resolusi yang belum tercatat di lock file.

## Menambah atau menghapus dependency

```powershell
uv add <package>
uv add --dev <package>
uv remove <package>
uv lock
```

Jalankan `uv sync` setelah checkout bila environment belum sinkron. Commit
`pyproject.toml` dan `uv.lock` bersama-sama.

## Menjalankan aplikasi

Selalu jalankan command dalam managed environment:

```powershell
uv run alembic upgrade head
uv run python -m market.collector --help
uv run python -m market.collector mt5-info --profile example_demo
uv run python -m market.collector ingest-bars --help
```

Project script ekuivalen juga tersedia:

```powershell
uv run market --help
```

## Quality workflow

```powershell
uv run ruff format .
uv run ruff check .
uv run pytest --cov=market --cov=database
uv run python scripts/generate_code_reference.py
uv run alembic upgrade head --sql
```

Migration offline tetap memerlukan `GOLD_ENVIRONMENT` dan
`GOLD_DATABASE_URL`, tetapi tidak membuka koneksi dan tidak mengakses MT5.

## Update yang terkendali

```powershell
uv lock --upgrade-package sqlalchemy
uv sync
uv run pytest
```

Hindari upgrade seluruh dependency tanpa review. Periksa perubahan `uv.lock`,
jalankan semua quality checks, lalu uji migration pada PostgreSQL disposable.
