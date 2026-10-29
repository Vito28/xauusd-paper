# Docker untuk FastAPI dan PostgreSQL 18

## Tujuan

Docker menyediakan runtime Linux yang reproducible untuk dashboard FastAPI,
migrasi Alembic, dan PostgreSQL 18. Stack ini ditujukan untuk development lokal
serta eksplorasi data; terminal MetaTrader 5 tetap berjalan di Windows host.

File utama:

- `Dockerfile`: image Python 3.12 dengan environment dependency yang dikelola
  `uv` dari `pyproject.toml` dan `uv.lock`.
- `compose.yaml`: service `api`, service `postgres`, health check, jaringan
  internal Compose, dan named volume database.
- `.dockerignore`: mengecualikan secret, virtual environment host, cache, model,
  dan report dari build context.
- `.env.example`: nilai konfigurasi development yang dapat disalin ke `.env`.

## Menjalankan stack

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Saat PostgreSQL sudah sehat, container API menjalankan migrasi Alembic sebelum
Uvicorn dimulai. Buka:

- dashboard: `http://127.0.0.1:8080/`
- Swagger: `http://127.0.0.1:8080/docs`
- ReDoc: `http://127.0.0.1:8080/redoc`

Lihat status dan log tanpa masuk ke container:

```powershell
docker compose ps
docker compose logs -f api
docker compose logs -f postgres
```

Hentikan container tanpa menghapus data:

```powershell
docker compose down
```

Named volume `postgres_data` dipasang pada `/var/lib/postgresql`, sesuai layout
image resmi PostgreSQL 18 yang menempatkan `PGDATA` pada subdirectory versi.

## Konfigurasi

Compose membaca `POSTGRES_DB`, `POSTGRES_USER`, dan `POSTGRES_PASSWORD` dari
`.env`. Nilai tersebut dipakai untuk membangun `GOLD_DATABASE_URL` dengan
hostname internal `postgres`; URL `localhost` dari `GOLD_DATABASE_URL` host
tidak diteruskan ke container.

Nilai default hanya cocok untuk development lokal. Ganti password sebelum
memakai stack pada jaringan bersama. Jangan commit `.env`.

## Boundary MetaTrader 5

Dependency `MetaTrader5` dibatasi ke Windows di `pyproject.toml`. Karena itu,
image Linux dapat menjalankan FastAPI, migrasi, dan akses PostgreSQL, tetapi
tidak dapat melakukan ingestion langsung dari terminal MT5. Jalankan command
collector pada Windows host dan arahkan `GOLD_DATABASE_URL` host ke port 5432
yang dipublikasikan Compose.

## Pemeriksaan konfigurasi

```powershell
docker compose config
docker compose build
```

`docker compose config` juga membantu menemukan variable kosong atau kesalahan
struktur YAML sebelum stack dinyalakan.
