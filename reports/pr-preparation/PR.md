Project sebelumnya baru memiliki eksperimen LSTM dan konfigurasi setup. Perubahan ini menambahkan pipeline market data XAUUSD dari MetaTrader 5 ke PostgreSQL, analytics, dan dashboard FastAPI.

- Collector bars/ticks dengan profile broker, validasi, partition UTC, retry, checkpoint, dan audit rejection.
- Schema serta migrasi Alembic; repository penyimpanan dan view processed M5.
- Pembacaan analytics read-only, EDA, analisis time series, dan eksperimen forecast dengan pemisahan waktu.
- UI serta API dashboard dengan filter, sorting, pagination, dan pilihan environment.
- Dokumentasi arsitektur, alur data, tooling, dan referensi AST sesuai source yang dikirim.

Refs #2, #3, #4, #5, #6, #7, #8.

Validasi: 64 unit tests lulus dan 1 test integrasi PostgreSQL tidak dijalankan pada salinan bersih hasil git archive; Ruff lulus; mkdocs build --strict lulus; git diff --check lulus. Dokumentasi AST mencakup 14/14 modul dan 217/217 komponen publik.

Draft karena koneksi terminal MT5 dan PostgreSQL nyata belum diverifikasi. Modul technical, XGBoost matematis mandiri, artefak model, serta laporan lokal disiapkan terpisah.

Base: main. Head: codex/market-data-dashboard.
