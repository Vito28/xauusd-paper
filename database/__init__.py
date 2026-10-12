"""PostgreSQL persistence package for raw and derived market data.

The database is the platform's only market-data source of truth.  This package
contains two implementation modules: database.schema for ORM tables/relations,
and database.storage for engine/session lifecycle and repository operations.
Alembic's versioned migration files remain under database/migrations.
"""
