# Code Reference

> Generated from source AST. Do not edit manually.

## Documentation Coverage

- Modules: 20/20
- Public classes/functions/methods: 288/288

## `analisis_teknikall/xgboost_matematis.py`

XGBoost matematis dalam satu file, hanya standard library Python.

- **Function** `f32` `(value: float) -> float` (line 41)
  - Round one value to IEEE-754 binary32, matching tree input/accumulation precision.
- **Function** `softmax` `(margins: list[float], single: bool) -> list[float]` (line 46)
  - Convert additive class margins into probabilities using stable exp(z-max(z)).
- **Function** `sigmoid` `(value: float) -> float` (line 57)
  - Compute logistic probability without overflowing large negative margins.
- **Function** `soft_threshold` `(gradient: float, alpha: float) -> float` (line 65)
  - Apply L1 regularization: sign(G) * max(abs(G)-alpha, 0).
- **Function** `validate_matrix` `(x: list[list[float]], y: list[float | int]) -> None` (line 70)
  - Require aligned finite targets and a rectangular matrix; NaN features are allowed.
- **Class** `TreeConfig` (line 88)
  - Controls for educational exact-split training, independently of technical tuning.
- **Class** `Node` (line 105)
  - A learned scalar leaf or numeric decision node with explicit missing-value routing.
- **Method** `Node.predict` `(self: Missing, row: list[float]) -> float` (line 115)
  - Traverse stored splits until a leaf; equality follows the right branch.
- **Function** `grow_tree` `(x: list[list[float]], gradients: list[float], hessians: list[float], config: TreeConfig) -> Node` (line 128)
  - Fit an exact greedy Newton tree with prefix sums and no external ML implementation.
- **Class** `ScratchRegressor` (line 213)
  - Train squared or pseudo-Huber Newton boosting manually using exact numeric splits.
- **Method** `ScratchRegressor.fit` `(self: Missing, x: list[list[float]], y: list[float]) -> ScratchRegressor` (line 237)
  - Learn from supplied TRAIN rows only; reset previous state on every fit.
- **Method** `ScratchRegressor.predict` `(self: Missing, x: list[list[float]]) -> list[float]` (line 258)
  - Sum learned scalar leaves with learning-rate shrinkage to obtain numeric predictions.
- **Class** `ScratchClassifier` (line 275)
  - Train three-class softmax Newton boosting independently from scalar regression.
- **Method** `ScratchClassifier.fit` `(self: Missing, x: list[list[float]], y: list[int]) -> ScratchClassifier` (line 288)
  - Fit multiclass cross entropy from supplied TRAIN data; require all three classes.
- **Method** `ScratchClassifier.predict_proba` `(self: Missing, x: list[list[float]]) -> list[list[float]]` (line 311)
  - Produce uncalibrated SELL/NO_TRADE/BUY probabilities from learned margins.
- **Method** `ScratchClassifier.predict` `(self: Missing, x: list[list[float]]) -> list[int]` (line 326)
  - Return first argmax class; this raw classifier is not a validated trading policy.
- **Class** `FrozenTrees` (line 337)
  - Read native XGBoost JSON and evaluate numeric scalar-leaf gbtree ensembles manually.
- **Method** `FrozenTrees.margins` `(self: Missing, features: dict[str, float]) -> list[float]` (line 379)
  - Sum leaves; reject absent columns and route explicit NaN through default branches.
- **Method** `FrozenTrees.predict` `(self: Missing, features: dict[str, float]) -> list[float]` (line 398)
  - Apply softmax only to the classifier; regression/quantile outputs remain numeric.
- **Function** `choose_signal` `(probabilities: list[float], point: float, lower: float, upper: float, cost: float, policy: dict[str, Any]) -> tuple[str, str]` (line 410)
  - Reproduce frozen technical abstention rules without using any actual outcome.
- **Class** `TechnicalMath` (line 439)
  - Combine separate classifier, regressor and quantile predictions with frozen metadata.
- **Method** `TechnicalMath.predict` `(self: Missing, features: dict[str, float], close: float, spread: float) -> dict[str, Any]` (line 466)
  - Return calibrated probabilities, return/price forecasts and one policy decision.
- **Function** `outcome` `(signal: str, close: float, actual: float | None, spread: float, slippage_bps: float) -> dict[str, Any]` (line 525)
  - Evaluate actual after inference in price units, independently of account P/L.
- **Function** `read_feature_rows` `(path: Path) -> list[dict[str, str]]` (line 548)
  - Read named feature rows; empty numeric cells are explicit missing observations.
- **Function** `predict_csv` `(model: TechnicalMath, path: Path) -> list[dict[str, Any]]` (line 554)
  - Predict CSV features; labels are excluded from input and indicators are not recomputed.
- **Function** `verify` `(model: TechnicalMath, inputs: Path, reference: Path) -> dict[str, Any]` (line 586)
  - Compare manual outputs with independent technical reference, failing on any tolerance breach.
- **Function** `demo` `() -> dict[str, Any]` (line 638)
  - Train both scratch models on synthetic data, explicitly separate from market evaluation.
- **Function** `main` `() -> None` (line 658)
  - Run pure-Python training demo, manual inference or strict reference verification.

## `analytics/analytics.py`

Notebook-style UNI, BI, MULTI, time-series and model analysis.

- **Function** `validate_columns` `(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]` (line 91)
  - Return explicit columns after rejecting missing or unsafe names.
- **Function** `validate_feature_selection` `(frame: pd.DataFrame, features: Sequence[str]) -> list[str]` (line 106)
  - Reject metadata, targets, IDs, and future information as predictors.
- **Function** `safe_output_name` `(value: str) -> str` (line 122)
  - Convert a column label into a safe filename stem.
- **Function** `save_plot` `(output_path: Path) -> Path` (line 131)
  - Save the current figure as a non-empty PNG, replacing a prior report.
- **Function** `numeric_values` `(frame: pd.DataFrame, column: str) -> pd.Series` (line 152)
  - Return a finite numeric copy for one plot without mutating source data.
- **Function** `paired_values` `(frame: pd.DataFrame, x_column: str, y_column: str) -> pd.DataFrame` (line 163)
  - Return finite X/Y pairs, dropping NaN only for the requested pair.
- **Function** `plot_time_series` `(frame: pd.DataFrame, timestamp_column: str, value_column: str, output_path: Path, title: str | None, x_label: str | None, y_label: str | None, resample_rule: str | None) -> Path` (line 179)
  - Plot one numeric variable against UTC time without filling missing rows.
- **Function** `plot_histogram` `(frame: pd.DataFrame, column: str, output_path: Path, bins: int, title: str | None, x_label: str | None, y_label: str) -> Path` (line 220)
  - Plot a finite-value histogram while leaving the source frame intact.
- **Function** `plot_boxplot` `(frame: pd.DataFrame, column: str, output_path: Path, title: str | None, y_label: str | None) -> Path` (line 243)
  - Plot a numeric boxplot without clipping outliers or filling NaN.
- **Function** `plot_missing_values` `(frame: pd.DataFrame, columns: Sequence[str], output_path: Path) -> Path` (line 261)
  - Plot missing counts and percentages for explicit columns without imputation.
- **Function** `cleanup_legacy_univariate_outputs` `(output_dir: Path) -> None` (line 285)
  - Remove obsolete charts so the report directory holds only the ten current PNGs.
- **Function** `report_uninformative_real_volume` `(frame: pd.DataFrame) -> None` (line 301)
  - Log when broker real volume is absent or almost entirely zero.
- **Function** `plot_close_time_series` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 322)
  - Plot daily close points for a legible view of the M5 price history.
- **Function** `plot_close_histogram` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 337)
  - Plot the unscaled distribution of closing prices.
- **Function** `plot_close_boxplot` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 350)
  - Plot closing-price quartiles without treating outliers as invalid data.
- **Function** `plot_return_histogram` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 362)
  - Plot percentage-return distribution without clipping extreme movements.
- **Function** `plot_return_boxplot` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 375)
  - Plot percentage-return quartiles without automatic outlier removal.
- **Function** `plot_tick_volume_histogram` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 387)
  - Plot the distribution of broker-reported tick activity.
- **Function** `plot_tick_volume_boxplot` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 400)
  - Plot tick-volume quartiles without removing active-market observations.
- **Function** `plot_spread_price_histogram` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 412)
  - Plot the raw distribution of bid-ask spread prices.
- **Function** `plot_spread_price_boxplot` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 425)
  - Plot spread-price quartiles without treating high spreads as invalid.
- **Function** `run_univariate_analysis` `(frame: pd.DataFrame, output_dir: Path) -> list[Path]` (line 437)
  - Create the fixed ten-chart XAU/USD M5 univariate EDA report.
- **Function** `plot_scatter` `(frame: pd.DataFrame, x_column: str, y_column: str, output_path: Path) -> Path` (line 470)
  - Plot valid numeric pairs as a diagnostic, not a causal conclusion.
- **Function** `plot_grouped_boxplot` `(frame: pd.DataFrame, value_column: str, group_column: str, output_path: Path) -> Path` (line 482)
  - Compare a numeric value across non-empty categories using boxplots.
- **Function** `plot_correlation_matrix` `(frame: pd.DataFrame, columns: Sequence[str], output_path: Path) -> Path` (line 509)
  - Plot a diagnostic Pearson correlation matrix for explicit columns.
- **Function** `plot_feature_histogram` `(frame: pd.DataFrame, feature: str, output_path: Path) -> Path` (line 536)
  - Plot one explicitly approved feature distribution.
- **Function** `plot_feature_boxplot` `(frame: pd.DataFrame, feature: str, output_path: Path) -> Path` (line 543)
  - Plot one explicitly approved feature boxplot.
- **Function** `plot_feature_missing_values` `(frame: pd.DataFrame, features: Sequence[str], output_path: Path) -> Path` (line 550)
  - Plot missing counts for explicit predictor features without filling.
- **Function** `plot_feature_correlation` `(frame: pd.DataFrame, features: Sequence[str], output_path: Path) -> Path` (line 559)
  - Plot feature correlations without automatic feature removal.
- **Function** `plot_feature_by_target` `(frame: pd.DataFrame, feature: str, target_column: str, output_path: Path) -> Path` (line 568)
  - Compare a feature across SELL, NO_TRADE, and BUY target classes.
- **Function** `plot_feature_drift` `(reference_frame: pd.DataFrame, current_frame: pd.DataFrame, feature: str, output_path: Path) -> Path` (line 592)
  - Compare reference and current distributions without mutation.
- **Function** `validate_evaluation_split` `(frame: pd.DataFrame) -> str` (line 617)
  - Require predictions to contain exactly one evaluation split.
- **Function** `plot_confusion_matrix` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 627)
  - Plot target versus prediction counts for one evaluation split.
- **Function** `plot_classification_metrics` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 646)
  - Plot per-class precision, recall, and F1 for one split.
- **Function** `plot_probability_histograms` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 673)
  - Plot SELL, NO_TRADE, and BUY probability distributions.
- **Function** `plot_feature_importance` `(importance_frame: pd.DataFrame, output_path: Path, top_n: int) -> Path` (line 689)
  - Plot top frozen-model feature importance values as horizontal bars.
- **Function** `plot_accuracy_by_month` `(frame: pd.DataFrame, output_path: Path) -> Path` (line 710)
  - Plot accuracy by UTC month for one split; accuracy is not profit.
- **Function** `split_training_and_actual` `(daily_close: pd.DataFrame, forecast_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]` (line 735)
  - Split model-visible history from ground truth with a strict UTC cutoff.
- **Function** `build_lagged_training_data` `(history: pd.DataFrame, lag_count: int, forecast_start: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]` (line 763)
  - Create prior-close lag features from history strictly before 2026.
- **Function** `train_xgboost_forecaster` `(history: pd.DataFrame, lag_count: int, random_state: int, model_parameters: Mapping[str, Any] | None, forecast_start: pd.Timestamp) -> tuple[Any, list[str]]` (line 792)
  - Fit XGBoost only from pre-2026 daily prices.
- **Function** `forecast_recursively` `(model: Any, history: pd.DataFrame, forecast_dates: Sequence[pd.Timestamp], lag_count: int, forecast_start: pd.Timestamp) -> pd.DataFrame` (line 842)
  - Produce 2026 prices using only prior history and prior predictions.
- **Function** `forecast_dates_for_actual` `(actual: pd.DataFrame) -> pd.DatetimeIndex` (line 882)
  - Return business-day prediction dates through the latest actual date.
- **Function** `align_actual_and_forecast` `(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame` (line 892)
  - Join actual and forecast prices on exact dates and compute residuals.
- **Function** `calculate_forecast_metrics` `(comparison: pd.DataFrame) -> dict[str, float]` (line 904)
  - Calculate MAE, RMSE, and MAPE for aligned actual and forecast prices.
- **Function** `create_interactive_dashboard` `(history: pd.DataFrame, forecast: pd.DataFrame, actual: pd.DataFrame, comparison: pd.DataFrame, metrics: Mapping[str, float], output_path: Path, forecast_start: pd.Timestamp) -> Path` (line 927)
  - Write a self-contained Plotly dashboard with zoom, pan, and hover.
- **Function** `plot_historical_forecast_actual` `(history: pd.DataFrame, forecast: pd.DataFrame, actual: pd.DataFrame, output_path: Path) -> Path` (line 1027)
  - Save the principal blue-history, red-forecast, green-actual time series.
- **Function** `plot_actual_vs_predicted` `(comparison: pd.DataFrame, output_path: Path) -> Path` (line 1068)
  - Save an aligned 2026 actual-versus-XGBoost-forecast line chart.
- **Function** `plot_residuals` `(comparison: pd.DataFrame, output_path: Path) -> Path` (line 1088)
  - Save actual-minus-forecast residuals over the reporting period.
- **Function** `plot_monthly_comparison` `(comparison: pd.DataFrame, output_path: Path) -> Path` (line 1102)
  - Save monthly mean actual and forecast closes for 2026.
- **Function** `plot_error_distribution` `(comparison: pd.DataFrame, output_path: Path, bins: int) -> Path` (line 1123)
  - Save the distribution of actual-minus-forecast residuals.
- **Function** `generate_forecast_reports` `(source_id: str, symbol: str, timeframe: str, history_start: str, actual_end: str, lag_count: int, output_dir: Path) -> tuple[dict[str, Path], dict[str, float]]` (line 1138)
  - Load data, forecast without leakage, and write PNG/metric report artifacts.
- **Function** `plot_trend_drawdown` `(daily: pd.DataFrame, output_path: Path) -> Path` (line 1293)
  - Compare price levels, relative growth and observed drawdown over UTC sessions.
- **Function** `plot_return_volatility` `(daily: pd.DataFrame, output_path: Path) -> Path` (line 1314)
  - Show session returns and trailing observed-return volatility without annualising.
- **Function** `plot_monthly_return_heatmap` `(daily: pd.DataFrame, output_path: Path) -> Path` (line 1331)
  - Plot month-close returns with absent months blank and an explicit sample-end label.
- **Function** `plot_yearly_return_distribution` `(daily: pd.DataFrame, output_path: Path) -> Path` (line 1365)
  - Compare return distributions across years with observed sample counts.
- **Function** `plot_return_dependence` `(daily: pd.DataFrame, output_path: Path, max_lag: int) -> Path` (line 1385)
  - Diagnose return memory and volatility clustering over observed-session lags.
- **Function** `plot_activity_relationships` `(native: pd.DataFrame, output_path: Path) -> Path` (line 1411)
  - Use density plots for native activity/spread versus absolute log returns.
- **Function** `plot_return_feature_correlation` `(daily: pd.DataFrame, output_path: Path) -> Path` (line 1435)
  - Plot pairwise Spearman correlations and counts for scale-relative daily features.
- **Function** `plot_observed_coverage` `(native: pd.DataFrame, output_path: Path) -> Path` (line 1473)
  - Show observed candle counts by UTC month without assuming a trading calendar.
- **Function** `summarize_time_series` `(native: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]` (line 1495)
  - Report measured changes, declining sessions, drawdowns and observed yearly counts.
- **Function** `evaluate_return_models` `(daily: pd.DataFrame, folds: int, holdout_start: pd.Timestamp) -> pd.DataFrame` (line 1535)
  - Compare fixed Ridge models and baselines on future observed-session returns.
- **Function** `plot_return_model_comparison` `(metrics: pd.DataFrame, output_path: Path) -> Path` (line 1627)
  - Plot each temporal split's RMSE, preserving the distinction from the final holdout.
- **Function** `run_time_series_analysis` `(frame: pd.DataFrame, output_dir: Path, timeframe: str, include_models: bool) -> list[Path]` (line 1643)
  - Generate eight scientific figures and measured JSON insights from isolated source copies.
- **Function** `run_time_series` `(arguments: argparse.Namespace) -> list[Path]` (line 1678)
  - Read one source and report observed time-series behaviour in a source-specific folder.
- **Function** `run_evaluation` `(arguments: argparse.Namespace) -> list[Path]` (line 1693)
  - Load one frozen prediction split and generate its evaluation reports.
- **Function** `run_forecast` `(arguments: argparse.Namespace) -> list[Path]` (line 1719)
  - Execute the retained pre-2026 recursive XGBoost experiment and display its metrics.
- **Function** `add_market_arguments` `(parser: argparse.ArgumentParser) -> None` (line 1738)
  - Attach shared required market-query arguments to a subcommand.
- **Function** `add_refresh_argument` `(parser: argparse.ArgumentParser) -> None` (line 1748)
  - Add optional polling refresh support to an analytics subcommand.
- **Function** `run_univariate` `(arguments: argparse.Namespace) -> list[Path]` (line 1759)
  - Load market bars and create the fixed XAU/USD M5 univariate reports.
- **Function** `run_bivariate` `(arguments: argparse.Namespace) -> list[Path]` (line 1772)
  - Load market bars and create a two-variable scatter report.
- **Function** `run_multivariate` `(arguments: argparse.Namespace) -> list[Path]` (line 1794)
  - Load market bars and create one explicit correlation matrix.
- **Function** `run_feature_eda` `(arguments: argparse.Namespace) -> list[Path]` (line 1814)
  - Load one feature version and create explicit feature EDA reports.
- **Function** `build_parser` `() -> argparse.ArgumentParser` (line 1852)
  - Build the unified UNI, BI, MULTI, time-series, evaluation and forecast CLI.
- **Function** `main` `() -> None` (line 1939)
  - Run EDA once or poll PostgreSQL and refresh PNG reports continuously.

## `analytics/data.py`

Read analytics inputs from PostgreSQL through read-only transactions.

- **Function** `parse_utc` `(value: str) -> pd.Timestamp` (line 26)
  - Parse a timezone-aware timestamp and normalize it to UTC.
- **Function** `validate_time_range` `(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]` (line 38)
  - Validate and return a non-empty half-open UTC query interval.
- **Function** `require_data` `(frame: pd.DataFrame, context: str) -> None` (line 48)
  - Raise a clear error when a PostgreSQL query returns no observations.
- **Function** `read_postgres` `(query: str, parameters: Mapping[str, Any], datetime_columns: Sequence[str], engine: Engine | None) -> pd.DataFrame` (line 55)
  - Execute bound SQL inside an explicit read-only transaction.
- **Function** `load_market_data` `(source_id: str, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame` (line 97)
  - Load revision-resolved native bars for one source and UTC range.
- **Function** `load_features` `(source_id: str, feature_version: str, start: str, end: str) -> pd.DataFrame` (line 143)
  - Load observations from the future feature pipeline.
- **Function** `load_predictions` `(source_id: str, model_version: str, evaluation_split: str, start: str, end: str) -> pd.DataFrame` (line 178)
  - Load one explicit split from frozen model predictions.
- **Function** `load_feature_importance` `(source_id: str, model_version: str) -> pd.DataFrame` (line 220)
  - Load importance values for one frozen model from a read-only table.
- **Function** `prepare_daily_close` `(bars: pd.DataFrame) -> pd.DataFrame` (line 241)
  - Aggregate valid intraday bars to final UTC daily closing prices.
- **Function** `require_columns` `(frame: pd.DataFrame, columns: Sequence[str]) -> None` (line 271)
  - Reject missing input columns before transforming an analysis copy.
- **Function** `timeframe_interval` `(timeframe: str) -> pd.Timedelta` (line 279)
  - Return the native candle duration for supported intraday timeframes.
- **Function** `utc_observations` `(values: pd.Series) -> pd.Series` (line 288)
  - Validate explicit timezone information and return UTC timestamps.
- **Function** `prepare_time_series` `(frame: pd.DataFrame, timeframe: str, as_of: pd.Timestamp | None) -> pd.DataFrame` (line 302)
  - Build an ordered, gap-aware analysis copy for one native market stream.
- **Function** `prepare_univariate_data` `(frame: pd.DataFrame) -> pd.DataFrame` (line 388)
  - Prepare an isolated native stream for the ten existing UNI plots.
- **Function** `calculate_return` `(frame: pd.DataFrame, timeframe: str | None) -> pd.DataFrame` (line 397)
  - Add percentage returns only for adjacent native bars, preserving missing values.
- **Function** `prepare_daily_summary` `(native: pd.DataFrame, window: int) -> pd.DataFrame` (line 413)
  - Aggregate observed UTC dates for descriptive insight, keeping absent dates missing.
- **Function** `monthly_returns` `(daily: pd.DataFrame) -> pd.Series` (line 458)
  - Calculate adjacent calendar-month close returns without filling absent months.

## `database/__init__.py`

PostgreSQL persistence package for raw and derived market data.


## `database/migrations/env.py`

Alembic runtime configured from validated ``GOLD_*`` settings.

- **Function** `get_database_url` `() -> str` (line 24)
  - Load the environment-validated PostgreSQL URL for this command.
- **Function** `run_migrations_offline` `() -> None` (line 30)
  - Render migrations as SQL without creating an engine.
- **Function** `run_migrations_online` `() -> None` (line 45)
  - Run migrations through a short-lived PostgreSQL connection.

## `database/migrations/versions/20260929_0001_initial_market_schema.py`

Create the auditable multi-broker XAUUSD ingestion schema.

- **Function** `upgrade` `() -> None` (line 19)
  - Create schemas, partitioned observations, helpers, indexes, and views.
- **Function** `downgrade` `() -> None` (line 290)
  - Remove all collector-owned database objects in dependency order.

## `database/migrations/versions/20260929_0002_processed_m5_view.py`

Create a research-ready M5 view combining native bars and quote aggregates.

- **Function** `upgrade` `() -> None` (line 23)
  - Create the unified, read-only M5 research view.
- **Function** `downgrade` `() -> None` (line 83)
  - Remove the unified M5 research view without touching source data.

## `database/schema.py`

Define PostgreSQL tables and relationships in notebook-style sections.

- **Class** `Base` (line 33)
  - Declarative metadata root used by Alembic and repository mappings.
- **Class** `DataSource` (line 40)
  - Stable broker/server/environment/symbol provenance identity.
- **Class** `SymbolSpecVersion` (line 63)
  - Versioned broker symbol specification observed during preflight.
- **Class** `IngestionRun` (line 93)
  - Audit record and aggregate counters for one collection invocation.
- **Class** `Checkpoint` (line 126)
  - Authoritative resume state for one deterministic fetch partition.
- **Class** `BarObservationModel` (line 162)
  - Append-only native or explicitly derived OHLC observation.
- **Class** `TickObservationModel` (line 202)
  - Append-only native tick with fingerprint plus occurrence identity.
- **Class** `Rejection` (line 241)
  - Invalid provider record retained with reason and raw diagnostic payload.

## `database/storage.py`

Manage database sessions and persistence in notebook-style sections.

- **Function** `get_engine` `(database_url: str | None, connect_timeout_seconds: int | None) -> Engine` (line 50)
  - Create a lazy PostgreSQL engine from a URL or validated settings.
- **Class** `Database` (line 82)
  - Own the SQLAlchemy engine and produce explicit transaction scopes.
- **Method** `Database.session` `(self: Missing) -> Iterator[Session]` (line 110)
  - Yield a session and always close it without implicit commit.
- **Method** `Database.transaction` `(self: Missing) -> Iterator[Session]` (line 118)
  - Commit a complete unit of work or roll it back on any exception.
- **Method** `Database.dispose` `(self: Missing) -> None` (line 124)
  - Release pooled PostgreSQL connections during CLI shutdown.
- **Class** `SourceIdentityConflict` (line 134)
  - Raised when a stable source key is reused for different provenance.
- **Class** `SourceRepository` (line 138)
  - Register source identities and append versioned symbol metadata.
- **Method** `SourceRepository.register` `(self: Missing, session: Session, profile: BrokerProfile, actual_symbol: str) -> DataSource` (line 142)
  - Create or verify a source without changing its provenance identity.
- **Method** `SourceRepository.get_by_key` `(self: Missing, session: Session, source_key: str) -> DataSource` (line 182)
  - Return an active source by stable key or raise a clear error.
- **Method** `SourceRepository.save_symbol_spec` `(self: Missing, session: Session, source_id: UUID, metadata: SymbolMetadata) -> UUID` (line 195)
  - Insert a changed symbol specification or reuse an identical version.
- **Class** `MarketDataRepository` (line 275)
  - Ensure monthly partitions and insert idempotent observation batches.
- **Method** `MarketDataRepository.ensure_partitions` `(self: Missing, session: Session, table_name: str, partition: TimePartition) -> None` (line 287)
  - Ask the database helper to create all required monthly partitions.
- **Method** `MarketDataRepository.insert_bars` `(self: Missing, session: Session, source_id: UUID, symbol_spec_version_id: UUID | None, bars: list[BarObservation]) -> int` (line 304)
  - Bulk insert bars, skipping exact duplicates while retaining revisions.
- **Method** `MarketDataRepository.insert_ticks` `(self: Missing, session: Session, source_id: UUID, ticks: list[TickObservation]) -> int` (line 352)
  - Bulk insert ticks using deterministic fingerprint and occurrence identity.
- **Class** `IngestionRepository` (line 403)
  - Maintain collection audit records and partition resume state.
- **Method** `IngestionRepository.start_run` `(self: Missing, session: Session, source_id: UUID, kind: str, start_utc: datetime | None, end_utc: datetime | None, collector_version: str) -> IngestionRun` (line 406)
  - Create a running ingestion audit row before provider access.
- **Method** `IngestionRepository.finish_run` `(self: Missing, session: Session, run: IngestionRun, status: str, counts: IngestionCounts, error: Exception | None) -> None` (line 435)
  - Finalize run counters and sanitized error details.
- **Method** `IngestionRepository.partition_completed` `(self: Missing, session: Session, source_id: UUID, kind: str, timeframe: str, partition: TimePartition) -> bool` (line 457)
  - Return whether a partition is already fetched or confirmed empty.
- **Method** `IngestionRepository.save_checkpoint` `(self: Missing, session: Session, source_id: UUID, run_id: UUID, kind: str, timeframe: str, partition: TimePartition, status: str, error: str | None) -> None` (line 480)
  - Upsert partition state within the same transaction as inserted data.
- **Class** `RejectionRepository` (line 523)
  - Persist rejected payloads without contaminating valid market tables.
- **Method** `RejectionRepository.insert_many` `(self: Missing, session: Session, run_id: UUID, source_id: UUID, records: list[RejectedRecord]) -> int` (line 527)
  - Bulk insert rejected records and their machine-readable reason codes.

## `market/__init__.py`

XAUUSD market-data acquisition package.


## `market/cli.py`

Compatibility entry point for installed scripts and previous market CLI commands.


## `market/collector.py`

Collect and audit MT5 market data in one notebook-style workflow.

- **Class** `SymbolResolutionError` (line 64)
  - Raised when a configured broker symbol is missing or ambiguous.
- **Function** `resolve_symbol` `(requested: str, available: Iterable[str]) -> tuple[str, list[str]]` (line 68)
  - Resolve an exact symbol or report related candidates without guessing.
- **Class** `MT5Error` (line 98)
  - Base error for terminal initialization and provider calls.
- **Class** `MT5PreflightError` (line 102)
  - Permanent profile, environment, broker, or server mismatch.
- **Class** `MT5Provider` (line 106)
  - Collect native bars and ticks from a locally installed MT5 terminal.
- **Method** `MT5Provider.initialize` `(self: Missing) -> None` (line 138)
  - Lazily import and initialize the official MT5 Python API.
- **Method** `MT5Provider.shutdown` `(self: Missing) -> None` (line 153)
  - Shut down the initialized MT5 bridge safely.
- **Method** `MT5Provider.terminal_snapshot` `(self: Missing) -> TerminalSnapshot` (line 178)
  - Read non-sensitive terminal state and history-limit information.
- **Method** `MT5Provider.account_snapshot` `(self: Missing) -> AccountSnapshot` (line 193)
  - Read safe account attributes without retaining the account number.
- **Method** `MT5Provider.preflight` `(self: Missing, profile: BrokerProfile, environment: Environment) -> tuple[TerminalSnapshot, AccountSnapshot]` (line 217)
  - Validate terminal, account mode, broker, and server before fetching.
- **Method** `MT5Provider.available_symbols` `(self: Missing) -> list[str]` (line 255)
  - List broker symbols for conservative exact symbol resolution.
- **Method** `MT5Provider.ensure_symbol_selected` `(self: Missing, symbol: str) -> None` (line 263)
  - Make a configured symbol visible for market-data retrieval.
- **Method** `MT5Provider.symbol_metadata` `(self: Missing, symbol: str) -> SymbolMetadata` (line 271)
  - Capture a versionable subset and raw snapshot of symbol metadata.
- **Method** `MT5Provider.fetch_bars` `(self: Missing, symbol: str, timeframe: str, partition: TimePartition) -> list[dict[str, Any]]` (line 301)
  - Retrieve native broker OHLC bars for a half-open UTC partition.
- **Method** `MT5Provider.fetch_ticks` `(self: Missing, symbol: str, partition: TimePartition) -> list[dict[str, Any]]` (line 322)
  - Retrieve all native historical tick fields for a UTC partition.
- **Function** `require_aware_utc` `(value: datetime, name: str) -> datetime` (line 346)
  - Return a datetime in UTC and reject timezone-naive input.
- **Function** `monthly_partitions` `(start: datetime, end: datetime) -> list[TimePartition]` (line 354)
  - Split a requested range at UTC calendar-month boundaries.
- **Function** `daily_partitions` `(start: datetime, end: datetime) -> list[TimePartition]` (line 377)
  - Split a requested range at UTC midnight boundaries.
- **Function** `canonical_payload_hash` `(payload: Mapping[str, Any]) -> str` (line 411)
  - Create a stable SHA-256 fingerprint from provider-native fields.
- **Function** `decimal_or_none` `(value: Any) -> Decimal | None` (line 424)
  - Convert a provider numeric value to lossless decimal text semantics.
- **Function** `normalize_bar` `(payload: Mapping[str, Any], actual_symbol: str, timeframe: str, point: Decimal | None, now_utc: datetime, finalization_delay_seconds: int) -> BarObservation | RejectedRecord` (line 442)
  - Normalize and validate one native MT5 OHLC bar.
- **Function** `normalize_bars` `(payloads: Iterable[Mapping[str, Any]], actual_symbol: str, timeframe: str, point: Decimal | None, now_utc: datetime, finalization_delay_seconds: int) -> tuple[list[BarObservation], list[RejectedRecord]]` (line 522)
  - Normalize a batch of bars without discarding invalid provider records.
- **Function** `normalize_ticks` `(payloads: Iterable[Mapping[str, Any]], actual_symbol: str) -> tuple[list[TickObservation], list[RejectedRecord]]` (line 549)
  - Normalize ticks and assign deterministic duplicate occurrences.
- **Function** `aggregate_m5_bars` `(bars: list[BarObservation], target_timeframe: str) -> list[BarObservation]` (line 623)
  - Aggregate complete M5 groups into M15, M30, H1, or H4 bars.
- **Function** `detect_candidate_gaps` `(bars: list[BarObservation], timeframe: str) -> list[tuple[datetime, datetime]]` (line 682)
  - Return temporal gaps for investigation without attempting repair.
- **Class** `MarketProvider` (line 709)
  - Minimal read-only provider surface required by the ingestion service.
- **Method** `MarketProvider.fetch_bars` `(self: Missing, symbol: str, timeframe: str, partition: TimePartition) -> list[dict[str, Any]]` (line 712)
  - Fetch native bars for a half-open partition.
- **Method** `MarketProvider.fetch_ticks` `(self: Missing, symbol: str, partition: TimePartition) -> list[dict[str, Any]]` (line 717)
  - Fetch native ticks for a half-open partition.
- **Class** `IngestionService` (line 721)
  - Coordinate bounded, idempotent, partition-based market collection.
- **Method** `IngestionService.ingest_bars` `(self: Missing, start_utc: datetime, end_utc: datetime, timeframe: str) -> IngestionCounts` (line 759)
  - Collect monthly native-bar partitions and resume completed work.
- **Method** `IngestionService.ingest_ticks` `(self: Missing, start_utc: datetime, end_utc: datetime) -> IngestionCounts` (line 770)
  - Collect daily historical tick partitions and resume completed work.
- **Function** `coverage_report` `(session: Session, source_key: str) -> list[Mapping[str, Any]]` (line 1020)
  - Return requested versus actual coverage for one stable source key.
- **Function** `run_status` `(session: Session, source_key: str, limit: int) -> list[Mapping[str, Any]]` (line 1043)
  - Return recent ingestion audit rows without sensitive configuration.
- **Function** `validation_report` `(session: Session, source_key: str) -> Mapping[str, Any]` (line 1066)
  - Count persisted invariant violations and recorded rejection reasons.
- **Function** `redact_sensitive_fields` `(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]` (line 1104)
  - Remove sensitive values before the event reaches any renderer.
- **Function** `configure_logging` `(level: str) -> None` (line 1115)
  - Configure standard logging and structlog for newline-delimited JSON.
- **Function** `parse_utc` `(value: str) -> datetime` (line 1146)
  - Parse an ISO-8601 timestamp and require an explicit timezone.
- **Function** `print_json` `(value: Any) -> None` (line 1166)
  - Emit deterministic, human-readable JSON without exposing settings.
- **Function** `load_runtime` `(profile_name: str) -> tuple[Settings, BrokerProfile]` (line 1172)
  - Load settings/profile, validate environment, and configure safe logging.
- **Function** `inspected_provider` `(settings: Settings, profile: BrokerProfile) -> Iterator[tuple[MT5Provider, str, SymbolMetadata]]` (line 1183)
  - Yield a preflight-validated provider, exact symbol, and metadata.
- **Function** `build_ingestion_service` `(database: Database, provider: MT5Provider, settings: Settings, profile: BrokerProfile, actual_symbol: str, metadata: SymbolMetadata) -> IngestionService` (line 1203)
  - Verify the registered source and assemble its ingestion service.
- **Function** `source_register` `(profile: str) -> None` (line 1250)
  - Preflight the active terminal and persist source plus symbol provenance.
- **Function** `mt5_info` `(profile: str) -> None` (line 1277)
  - Run mandatory preflight without displaying account number or secrets.
- **Function** `symbol_info` `(profile: str) -> None` (line 1303)
  - Validate the configured exact symbol and display its current specification.
- **Function** `ingest_bars` `(profile: str, start: str, end: str, timeframe: str) -> None` (line 1358)
  - Collect a requested native-bar range with checkpoint/resume semantics.
- **Function** `ingest_ticks` `(profile: str, start: str, end: str) -> None` (line 1370)
  - Collect all ticks available for the requested range and record coverage.
- **Function** `sync_bars` `(profile: str, timeframe: str) -> None` (line 1399)
  - Incrementally collect bars from the latest logical candle through now.
- **Function** `sync_ticks` `(profile: str) -> None` (line 1417)
  - Incrementally re-fetch the latest UTC day for deterministic tick identity.
- **Function** `status` `(profile: str) -> None` (line 1452)
  - Display recent run states and counters for a source.
- **Function** `coverage` `(profile: str) -> None` (line 1459)
  - Display partition and observation coverage for a source.
- **Function** `validate` `(profile: str) -> None` (line 1466)
  - Report stored invalid-row counts without mutating the dataset.

## `market/contracts.py`

Typed contracts shared by the collector, validation, and persistence layers.

- **Class** `Environment` (line 17)
  - Supported physical data environments.
- **Class** `AccountKind` (line 24)
  - Normalized MT5 account modes used by preflight validation.
- **Class** `TerminalSnapshot` (line 34)
  - Non-sensitive terminal state captured during MT5 preflight.
- **Class** `AccountSnapshot` (line 45)
  - Safe account metadata; the account login is intentionally excluded.
- **Class** `SymbolMetadata` (line 55)
  - Versionable MT5 symbol specification associated with one source.
- **Class** `BarObservation` (line 73)
  - Normalized native MT5 OHLC bar ready for validation and persistence.
- **Class** `TickObservation` (line 93)
  - Normalized native MT5 tick with deterministic identity fields.
- **Class** `RejectedRecord` (line 113)
  - Rejected provider payload retained for an auditable failure trail.
- **Class** `TimePartition` (line 123)
  - Half-open UTC interval used as a deterministic collection unit.
- **Method** `TimePartition.label` `(self: Missing) -> str` (line 130)
  - Return a compact date label suitable for logs.
- **Class** `IngestionCounts` (line 139)
  - Mutable row counters accumulated for a single ingestion run.
- **Method** `IngestionCounts.add` `(self: Missing, other: IngestionCounts) -> None` (line 147)
  - Accumulate another partition's counters into this run.

## `market/dashboard.py`

Read-only FastAPI browser for bars, ticks, and unified M5 data.

- **Class** `DatasetSpec` (line 131)
  - Whitelisted SQL and presentation metadata for one dashboard dataset.
- **Class** `MarketDataPage` (line 200)
  - One filtered and sorted page from a whitelisted market dataset.
- **Class** `ProcessedM5Page` (line 216)
  - Backward-compatible page returned by the original processed-M5 API.
- **Class** `DashboardConfigurationError` (line 226)
  - Raised when the selected read-only demo or live database is not configured.
- **Function** `get_database` `(environment: Environment) -> Database` (line 234)
  - Return the lazy read-only database manager selected by a safe environment value.
- **Function** `lifespan` `(_: FastAPI) -> Iterator[None]` (line 254)
  - Dispose every lazily created demo/live engine when the web server stops.
- **Function** `query_market_data` `(database: Database, dataset: DatasetKey, symbol: str | None, source_key: str | None, timeframe: str | None, start_utc: datetime | None, end_utc: datetime | None, quote_status: QuoteStatus, sort_by: str | None, sort_direction: SortDirection, limit: int, offset: int) -> MarketDataPage` (line 275)
  - Read one safe page from bars, ticks, or the unified M5 view.
- **Function** `query_processed_m5` `(database: Database, symbol: str | None, source_key: str | None, start_utc: datetime | None, end_utc: datetime | None, quote_status: QuoteStatus, limit: int, offset: int) -> ProcessedM5Page` (line 357)
  - Read one backward-compatible page from ``market.processed_m5``.
- **Function** `dashboard_page` `() -> HTMLResponse` (line 394)
  - Serve the interactive market-data interface without querying PostgreSQL.
- **Function** `market_data` `(environment: Annotated[Environment, Query(description='Configured database to browse: demo or live.')], dataset: Annotated[DatasetKey, Query(description='Dataset: processed_m5, bars, or ticks.')], symbol: Annotated[str | None, Query(description='Exact broker symbol, for example XAUUSD or XAUUSDm.')], source_key: Annotated[str | None, Query(description='Stable configured data-source key.')], timeframe: Annotated[str | None, Query(description='Exact timeframe for bars, for example M5, M15, or H1.')], start_utc: Annotated[datetime | None, Query(description='Inclusive lower UTC timestamp.')], end_utc: Annotated[datetime | None, Query(description='Exclusive upper UTC timestamp.')], quote_status: Annotated[QuoteStatus, Query(description='All rows, valid/available quotes, or missing/invalid quotes.')], sort_by: Annotated[str | None, Query(description='Column to sort; must exist in the selected dataset.')], sort_direction: Annotated[SortDirection, Query(description='Ascending or descending ordering.')], limit: Annotated[int, Query(ge=10, le=200, description='Rows per page.')], offset: Annotated[int, Query(ge=0, description='Rows skipped for pagination.')]) -> MarketDataPage` (line 411)
  - Return a validated page from the selected demo or live market database.
- **Function** `processed_m5_data` `(environment: Annotated[Environment, Query(description='Configured database to browse: demo or live.')], symbol: Annotated[str | None, Query(description='Exact broker symbol.')], source_key: Annotated[str | None, Query(description='Stable source key.')], start_utc: Annotated[datetime | None, Query(description='Inclusive lower UTC candle timestamp.')], end_utc: Annotated[datetime | None, Query(description='Exclusive upper UTC candle timestamp.')], quote_status: Annotated[QuoteStatus, Query(description='Filter by paired Bid/Ask availability.')], limit: Annotated[int, Query(ge=10, le=200, description='Rows per page.')], offset: Annotated[int, Query(ge=0, description='Rows skipped for pagination.')]) -> ProcessedM5Page` (line 498)
  - Return the original processed-M5 contract from the selected environment.
- **Function** `run` `() -> None` (line 547)
  - Start the read-only dashboard on the local loopback interface.

## `market/profile.py`

Manage broker TOML profiles and runtime settings in notebook-style sections.

- **Class** `SourceProfile` (line 26)
  - Stable source identity and expected account attributes.
- **Class** `MarketProfile` (line 37)
  - Market and timeframe selection for one broker source.
- **Method** `MarketProfile.require_utc` `(self: Missing) -> MarketProfile` (line 45)
  - Enforce UTC as the collector's canonical timezone.
- **Class** `CollectionProfile` (line 53)
  - Chunking, retry, and candle-finalization behavior.
- **Method** `CollectionProfile.validate_policy` `(self: Missing) -> CollectionProfile` (line 63)
  - Require the deterministic partition policy mandated by the platform.
- **Class** `BrokerProfile` (line 73)
  - Complete validated non-secret configuration for one data source.
- **Method** `BrokerProfile.validate_environment` `(self: Missing, environment: Environment) -> None` (line 80)
  - Fail fast when a profile targets another physical environment.
- **Function** `load_broker_profile` `(name: str, root: Path) -> BrokerProfile` (line 89)
  - Load a named broker profile without allowing directory traversal.
- **Class** `Settings` (line 115)
  - Runtime settings loaded from ``GOLD_*`` environment variables.
- **Method** `Settings.empty_terminal_path_is_none` `(cls: Missing, value: object) -> object` (line 143)
  - Treat an empty optional environment variable as no explicit path.
- **Method** `Settings.valid_tick_raw_payload_must_remain_disabled` `(cls: Missing, value: bool) -> bool` (line 150)
  - Allow environment parsing of ``false`` while rejecting raw tick payloads.
- **Method** `Settings.validate_database_environment` `(self: Missing) -> Settings` (line 158)
  - Reject obvious cross-environment collector and dashboard DSN mistakes.
- **Method** `Settings.dashboard_database_url` `(self: Missing, environment: Environment) -> str` (line 179)
  - Return the configured read-only dashboard database for one environment.
- **Function** `load_settings` `() -> Settings` (line 203)
  - Load and validate runtime settings for an explicit command invocation.
- **Function** `save_broker_profile` `(name: str, profile: BrokerProfile, root: Path, overwrite: bool) -> Path` (line 214)
  - Validate and save a non-secret broker profile as UTF-8 TOML.

## `technical/__init__.py`

Read-only XAUUSD M5 research: causal features and frozen walk-forward models.


## `technical/config.py`

Reproducible M5 feature, selection, and walk-forward configuration.

- **Class** `Config` (line 11)
  - Control research cost and causal indicator parameters without changing M5.

## `technical/dashboard.py`

Embed Plotly, frozen predictions and audits into one offline broker-style HTML file.

- **Function** `write_dashboard` `(predictions: pd.DataFrame, candles: pd.DataFrame, metrics: dict[str, Any], folds: list[dict[str, Any]], feature_audits: list[dict[str, Any]], output: Path, rolling_window: int) -> Path` (line 14)
  - Build an offline dashboard with windowed candle/line charts and paginated filters.
- **Function** `main` `() -> None` (line 69)
  - Rebuild a dashboard from saved fold artifacts without querying or retraining.

## `technical/data_preparation.py`

Prepare canonical M5 data and temporal folds in notebook-style sections.

- **Function** `load_data` `(source_key: str, symbol: str, start: str, end: str, engine: Engine | None) -> pd.DataFrame` (line 26)
  - Read one source/symbol M5 interval with bound SQL in a read-only transaction.
- **Function** `prepare_data` `(frame: pd.DataFrame, as_of: pd.Timestamp | None) -> tuple[pd.DataFrame, dict[str, Any]]` (line 67)
  - Normalize UTC/numbers, keep last stable duplicate, reject invalid OHLC and report gaps.
- **Function** `make_labels` `(data: pd.DataFrame, horizon_bars: int, minimum_edge_bps: float, slippage_bps: float) -> pd.DataFrame` (line 184)
  - Attach spread/slippage-aware training labels separately from direction outcomes.
- **Class** `Fold` (line 247)
  - Five disjoint half-open UTC blocks: TRAIN/TUNE/CALIBRATION/POLICY/TEST.
- **Method** `Fold.split` `(self: Missing, labels: pd.DataFrame) -> dict[str, pd.Index]` (line 253)
  - Select known development labels whose target closes strictly inside their block.
- **Function** `make_folds` `(start: str, evaluation_end: str, tune_start: str, calibration_start: str, policy_start: str, test_start: str) -> list[Fold]` (line 269)
  - Build annual tests 2020–2025 and a configurable frozen final evaluation.
- **Function** `purged_cv` `(times: pd.Series, targets: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]` (line 328)
  - Return expanding CV indices, removing any train target at/after validation start.

## `technical/xgboost_model.py`

Run the complete XGBoost M5 workflow in notebook-style sections.

- **Function** `safe_divide` `(numerator: pd.Series, denominator: pd.Series) -> pd.Series` (line 64)
  - Divide aligned numeric series, mapping zero denominators/infinity to NaN.
- **Function** `build_features` `(data: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, list[dict[str, Any]]]` (line 69)
  - Return float32 candidates and a formula/parameter/provenance manifest.
- **Function** `classifier` `(config: Config, **parameters: Any) -> XGBClassifier` (line 473)
  - Create a deterministic histogram XGBoost classifier with fixed three-class order.
- **Function** `quality_filter` `(train: pd.DataFrame, config: Config) -> tuple[list[str], dict[str, Any]]` (line 497)
  - Drop numerical/missing/constant/duplicate problems using only provided TRAIN rows.
- **Function** `relieff_rank` `(train: pd.DataFrame, target: pd.Series, config: Config) -> pd.DataFrame` (line 537)
  - Rank multiclass feature relevance with deterministic TRAIN-only reference/anchors.
- **Function** `select_features` `(features: pd.DataFrame, labels: pd.DataFrame, train_indices: pd.Index, config: Config) -> tuple[list[str], dict[str, Any], pd.DataFrame, pd.DataFrame]` (line 587)
  - Screen on an early TRAIN prefix, then run purged RFECV with frozen candidates.
- **Function** `macro_f1` `(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> float` (line 680)
  - Score all three fixed classes, including absent classes with zero F1.
- **Class** `PlattCalibration` (line 686)
  - One-vs-rest sigmoid calibration of log odds, normalized to three probabilities.
- **Method** `PlattCalibration.logits` `(probabilities: np.ndarray) -> np.ndarray` (line 692)
  - Clip raw probabilities and return finite one-vs-rest log odds.
- **Method** `PlattCalibration.fit` `(cls: Missing, probabilities: np.ndarray, targets: pd.Series, seed: int) -> 'PlattCalibration'` (line 698)
  - Fit sigmoid maps only on CALIBRATION; reject unidentifiable missing classes.
- **Method** `PlattCalibration.predict` `(self: Missing, probabilities: np.ndarray) -> np.ndarray` (line 710)
  - Apply frozen sigmoid maps and normalize class probabilities to sum to one.
- **Function** `apply_policy` `(probabilities: np.ndarray, threshold: float) -> np.ndarray` (line 719)
  - Return class argmax; abstain to NO_TRADE when maximum confidence is too low.
- **Function** `load_model_bundle` `(path: Path) -> dict[str, Any]` (line 724)
  - Load a locally generated model bundle, including earlier calibration paths.
- **Function** `regressor` `(config: Config, **parameters: Any) -> XGBRegressor` (line 744)
  - Create a return regressor independently of classifier-only parameters.
- **Function** `decision_costs` `(labels: pd.DataFrame, slippage_bps: float) -> np.ndarray` (line 766)
  - Return spread/edge/slippage threshold without double counting labeled slippage.
- **Function** `attach_direction_outcomes` `(predictions: pd.DataFrame) -> pd.DataFrame` (line 772)
  - Judge accepted signals at the fixed target close; abstentions are always neutral.
- **Function** `load_frozen_models` `(directory: Path) -> dict[str, Any]` (line 809)
  - Load a verified pair of locally trusted models and their immutable inference contract.
- **Function** `predict_latest` `(bundle: dict[str, Any], history: pd.DataFrame, as_of: pd.Timestamp, max_staleness_minutes: float) -> pd.DataFrame` (line 841)
  - Predict one closed candle from actual history with a frozen validated model pair.
- **Function** `select_regression_features` `(features: pd.DataFrame, labels: pd.DataFrame, train_indices: pd.Index, config: Config) -> tuple[list[str], dict[str, Any]]` (line 881)
  - Rank regression features on TRAIN return targets, independently of class RFECV.
- **Function** `precision_statistics` `(actual: np.ndarray, predicted: np.ndarray, label: int) -> dict[str, Any]` (line 910)
  - Report per-direction precision, recall, F1, support and descriptive Wilson bounds.
- **Function** `non_overlapping_positions` `(labels: pd.DataFrame) -> np.ndarray` (line 945)
  - Select fixed chronological evaluation anchors with nonoverlapping target horizons.
- **Function** `apply_selective_policy` `(probabilities: np.ndarray, returns: np.ndarray, costs: np.ndarray, policy: dict[str, Any], lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (line 958)
  - Abstain unless class confidence, regression direction, cost and uncertainty agree.
- **Function** `select_precision_policy` `(probabilities: np.ndarray, returns: np.ndarray, labels: pd.DataFrame, lower: np.ndarray, upper: np.ndarray, config: Config) -> dict[str, Any]` (line 1007)
  - Choose per-class rules by F1 under a precision target, then confirm on later POLICY data.
- **Function** `predict_regression` `(bundle: dict[str, Any], features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (line 1124)
  - Return point and empirical interval forecasts in decimal-return units.
- **Function** `fit_development` `(blocks: dict[str, tuple[pd.DataFrame, pd.DataFrame]], config: Config, regression_blocks: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None) -> tuple[dict[str, Any], dict[str, Any]]` (line 1139)
  - Fit class/return models and selective decisions using only four development blocks.
- **Function** `predict_test` `(bundle: dict[str, Any], features: pd.DataFrame, labels: pd.DataFrame, test_indices: pd.Index, fold: str, evaluation_end: pd.Timestamp) -> pd.DataFrame` (line 1379)
  - Predict from completed M5 snapshots with frozen models; attach actuals afterwards.
- **Function** `evaluate` `(predictions: pd.DataFrame) -> dict[str, Any]` (line 1469)
  - Report raw/action F1, selective precision, return baselines and cost-adjusted proxy.
- **Function** `replay_single_position` `(predictions: pd.DataFrame, candles: pd.DataFrame, delay_bars: int, slippage_bps: float, commission_bps: float) -> dict[str, Any]` (line 1674)
  - Replay at most one position using delayed OHLC bid/ask fill assumptions.
- **Function** `write_evaluation_diagnostics` `(predictions: pd.DataFrame, destination: Path) -> None` (line 1791)
  - Save monthly metrics and a static regime comparison without fitting any model.
- **Function** `write_json` `(path: Path, payload: Any) -> None` (line 1883)
  - Persist strict UTF-8 JSON, refusing non-finite metrics or hidden NaN values.
- **Function** `run_fold` `(features: pd.DataFrame, labels: pd.DataFrame, manifest: list[dict[str, Any]], fold: Fold, config: Config, destination: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]` (line 1891)
  - Execute isolated selection, development fitting and frozen TEST for one fold.
- **Function** `run_pipeline` `(source_key: str, symbol: str, start: str, evaluation_end: str, config: Config | None, output_dir: Path, frame: pd.DataFrame | None, folds: list[Fold] | None, fold_names: list[str] | None) -> Path` (line 2063)
  - Run all seven research folds on PostgreSQL data (or injected test fixture).
- **Function** `main` `() -> None` (line 2208)
  - Expose temporal development boundaries, model and selective precision settings.
