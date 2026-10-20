from datetime import datetime

import pytest
from pyspark.sql import SparkSession

from features import (
    add_amount_ratio,
    add_transaction_duration,
    add_transaction_frequency,
    cast_timestamp,
    run_feature_pipeline,
)
from ingest import get_transaction_schema
from train import build_pipeline


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("aml-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def _row(ts, from_bank, to_bank, received, paid, fmt="ACH", label=0):
    return (
        ts,
        from_bank,
        "h",
        to_bank,
        "h",
        received,
        "US Dollar",
        paid,
        "US Dollar",
        fmt,
        label,
    )


@pytest.fixture
def df(spark):
    rows = [
        _row("2022/09/01 00:20", 10, 20, 100.0, 100.0),
        _row("2022/09/01 00:25", 11, 20, 0.0, 50.0),
        _row("2022/09/01 01:25", 10, 20, 200.0, 100.0, label=1),
        _row("2022/09/02 00:00", 12, 30, 10.0, 10.0, fmt="Wire"),
    ]
    return spark.createDataFrame(rows, get_transaction_schema())


def test_timestamps_parse_in_dataset_format(df):
    ts = [r.Timestamp for r in cast_timestamp(df).orderBy("Timestamp").collect()]
    assert ts[0] == datetime(2022, 9, 1, 0, 20)
    assert None not in ts


def test_duration_is_seconds_since_previous_transaction_to_same_bank(df):
    out = add_transaction_duration(cast_timestamp(df)).orderBy("Timestamp").collect()
    by_ts = {r.Timestamp.strftime("%d %H:%M"): r.Transaction_Duration for r in out}
    assert by_ts["01 00:20"] == 0  # first transaction to bank 20
    assert by_ts["01 00:25"] == 5 * 60
    assert by_ts["01 01:25"] == 60 * 60
    assert by_ts["02 00:00"] == 0  # first transaction to bank 30


def test_amount_ratio_guards_zero_received(df):
    ratios = {
        (r["From Bank"], r["Amount Received"]): r.Amount_Ratio
        for r in add_amount_ratio(df).collect()
    }
    assert ratios[(10, 100.0)] == 1.0
    assert ratios[(10, 200.0)] == 0.5
    assert ratios[(11, 0.0)] == 1.0  # no division by zero


def test_frequency_counts_sending_bank(df):
    freq = {
        r["From Bank"]: r.Transaction_Frequency
        for r in add_transaction_frequency(df).collect()
    }
    assert freq == {10: 2, 11: 1, 12: 1}


def test_pipeline_fits_and_predicts(df):
    engineered = run_feature_pipeline(df)
    model = build_pipeline().fit(engineered)
    preds = model.transform(engineered).select("prediction").collect()
    assert len(preds) == 4
