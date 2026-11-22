import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from pipeline import (
    CALENDAR_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    _counter_differences,
    _stuck_active_counter_mask,
    _impute_gaps,
    _calendar_features,
    load_exports,
    _parse_timestamps,
    prepare_dataset,
    build_window_index,
    iter_window_batches,
)


class PipelineTests(unittest.TestCase):
    def test_counter_reset_and_nonconsecutive_reading(self):
        index = pd.date_range("2024-01-01", periods=4, freq="15min")
        counter = pd.Series([10.0, 12.0, 1.0, 4.0], index=index)
        delta, resets = _counter_differences(counter)
        self.assertTrue(np.isnan(delta.iloc[0]))
        self.assertEqual(delta.iloc[1], 2.0)
        self.assertEqual(delta.iloc[2], 1.0)
        self.assertEqual(resets, 1)

        with_gap = counter.copy()
        with_gap.index = pd.DatetimeIndex(
            [index[0], index[1], index[3], index[3] + pd.Timedelta(minutes=15)]
        )
        gap_delta, _ = _counter_differences(with_gap)
        self.assertTrue(np.isnan(gap_delta.iloc[2]))

    def test_short_gap_is_linearly_interpolated_and_records_future_boundary(self):
        index = pd.date_range("2024-01-01", periods=5, freq="15min")
        values = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0], index=index)
        filled, counts, future_boundary = _impute_gaps(values, max_short_gap=2)
        self.assertEqual(filled.iloc[1], 2.0)
        self.assertEqual(
            future_boundary.iloc[1], index[2]
        )
        self.assertEqual(counts["linear_interpolation_short_gap"], 1)
        self.assertEqual(counts["prior_day_or_week"], 0)

    def test_long_gap_is_not_partially_interpolated(self):
        index = pd.date_range("2024-01-01", periods=8, freq="15min")
        values = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0], index=index)
        filled, counts, _ = _impute_gaps(values, max_short_gap=2)
        self.assertTrue(filled.iloc[1:4].isna().all())
        self.assertEqual(counts["linear_interpolation_short_gap"], 0)

    def test_unresolved_missing_values_have_a_separate_quality_flag(self):
        index = pd.date_range("2024-01-01", periods=8, freq="15min")
        records = pd.DataFrame(
            {
                "cumulative_active_kwh": [np.nan, np.nan, 10, 12, 14, 16, 18, 20],
                "cumulative_reactive_kvarh": [np.nan, np.nan, 4, 5, 6, 7, 8, 9],
                "pmax_kw": [np.nan, np.nan, 10, 11, 12, 13, 14, 15],
                "smax_kva": [np.nan, np.nan, 12, 13, 14, 15, 16, 17],
            },
            index=index,
        )
        clean, report = prepare_dataset(records, {}, max_short_gap=0)

        self.assertTrue(clean["has_missing_physical"].iloc[:3].all())
        self.assertFalse(clean["was_imputed"].iloc[:3].any())
        self.assertEqual(
            report["physical_quality_after_imputation"]["pmax_kw"][
                "longest_missing_runs"
            ][0]["steps"],
            2,
        )
        self.assertEqual(report["rows_with_unresolved_physical_missing"], 3)

    def test_stuck_counter_detection_marks_plateau_and_recovery(self):
        index = pd.date_range("2024-01-01", periods=7, freq="15min")
        counter = pd.Series([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0], index=index)
        demand = pd.Series(10.0, index=index)
        flagged = _stuck_active_counter_mask(counter, demand)
        self.assertEqual(int(flagged.sum()), 5)
        self.assertFalse(flagged.iloc[1])
        self.assertTrue(flagged.iloc[2:7].all())

    def test_calendar_features_have_seven_columns(self):
        index = pd.date_range("2024-01-01", periods=2, freq="D")
        features = _calendar_features(index, {index[0].date()})
        self.assertEqual(tuple(features.columns), CALENDAR_COLUMNS)
        self.assertEqual(len(features.columns), 7)
        self.assertEqual(features.iloc[0]["is_nonworking_day"], 1.0)
        self.assertEqual(features.iloc[1]["is_nonworking_day"], 0.0)

    def test_end_of_day_timestamp_is_next_midnight(self):
        dates = pd.Series(["02-04-2023", "2023-11-02 00:00:00"])
        times = pd.Series(["24:00:00", "00:15:00"])
        parsed = _parse_timestamps(dates, times)
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2023-04-03 00:00:00"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2023-11-02 00:15:00"))

    def test_load_exports_from_zip_archive(self):
        csv = (
            "Date;Time;EM_VIRTUAL kWh (kWh) [ABS];EM_VIRTUAL kW dmd (kW) [MAX];"
            "EM_VIRTUAL kvarh (kvarh) [ABS];EM_VIRTUAL kVA dmd (kVA) [MAX]\n"
            "01-01-2024;00:00:00;100;10;20;12\n"
            "01-01-2024;00:15:00;102;11;21;13\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "sinert.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "SINERT_DATA_CONCENTRATOR/January2024/export.csv", csv
                )
            records, audit = load_exports(archive_path)
        self.assertEqual(len(records), 2)
        self.assertEqual(audit["files_read"], 1)
        self.assertEqual(records.iloc[1]["cumulative_active_kwh"], 102)

    def test_window_batch_shapes_and_chronological_split(self):
        index = pd.date_range("2024-01-01", periods=20, freq="15min")
        frame = pd.DataFrame(
            np.ones((len(index), len(FEATURE_COLUMNS))),
            index=index,
            columns=FEATURE_COLUMNS,
        )
        frame["was_imputed"] = False
        frame.loc[index[8], "was_imputed"] = True
        frame["imputation_future_boundary"] = pd.NaT
        frame.loc[index[8], "imputation_future_boundary"] = index[10]
        windows = build_window_index(
            frame,
            window_steps=4,
            horizon_steps=2,
            train_fraction=0.6,
            validation_fraction=0.2,
        )
        uses_interpolation = (
            (windows["input_start"] <= index[8])
            & (windows["origin"] >= index[8])
        )
        self.assertTrue((windows.loc[uses_interpolation, "origin"] >= index[10]).all())
        batches = list(
            iter_window_batches(
                frame,
                split="train",
                window_steps=4,
                horizon_steps=2,
                train_fraction=0.6,
                validation_fraction=0.2,
                batch_size=3,
            )
        )
        x, y, timestamps = batches[0]
        self.assertEqual(x.shape[1:], (4, len(FEATURE_COLUMNS)))
        self.assertEqual(y.shape[1:], (2, len(TARGET_COLUMNS)))
        self.assertEqual(timestamps.shape[1], 2)
        self.assertTrue((timestamps[:, -1] < index[12].to_datetime64()).all())
        self.assertFalse((timestamps == index[8].to_datetime64()).any())
        for batch in batches:
            target_times = pd.to_datetime(batch[2][:, 0])
            # Every returned window obeys either its clean-target constraint
            # and its imputation-boundary constraint.
            self.assertFalse((target_times == index[8]).any())


if __name__ == "__main__":
    unittest.main()
