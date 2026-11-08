import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from build_calendar import (
    build_calendar_daily_features,
    load_supcom_calendar_events,
)
from pipeline import (
    CALENDAR_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    _counter_differences,
    _stuck_active_counter_mask,
    _impute_gaps,
    _calendar_features,
    _calendar_metadata,
    _circular_clock_statistics,
    _rolling_outlier_masks,
    build_behavior_reports,
    load_exports,
    load_behavior_calendar,
    load_holiday_dates,
    _parse_timestamps,
    prepare_dataset,
    build_window_index,
    iter_window_batches,
    validate_window_index,
    write_phase1_visualizations,
)


class PipelineTests(unittest.TestCase):
    def test_calendar_builder_expands_events_and_preserves_unknown_dates(self):
        events = pd.DataFrame(
            [
                {
                    "event_id": "annual",
                    "event": "Holiday",
                    "event_type": "public_holiday",
                    "start_date": pd.NA,
                    "end_date": pd.NA,
                    "recurrence_month_day": "01-01",
                    "source_id": "S1",
                    "confidence": "high",
                    "notes": "Statutory",
                },
                {
                    "event_id": "break",
                    "event": "Winter break",
                    "event_type": "university_vacation",
                    "start_date": "2023-12-30",
                    "end_date": "2024-01-02",
                    "recurrence_month_day": pd.NA,
                    "source_id": "S2",
                    "confidence": "medium",
                    "notes": "Reported",
                },
            ]
        )
        calendar = build_calendar_daily_features(
            events, "2023-12-31", "2024-01-02"
        ).set_index("date")
        self.assertTrue(calendar.loc["2024-01-01", "is_holiday"])
        self.assertTrue(calendar.loc["2024-01-01", "is_vacation"])
        self.assertEqual(
            calendar.loc["2024-01-01", "calendar_confidence"], "high"
        )
        self.assertEqual(calendar.loc["2024-01-01", "source_ids"], "S1|S2")
        self.assertTrue(calendar.loc["2023-12-31", "is_vacation"])
        self.assertTrue(pd.isna(calendar.loc["2023-12-31", "is_holiday"]))
        self.assertTrue(pd.isna(calendar.loc["2024-01-02", "is_study_week"]))
        with tempfile.TemporaryDirectory() as directory:
            calendar_path = Path(directory) / "daily.csv"
            calendar.reset_index().to_csv(calendar_path, index=False)
            loaded = load_behavior_calendar(calendar_path)
        metadata = _calendar_metadata(loaded, pd.DatetimeIndex(calendar.index))
        self.assertEqual(metadata["labelled_days_in_data"], 3)
        self.assertEqual(metadata["label_confidence_days"]["high"], 1)
        self.assertEqual(metadata["label_confidence_days"]["medium"], 2)
        self.assertEqual(metadata["calendar_source_ids"], ["S1", "S2"])

    def test_supcom_workbook_preserves_status_and_excludes_estimates_from_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            workbook_path = Path(directory) / "supcom.xlsx"
            pd.DataFrame(
                [
                    {
                        "Année univ.": "2025-2026",
                        "Début": "2025-12-20",
                        "Fin": "2026-01-04",
                        "Durée (jours)": 16,
                        "Type": "Vacances",
                        "Libellé": "Vacances d'hiver",
                        "Statut": "Ministère (officiel)",
                    },
                    {
                        "Année univ.": "2025-2026",
                        "Début": "2026-01-05",
                        "Fin": "2026-01-10",
                        "Durée (jours)": 6,
                        "Type": "Examens",
                        "Libellé": "Examens S1",
                        "Statut": "Estimation",
                    },
                    {
                        "Année univ.": "2026-2027",
                        "Début": "2027-01-04",
                        "Fin": "2027-01-09",
                        "Durée (jours)": 6,
                        "Type": "Examens",
                        "Libellé": "Examens S1",
                        "Statut": "Confirmé (SUP'COM)",
                    },
                ]
            ).to_excel(workbook_path, sheet_name="Calendrier", index=False)
            events, sources = load_supcom_calendar_events(workbook_path)
            daily = build_calendar_daily_features(
                events, "2025-12-20", "2026-01-10"
            ).set_index("date")
            daily_path = Path(directory) / "daily.csv"
            daily.reset_index().to_csv(daily_path, index=False)
            loaded = load_behavior_calendar(daily_path)

        self.assertEqual(len(events), 3)
        self.assertEqual(len(sources), 3)
        self.assertTrue(daily.loc["2025-12-25", "is_vacation"])
        self.assertTrue(pd.isna(daily.loc["2026-01-05", "is_exam_period"]))
        self.assertIn("Examens S1", daily.loc["2026-01-05", "event_name"])
        self.assertEqual(daily.loc["2026-01-05", "calendar_confidence"], "low")
        self.assertTrue(pd.isna(daily["is_study_week"]).all())
        self.assertTrue(
            pd.isna(loaded.loc[pd.Timestamp("2026-01-05"), "is_exam_period"])
        )

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

    def test_zero_power_with_positive_energy_is_preserved_and_marked_suspect(self):
        index = pd.date_range("2024-01-01", periods=3, freq="15min")
        records = pd.DataFrame(
            {
                "cumulative_active_kwh": [100.0, 110.0, 120.0],
                "cumulative_reactive_kvarh": [20.0, 22.0, 24.0],
                "pmax_kw": [50.0, 0.0, 55.0],
                "smax_kva": [60.0, 0.0, 65.0],
            },
            index=index,
        )
        clean, report = prepare_dataset(records, {}, max_short_gap=0)

        self.assertEqual(clean.loc[index[1], "pmax_kw"], 0)
        self.assertEqual(clean.loc[index[1], "pmax_kw_original"], 0)
        self.assertEqual(clean.loc[index[1], "smax_kva_original"], 0)
        self.assertEqual(clean.loc[index[1], "power_zero_status"], "suspect")
        self.assertEqual(
            clean.loc[index[1], "power_zero_reason"],
            "zero_power_with_positive_active_energy",
        )
        self.assertEqual(
            report["suspicious_zero_measurements_not_removed"][
                "zero_pmax_with_positive_active_energy"
            ],
            1,
        )

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
            "01-01-2024;00:30:00;---;12;23;14\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "sinert.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "SINERT_DATA_CONCENTRATOR/January2024/export.csv", csv
                )
            records, audit = load_exports(archive_path)
        self.assertEqual(len(records), 3)
        self.assertEqual(audit["files_read"], 1)
        self.assertEqual(records.iloc[1]["cumulative_active_kwh"], 102)
        self.assertEqual(records["cumulative_active_kwh"].dtype, np.dtype("float64"))
        self.assertTrue(pd.isna(records.iloc[2]["cumulative_active_kwh"]))

    def test_duplicate_export_timestamps_keep_last_record_and_audit_duplicate(self):
        csv = (
            "Date;Time;EM_VIRTUAL kWh (kWh) [ABS];EM_VIRTUAL kW dmd (kW) [MAX];"
            "EM_VIRTUAL kvarh (kvarh) [ABS];EM_VIRTUAL kVA dmd (kVA) [MAX]\n"
            "01-01-2024;00:00:00;100;10;20;12\n"
            "01-01-2024;00:15:00;102;11;21;13\n"
            "01-01-2024;00:15:00;103;12;22;14\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "duplicates.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("part.csv", csv)
            records, audit = load_exports(archive_path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records.iloc[1]["cumulative_active_kwh"], 103)
        self.assertEqual(audit["duplicate_timestamps_dropped"], 1)

    def test_behavior_reports_include_energy_load_activity_and_calendar_summaries(self):
        index = pd.DatetimeIndex(
            list(pd.date_range("2024-01-01", periods=8, freq="15min"))
            + list(pd.date_range("2024-01-06", periods=8, freq="15min"))
            + list(pd.date_range("2024-01-07", periods=8, freq="15min"))
        )
        values = np.arange(1, len(index) + 1, dtype=float)
        pmax = pd.array(values, dtype="Float64")
        pmax[4:] = pd.NA
        pmax[:4] = values[:4]
        pmax[8:16] = values[8:16]
        pmax[2] = 0
        smax = values + 1
        smax[2] = 0
        clean = pd.DataFrame(
            {
                "delta_ea_kwh": values,
                "delta_er_kvarh": values / 10,
                "pmax_kw": pmax,
                "smax_kva": smax,
            },
            index=index,
        )
        clean["imputed_delta_ea_kwh"] = False
        clean.loc[index[0], "imputed_delta_ea_kwh"] = True
        with tempfile.TemporaryDirectory() as directory:
            calendar_path = Path(directory) / "calendar.csv"
            calendar_path.write_text(
                "date,is_holiday,is_vacation,is_study_week,is_exam_period\n"
                "2024-01-01,yes,no,yes,no\n"
                "2024-01-06,no,oui,non,yes\n"
                "2024-01-07,,,,\n",
                encoding="utf-8",
            )
            calendar = load_behavior_calendar(calendar_path)

        with tempfile.TemporaryDirectory() as directory:
            calendar_path = Path(directory) / "calendar.xlsx"
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-06"],
                    "is_holiday": ["yes", "no"],
                    "is_vacation": ["no", "oui"],
                    "is_study_week": ["yes", "non"],
                    "is_exam_period": ["non", "yes"],
                }
            ).to_excel(calendar_path, index=False)
            excel_calendar = load_behavior_calendar(calendar_path)
        self.assertEqual(calendar.iloc[:2].shape, excel_calendar.shape)
        self.assertTrue(calendar.iloc[:2].equals(excel_calendar))
        calendar_metadata = _calendar_metadata(calendar, clean.index)
        self.assertEqual(calendar_metadata["labelled_days_in_data"], 2)
        self.assertTrue(calendar_metadata["holiday_labels_available"])
        self.assertTrue(calendar_metadata["vacation_labels_available"])
        self.assertTrue(calendar_metadata["study_week_labels_available"])

        reports = build_behavior_reports(
            clean, calendar, peak_quantile=0.75, activity_threshold_kw=1
        )
        self.assertEqual(
            set(reports),
            {
                "energy_daily.csv",
                "energy_weekly.csv",
                "energy_monthly.csv",
                "energy_annual.csv",
                "behavior_by_group.csv",
                "activity_by_day.csv",
                "peak_episodes.csv",
                "behavior_summary.csv",
                "suspect_power_zeros.csv",
                "reactive_outliers.csv",
                "reactive_outlier_summary.csv",
                "missing_imputation_quality.csv",
                "missing_runs.csv",
                "coverage_by_month.csv",
                "coverage_by_year.csv",
                "coverage_summary.csv",
            },
        )
        daily = reports["energy_daily.csv"]
        self.assertEqual(len(daily), 3)
        self.assertEqual(daily.iloc[0]["energy_kwh"], values[:8].sum())
        self.assertEqual(daily.iloc[0]["observed_intervals"], 7)
        self.assertEqual(daily.iloc[0]["imputed_intervals"], 1)
        self.assertEqual(daily.iloc[0]["measured_energy_kwh"], values[1:8].sum())
        self.assertEqual(daily.iloc[0]["expected_intervals"], 96)
        coverage = reports["coverage_by_year.csv"].iloc[0]
        self.assertGreaterEqual(
            coverage["expected_in_scope_intervals"], len(clean)
        )
        self.assertEqual(coverage["observed_intervals"], len(clean) - 1)
        self.assertEqual(
            reports["behavior_summary.csv"].iloc[0]["pmax_kw_max"], values[15]
        )
        self.assertEqual(
            reports["behavior_summary.csv"].iloc[0]["calendar_days_labelled"], 2
        )
        grouped = reports["behavior_by_group.csv"]
        self.assertIn(
            ("vacation", "yes"),
            set(zip(grouped["dimension"], grouped["category"])),
        )
        self.assertIn(
            ("exam_period", "yes"),
            set(zip(grouped["dimension"], grouped["category"])),
        )
        activity = reports["activity_by_day.csv"]
        self.assertEqual(len(activity), 3)
        self.assertEqual(activity.iloc[0]["active_intervals"], 2)
        self.assertTrue(np.isnan(activity.iloc[2]["active_duration_hours"]))
        self.assertGreater(len(reports["peak_episodes.csv"]), 0)
        zero_report = reports["suspect_power_zeros.csv"]
        self.assertEqual(len(zero_report), 1)
        self.assertEqual(zero_report.iloc[0]["status"], "suspect")
        self.assertEqual(zero_report.iloc[0]["original_pmax_kw"], 0)
        self.assertIn(
            "raw exports unavailable", zero_report.iloc[0]["source_verification"]
        )
        with tempfile.TemporaryDirectory() as directory:
            visualization_files = write_phase1_visualizations(
                clean, reports, Path(directory)
            )
            self.assertEqual(len(visualization_files), 7)
            for relative_path in visualization_files:
                self.assertTrue((Path(directory) / relative_path).is_file())

    def test_rolling_outlier_detection_flags_extreme_reactive_value(self):
        index = pd.date_range("2024-01-01", periods=100, freq="15min")
        frame = pd.DataFrame(
            {
                "delta_ea_kwh": np.ones(100),
                "delta_er_kvarh": np.r_[np.ones(99), 100.0],
                "pmax_kw": np.ones(100),
                "smax_kva": np.ones(100),
            },
            index=index,
        )
        flags, baselines, _ = _rolling_outlier_masks(frame)
        self.assertTrue(flags.loc[index[-1], "delta_er_kvarh"])
        self.assertEqual(baselines["delta_er_kvarh"].loc[index[-1]], 1.0)

    def test_behavior_calendar_rejects_invalid_flags_and_duplicate_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            calendar_path = Path(directory) / "calendar.csv"
            calendar_path.write_text(
                "date,is_vacation\n2024-01-01,maybe\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "invalid boolean"):
                load_behavior_calendar(calendar_path)

            calendar_path.write_text(
                "date,is_vacation\n"
                "2024-01-01,yes\n"
                "2024-01-01,no\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate dates"):
                load_behavior_calendar(calendar_path)

    def test_calendar_keeps_legacy_holidays_and_requires_explicit_activity_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            calendar_path = Path(directory) / "calendar.csv"
            calendar_path.write_text(
                "date\n2024-01-06\n", encoding="utf-8"
            )
            legacy_calendar = load_behavior_calendar(calendar_path)
            self.assertTrue(legacy_calendar.iloc[0]["is_holiday"])
            self.assertNotIn("is_vacation", legacy_calendar.columns)
            self.assertEqual(
                load_holiday_dates(calendar_path), {pd.Timestamp("2024-01-06").date()}
            )

            calendar_path.write_text(
                "date,is_vacation\n2024-01-06,yes\n", encoding="utf-8"
            )
            vacation_calendar = load_behavior_calendar(calendar_path)
            self.assertNotIn("is_holiday", vacation_calendar.columns)
            self.assertTrue(vacation_calendar.iloc[0]["is_vacation"])
            self.assertEqual(load_holiday_dates(calendar_path), set())
            self.assertEqual(len(load_behavior_calendar(None).columns), 0)

    def test_calendar_metadata_marks_empty_labels_unavailable(self):
        index = pd.date_range("2024-01-01", periods=2, freq="D")
        calendar = pd.DataFrame(
            {"is_holiday": [pd.NA, pd.NA], "is_study_week": [pd.NA, pd.NA]},
            index=index,
        )
        metadata = _calendar_metadata(calendar, index)
        self.assertEqual(metadata["labelled_days_in_data"], 0)
        self.assertFalse(metadata["holiday_labels_available"])
        self.assertFalse(metadata["study_week_labels_available"])

    def test_clock_statistics_handle_midnight_circularly(self):
        times = pd.Series(
            pd.to_datetime(["2024-01-01 23:45", "2024-01-02 00:15"])
        )
        mean_hour, std_hours, variance = _circular_clock_statistics(times)
        self.assertLess(min(mean_hour, 24 - mean_hour), 0.01)
        self.assertLess(std_hours, 0.5)
        self.assertGreaterEqual(variance, 0)

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
        validation = validate_window_index(
            frame,
            windows,
            window_steps=4,
            horizon_steps=2,
            train_fraction=0.6,
            validation_fraction=0.2,
        ).iloc[0]
        self.assertEqual(validation["feature_count"], 11)
        self.assertTrue(validation["lookback_length_valid"])
        self.assertTrue(validation["horizon_length_valid"])
        self.assertTrue(validation["split_boundaries_valid"])
        self.assertTrue(validation["target_timestamps_disjoint"])
        self.assertTrue(validation["targets_exclude_imputations"])
        self.assertTrue(validation["no_future_imputation_reference"])
        train_only_validation = validate_window_index(
            frame,
            windows.loc[windows["split"].eq("train")],
            window_steps=4,
            horizon_steps=2,
            train_fraction=0.6,
            validation_fraction=0.2,
        ).iloc[0]
        self.assertTrue(train_only_validation["target_timestamps_disjoint"])
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
