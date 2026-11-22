import unittest

import numpy as np

from src.tariff import (
    TariffConfig,
    classify_time_period,
    power_factor_adjustment_rate,
)


class TariffModelTests(unittest.TestCase):
    def test_contract_time_periods(self):
        self.assertEqual(classify_time_period("2026-01-12 08:00"), "full")
        self.assertEqual(classify_time_period("2026-01-12 19:00"), "peak_evening")
        self.assertEqual(classify_time_period("2026-01-12 22:00"), "offpeak")
        self.assertEqual(classify_time_period("2026-07-13 10:00"), "peak_morning")
        self.assertEqual(classify_time_period("2026-07-13 20:00"), "peak_evening")
        self.assertEqual(classify_time_period("2026-07-12 12:00"), "offpeak")

    def test_power_factor_rules(self):
        values = power_factor_adjustment_rate(np.array([0.94, 0.85, 0.78, 0.74, 0.59]))
        np.testing.assert_allclose(values, [-0.02, 0.0, 0.005, 0.02, 0.215], rtol=1e-7)

    def test_invoice_tariff_parameters(self):
        tariff = TariffConfig(
            peak_price=0.291,
            full_price=0.291,
            offpeak_price=0.291,
            evening_peak_price=0.291,
            subscribed_power_price=5.0,
            municipal_surcharge=0.010,
            tariff_mode="uniforme_facture",
        )
        self.assertEqual(tariff.full_price, 0.291)
        self.assertEqual(tariff.subscribed_power_price, 5.0)
        self.assertEqual(tariff.municipal_surcharge, 0.010)


if __name__ == "__main__":
    unittest.main()
