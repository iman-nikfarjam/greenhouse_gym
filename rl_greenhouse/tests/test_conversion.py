import unittest

from rl_greenhouse.greenhouse.physics.unit_conversion import gpm3_to_ppm, ppm_to_gpm3


class ConversionTests(unittest.TestCase):

    def test_carbon_conversion(self):
        for ppm in range(0, 2000, 10):
            self.assertAlmostEqual(gpm3_to_ppm(ppm_to_gpm3(ppm)), ppm, places=8)
