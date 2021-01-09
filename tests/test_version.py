import unittest

import muscle_synergies


class VersionTestCase(unittest.TestCase):
    """ Version tests """

    def test_version(self):
        """ check muscle_synergies exposes a version attribute """
        self.assertTrue(hasattr(muscle_synergies, "__version__"))
        self.assertIsInstance(muscle_synergies.__version__, str)


if __name__ == "__main__":
    unittest.main()
