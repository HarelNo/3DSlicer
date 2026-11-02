"""Tests for the projection output directory used by the script and executable."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class ProjectionOutputDirectoryTests(unittest.TestCase):
    # Checks that script mode creates the projections folder beside main.py.
    def test_creates_directory_beside_script(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            script_path = Path(temporary_directory) / "main.py"
            expected_directory = Path(temporary_directory) / "projections"
            with patch.object(main, "__file__", str(script_path)), patch.object(
                main.sys, "frozen", False, create=True
            ):
                output_directory = main.get_projection_output_directory()

            self.assertEqual(output_directory, expected_directory)
            self.assertTrue(expected_directory.is_dir())

    # Checks that executable mode creates the projections folder beside the EXE.
    def test_creates_directory_beside_executable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            executable_path = Path(temporary_directory) / "app.exe"
            expected_directory = Path(temporary_directory) / "projections"
            with patch.object(main.sys, "frozen", True, create=True), patch.object(
                main.sys, "executable", str(executable_path)
            ):
                output_directory = main.get_projection_output_directory()

            self.assertEqual(output_directory, expected_directory)
            self.assertTrue(expected_directory.is_dir())


if __name__ == "__main__":
    unittest.main()
