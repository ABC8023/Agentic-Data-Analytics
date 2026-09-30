"""Loading rules for uploaded CSV and Excel files."""

import io
import os
import tempfile
import unittest

import pandas as pd

import file_io


class FakeUpload(io.BytesIO):
    """Stand-in for Streamlit's UploadedFile, which is bytes plus a name."""

    def __init__(self, payload: bytes, name: str):
        super().__init__(payload)
        self.name = name
        self.size = len(payload)


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "region": ["north", "south", "east", "west"] * 5,
        "amount": [10.5, 20.25, 30.0, 40.75] * 5,
        "label": ["a", "b"] * 10,
    })


class TestExtensionRules(unittest.TestCase):

    def test_supported_extensions(self):
        self.assertTrue(file_io.is_supported("data.csv"))
        self.assertTrue(file_io.is_supported("book.XLSX"))
        self.assertTrue(file_io.is_supported("book.xlsm"))

    def test_unsupported_extensions(self):
        self.assertFalse(file_io.is_supported("data.json"))
        self.assertFalse(file_io.is_supported("data.parquet"))
        self.assertFalse(file_io.is_supported("data"))
        self.assertFalse(file_io.is_supported(""))

    def test_excel_detection(self):
        self.assertTrue(file_io.is_excel("b.xlsx"))
        self.assertTrue(file_io.is_excel("b.xlsm"))
        self.assertFalse(file_io.is_excel("b.csv"))

    def test_formats_are_advertised_in_reading_order(self):
        self.assertEqual(
            file_io.describe_supported_formats(),
            "CSV, XLSX, XLSM"
        )


class TestDelimiterSniffing(unittest.TestCase):

    def test_each_candidate_delimiter_is_detected(self):
        for delimiter in [",", ";", "\t", "|"]:
            with self.subTest(delimiter=delimiter):
                text = frame().to_csv(index=False, sep=delimiter)

                self.assertEqual(
                    file_io.sniff_delimiter(text[:4096]),
                    delimiter
                )

    def test_each_candidate_delimiter_parses_to_the_same_shape(self):
        expected = frame().shape

        for delimiter in [",", ";", "\t", "|"]:
            with self.subTest(delimiter=delimiter):
                payload = frame().to_csv(
                    index=False,
                    sep=delimiter
                ).encode("utf-8")

                result = file_io.load_tabular_file(
                    FakeUpload(payload, "d.csv")
                )

                self.assertEqual(result["dataframe"].shape, expected)

    def test_empty_sample_falls_back_to_comma(self):
        self.assertEqual(file_io.sniff_delimiter("   "), ",")


class TestEncodings(unittest.TestCase):

    def test_accented_text_survives_common_encodings(self):
        accented = pd.DataFrame({
            "city": ["Köln", "Málaga"],
            "n": [1, 2],
        })

        for encoding in ["utf-8", "utf-8-sig", "cp1252"]:
            with self.subTest(encoding=encoding):
                payload = accented.to_csv(
                    index=False
                ).encode(encoding)

                result = file_io.load_tabular_file(
                    FakeUpload(payload, "a.csv")
                )

                self.assertEqual(
                    list(result["dataframe"]["city"]),
                    ["Köln", "Málaga"]
                )


class TestCeilings(unittest.TestCase):

    def test_oversized_file_is_refused_with_its_size(self):
        too_big = (file_io.MAX_FILE_SIZE_MB + 3) * 1024 ** 2

        with self.assertRaises(file_io.FileLoadError) as caught:
            file_io.check_file_size(too_big)

        self.assertIn("smaller than", str(caught.exception))

    def test_empty_file_is_refused(self):
        with self.assertRaises(file_io.FileLoadError):
            file_io.check_file_size(0)

    def test_row_cap_truncates_and_says_so(self):
        original_cap = file_io.MAX_ROWS
        file_io.MAX_ROWS = 12

        try:
            payload = frame().to_csv(index=False).encode("utf-8")

            result = file_io.load_tabular_file(
                FakeUpload(payload, "big.csv")
            )

            self.assertEqual(len(result["dataframe"]), 12)
            self.assertEqual(result["original_rows"], 20)
            self.assertTrue(result["truncated"])
            self.assertTrue(
                any("first 12 rows" in note for note in result["notes"])
            )

        finally:
            file_io.MAX_ROWS = original_cap

    def test_a_table_within_the_cap_is_not_flagged(self):
        payload = frame().to_csv(index=False).encode("utf-8")

        result = file_io.load_tabular_file(
            FakeUpload(payload, "small.csv")
        )

        self.assertFalse(result["truncated"])
        self.assertEqual(result["notes"], [])


class TestRejections(unittest.TestCase):

    def test_empty_payload(self):
        with self.assertRaises(file_io.FileLoadError) as caught:
            file_io.load_tabular_file(FakeUpload(b"", "empty.csv"))

        self.assertIn("empty", str(caught.exception).lower())

    def test_headers_without_rows(self):
        with self.assertRaises(file_io.FileLoadError) as caught:
            file_io.load_tabular_file(
                FakeUpload(b"a,b\n", "headers.csv")
            )

        self.assertIn("no data rows", str(caught.exception))

    def test_unsupported_format_names_the_alternatives(self):
        with self.assertRaises(file_io.FileLoadError) as caught:
            file_io.load_tabular_file(FakeUpload(b"x", "notes.json"))

        message = str(caught.exception)

        self.assertIn("not a supported format", message)
        self.assertIn("CSV", message)


class TestExcel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp()
        cls.xlsx_path = os.path.join(cls.directory, "book.xlsx")
        cls.xlsm_path = os.path.join(cls.directory, "book.xlsm")

        with pd.ExcelWriter(cls.xlsx_path, engine="openpyxl") as writer:
            frame().to_excel(writer, sheet_name="Sales", index=False)
            frame().head(7).to_excel(
                writer,
                sheet_name="Returns",
                index=False
            )

        with pd.ExcelWriter(cls.xlsm_path, engine="openpyxl") as writer:
            frame().to_excel(writer, sheet_name="Data", index=False)

    def test_worksheets_are_listed_in_workbook_order(self):
        self.assertEqual(
            file_io.list_worksheets(self.xlsx_path),
            ["Sales", "Returns"]
        )

    def test_first_worksheet_is_the_default(self):
        result = file_io.load_tabular_file(self.xlsx_path)

        self.assertEqual(result["worksheet"], "Sales")
        self.assertEqual(len(result["dataframe"]), 20)

    def test_named_worksheet_is_honoured(self):
        result = file_io.load_tabular_file(
            self.xlsx_path,
            worksheet="Returns"
        )

        self.assertEqual(result["worksheet"], "Returns")
        self.assertEqual(len(result["dataframe"]), 7)

    def test_worksheet_names_are_reported(self):
        result = file_io.load_tabular_file(self.xlsx_path)

        self.assertEqual(
            result["worksheet_names"],
            ["Sales", "Returns"]
        )

    def test_missing_worksheet_lists_what_is_available(self):
        with self.assertRaises(file_io.FileLoadError) as caught:
            file_io.load_tabular_file(
                self.xlsx_path,
                worksheet="Nope"
            )

        message = str(caught.exception)

        self.assertIn("no worksheet called", message)
        self.assertIn("Sales", message)

    def test_macro_workbook_loads_and_reports_ignored_macros(self):
        result = file_io.load_tabular_file(self.xlsm_path)

        self.assertEqual(result["source_format"], "xlsm")
        self.assertEqual(len(result["dataframe"]), 20)
        self.assertTrue(
            any("Macros" in note for note in result["notes"])
        )


if __name__ == "__main__":
    unittest.main()
