from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.contracts.golden_set import GoldenSetRow
from src.shared.normalization import is_empty_reference


class GoldenSetLoaderError(Exception):
    """Raised when the Golden Set cannot be loaded or validated."""


class GoldenSetLoader:
    """Loads and validates the Golden Set Excel file."""

    REQUIRED_COLUMNS = {
        "Question_ID",
        "Sample_Query",
        "Ground_Truth_Answer",
        "Source_Document",
        "Page_Number/Section",
    }

    def load(self, path: Path) -> list[GoldenSetRow]:
        """
        Load the Golden Set from an Excel file.

        Args:
            path: Path to the Golden Set Excel file.

        Returns:
            List of GoldenSetRow objects.

        Raises:
            GoldenSetLoaderError:
                If the file is invalid or cannot be loaded.
        """

        if not path.exists():
            raise GoldenSetLoaderError(
                f"Golden Set file not found: {path}"
            )

        try:
            dataframe = pd.read_excel(path)
        except Exception as exc:
            raise GoldenSetLoaderError(
                f"Failed to read Golden Set: {exc}"
            ) from exc

        self._validate_columns(dataframe)

        rows: list[GoldenSetRow] = []

        for index, row in dataframe.iterrows():
            rows.append(self._to_contract(index + 2, row))

        return rows

    def _validate_columns(self, dataframe: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(dataframe.columns)

        if missing:
            raise GoldenSetLoaderError(
                f"Missing required columns: {sorted(missing)}"
            )

    def _to_contract(
        self,
        excel_row: int,
        row: pd.Series,
    ) -> GoldenSetRow:

        self._validate_required_field(
            excel_row,
            row,
            "Question_ID",
        )

        self._validate_required_field(
            excel_row,
            row,
            "Sample_Query",
        )

        self._validate_required_field(
            excel_row,
            row,
            "Ground_Truth_Answer",
        )

        self._validate_required_field(
            excel_row,
            row,
            "Source_Document",
        )

        page_reference = str(row["Page_Number/Section"]).strip()

        # Allow empty or N/A page references.
        if not is_empty_reference(page_reference):
            pass
        return GoldenSetRow(
            question_id=str(row["Question_ID"]).strip(),
            question=str(row["Sample_Query"]).strip(),
            ground_truth_answer=str(row["Ground_Truth_Answer"]).strip(),
            source_document=str(row["Source_Document"]).strip(),
            source_reference=str(row["Page_Number/Section"]).strip(),
        )

    @staticmethod
    def _validate_required_field(
        excel_row: int,
        row: pd.Series,
        column: str,
    ) -> None:

        value = row[column]

        if pd.isna(value) or str(value).strip() == "":
            raise GoldenSetLoaderError(
                f"Row {excel_row}: '{column}' cannot be empty."
            )