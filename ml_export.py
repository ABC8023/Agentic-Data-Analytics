"""
Download a trained pipeline, with a model card that says what it is.

The model is saved in the skops format rather than pickle. Loading a
pickle can run any code the file contains. A skops file is read only
after its object types are listed and explicitly trusted, so a
tampered file fails to load instead of running.

skops is imported when a file is written or read, not when the app
starts, so the rest of the app works without it.
"""

import hashlib
import io
import json
import platform
import zipfile
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

MODEL_FILE = "model.skops"
CARD_FILE = "model_card.json"
GUIDE_FILE = "HOW_TO_LOAD.md"

# Type prefixes a model trained by this app may contain. Anything else
# in a file means it was not written here, and it is refused.
TRUSTED_TYPE_PREFIXES = ("numpy.", "sklearn.", "scipy.")


class UntrustedModelError(ValueError):
    """A model file holds object types outside the allowlist."""


def training_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    """
    Hash the training rows, so a model can be matched to its data.

    The hash covers values, column names and row order, not the index.
    Two exports trained on the same rows carry the same fingerprint.

    Args:
        X:
            Training features.

        y:
            Training target.

    Returns:
        "sha256:" followed by the hex digest.
    """

    digest = hashlib.sha256()

    digest.update(json.dumps([str(column) for column in X.columns]).encode())
    digest.update(
        pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes()
    )
    digest.update(
        pd.util.hash_pandas_object(y, index=False).to_numpy().tobytes()
    )

    return f"sha256:{digest.hexdigest()}"


def json_safe(value: Any) -> Any:
    """Convert a value for the card, falling back to its repr."""

    if isinstance(value, np.generic):
        return value.item()

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    return repr(value)


def library_versions() -> dict[str, str]:
    """Versions a reader needs to load the model the same way."""

    import sklearn
    import skops

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "skops": skops.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def model_card(
    ml_result: dict[str, Any],
    model_name: str,
    trusted_types: list[str],
) -> dict[str, Any]:
    """
    Describe one trained model: data, features, scores and settings.

    Args:
        ml_result:
            A successful result from train_classification_model or
            train_regression_model.

        model_name:
            Which of the trained models the card describes.

        trusted_types:
            Object types in the saved file, for the reader to compare
            against before loading it.

    Returns:
        A JSON-safe dictionary.
    """

    pipeline = ml_result["trained_models"][model_name]

    scores = next(
        row
        for row in ml_result["leaderboard"]
        if row["model"] == model_name
    )

    card = {
        "model": model_name,
        "task": ml_result["task"],
        "target_column": ml_result["target_column"],
        "created_utc": datetime.now(UTC).isoformat(
            timespec="seconds"
        ),
        "chosen_as_best": model_name == ml_result["best_model_name"],
        "ranked_by": ml_result.get("ranked_by"),
        "features": {
            "numeric": ml_result["numeric_columns"],
            "categorical": ml_result["categorical_columns"],
            "dates_converted_to_seconds": ml_result[
                "datetime_columns_converted"
            ],
            "removed": {
                "constant": ml_result["constant_columns_removed"],
                "identifier_like": ml_result[
                    "high_cardinality_columns_removed"
                ],
                "unsupported_type": ml_result[
                    "unsupported_columns_removed"
                ],
            },
        },
        "rows": {
            "training": ml_result["training_rows"],
            "testing": ml_result["testing_rows"],
        },
        "training_data": ml_result.get("training_fingerprint"),
        "scores": {
            key: value
            for key, value in scores.items()
            if key != "model"
        },
        "cross_validation": ml_result.get("cross_validation"),
        "tuned_parameters": ml_result.get(
            "tuned_parameters",
            {}
        ).get(model_name, {}),
        "model_parameters": json_safe(
            pipeline.named_steps["model"].get_params()
        ),
        "trusted_types": trusted_types,
        "library_versions": library_versions(),
        "caveats": [
            (
                "Scores describe one dataset and one split. Check them "
                "on new data before relying on the model."
            ),
            (
                "Date columns listed above must be converted to seconds "
                "since 1970-01-01 before calling predict, as in training."
            ),
            "The model has not been checked for drift or fairness.",
        ],
    }

    if ml_result["task"] == "classification":
        card["class_labels"] = json_safe(ml_result["class_labels"])

    return json_safe(card)


def loading_guide(card: dict[str, Any]) -> str:
    """A short guide to loading the model safely."""

    columns = (
        card["features"]["numeric"]
        + card["features"]["categorical"]
    )

    return (
        f"# {card['model']} for {card['target_column']}\n"
        "\n"
        "Load the model with skops, after checking the object types it "
        "contains. Every type should appear under `trusted_types` in "
        f"`{CARD_FILE}`, and should come from numpy, scipy or "
        "scikit-learn.\n"
        "\n"
        "```python\n"
        "import pandas as pd\n"
        "import skops.io as sio\n"
        "\n"
        f'types = sio.get_untrusted_types(file="{MODEL_FILE}")\n'
        "print(types)  # check this list before trusting it\n"
        f'model = sio.load("{MODEL_FILE}", trusted=types)\n'
        "\n"
        f"rows = pd.DataFrame(columns={json.dumps(columns)})\n"
        "# Fill rows with new data, then:\n"
        "# model.predict(rows)\n"
        "```\n"
        "\n"
        "Install the versions under `library_versions` in the card. "
        "A different scikit-learn version may refuse the file or "
        "predict differently.\n"
    )


def export_model(ml_result: dict[str, Any], model_name: str) -> bytes:
    """
    Package a trained model as a zip: the skops file, card and guide.

    Args:
        ml_result:
            A successful training result that still holds its pipelines.

        model_name:
            The model to export.

    Returns:
        The zip file as bytes.
    """

    import skops.io as sio

    pipeline = ml_result["trained_models"][model_name]

    model_bytes = sio.dumps(pipeline)
    trusted_types = sorted(sio.get_untrusted_types(data=model_bytes))

    card = model_card(ml_result, model_name, trusted_types)

    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MODEL_FILE, model_bytes)
        archive.writestr(CARD_FILE, json.dumps(card, indent=2))
        archive.writestr(GUIDE_FILE, loading_guide(card))

    return buffer.getvalue()


def load_model(model_bytes: bytes) -> Any:
    """
    Load a skops model written by this app, refusing unfamiliar types.

    Args:
        model_bytes:
            Contents of a model.skops file.

    Returns:
        The fitted pipeline.

    Raises:
        UntrustedModelError:
            If the file holds a type outside TRUSTED_TYPE_PREFIXES.
    """

    import skops.io as sio

    types = sio.get_untrusted_types(data=model_bytes)

    unexpected = [
        name
        for name in types
        if not name.startswith(TRUSTED_TYPE_PREFIXES)
    ]

    if unexpected:
        raise UntrustedModelError(
            "The model file holds types this app never writes: "
            + ", ".join(sorted(unexpected))
        )

    return sio.loads(model_bytes, trusted=types)
