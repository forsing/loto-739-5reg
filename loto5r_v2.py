from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


SEED = 39

NUMBER_MIN = 1
NUMBER_MAX = 39
TICKET_SIZE = 7

BASE_PROBABILITY = TICKET_SIZE / NUMBER_MAX
BASE_WAIT = NUMBER_MAX / TICKET_SIZE
PRIOR_STRENGTH = 10.0

TUNING_DRAWS = 250
HOLDOUT_DRAWS = 250
REFIT_INTERVAL = 125

LOTO_CSV = Path(
    "/data/"
    "loto7_4674_k68_loto_2959.csv"
)

LOTO_PLUS_CSV = Path(
    "/data/"
    "loto7_4674_k68_loto_plus_1715.csv"
)

MODEL_NAMES = (
    "RANDOM_FOREST",
    "EXTRA_TREES",
    "GRADIENT_BOOSTING",
    "SVR",
    "KNN",
)

DISPLAY_NAMES = {
    "RANDOM_FOREST": "RandomForestRegressor",
    "EXTRA_TREES": "ExtraTreesRegressor",
    "GRADIENT_BOOSTING": "GradientBoostingRegressor",
    "SVR": "SVR",
    "KNN": "KNeighborsRegressor",
}


def distribution_entropy(
    values: Any,
) -> float:
    """Return Shannon entropy of positive distribution values."""
    array = np.asarray(
        list(values),
        dtype=float,
    )

    array = array[array > 0.0]

    if len(array) == 0:
        return 0.0

    probabilities = array / array.sum()

    return float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
    )


def load_draws(
    path: Path,
) -> list[tuple[int, ...]]:
    """Load chronological headerless 7/39 draws."""
    draws: list[tuple[int, ...]] = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.reader(csv_file)

        for row_number, row in enumerate(
            reader,
            start=1,
        ):
            try:
                draw = tuple(sorted(map(int, row)))
            except ValueError as error:
                raise ValueError(
                    f"{path}: invalid row "
                    f"{row_number}: {row}"
                ) from error

            if len(draw) != TICKET_SIZE:
                raise ValueError(
                    f"{path}: row {row_number} "
                    f"does not contain exactly "
                    f"{TICKET_SIZE} numbers"
                )

            if len(set(draw)) != TICKET_SIZE:
                raise ValueError(
                    f"{path}: duplicate number "
                    f"in row {row_number}: {draw}"
                )

            if not all(
                NUMBER_MIN <= number <= NUMBER_MAX
                for number in draw
            ):
                raise ValueError(
                    f"{path}: number outside "
                    f"{NUMBER_MIN}-{NUMBER_MAX} "
                    f"in row {row_number}: {draw}"
                )

            draws.append(draw)

    minimum_rows = (
        TUNING_DRAWS
        + HOLDOUT_DRAWS
        + 2
    )

    if len(draws) < minimum_rows:
        raise ValueError(
            f"{path}: at least {minimum_rows} "
            f"chronological draws are required"
        )

    return draws


class DistributionState:
    """Incremental distribution and draw-context state."""

    def __init__(self) -> None:
        self.draw_count = 0

        self.last_seen: dict[int, int] = {}

        self.completed_gaps: dict[int, list[int]] = {
            number: []
            for number in range(
                NUMBER_MIN,
                NUMBER_MAX + 1,
            )
        }

        self.previous_draw: tuple[int, ...] = ()

        self.previous_draw_context: tuple[float, ...] = (
            0.0,
        ) * 7

        self.transition_counts: Counter[
            tuple[int, int]
        ] = Counter()

        self.source_opportunities: Counter[int] = Counter()

        self.cooccurrence_counts: Counter[
            tuple[int, int]
        ] = Counter()

        self.appearance_opportunities: Counter[int] = Counter()

    def add_draw(
        self,
        draw: tuple[int, ...],
    ) -> None:
        current_index = self.draw_count
        old_previous = self.previous_draw

        if old_previous:
            for source_number in old_previous:
                self.source_opportunities[source_number] += 1

                for destination_number in draw:
                    self.transition_counts[
                        (
                            source_number,
                            destination_number,
                        )
                    ] += 1

        for left_index, left_number in enumerate(draw):
            self.appearance_opportunities[left_number] += 1

            for right_number in draw[left_index + 1 :]:
                self.cooccurrence_counts[
                    (
                        left_number,
                        right_number,
                    )
                ] += 1

        for number in draw:
            if number in self.last_seen:
                completed_gap = (
                    current_index
                    - self.last_seen[number]
                )

                self.completed_gaps[number].append(
                    completed_gap
                )

            self.last_seen[number] = current_index

        odd_count = sum(
            number % 2
            for number in draw
        )

        even_count = TICKET_SIZE - odd_count

        low_count = sum(
            number <= 20
            for number in draw
        )

        high_count = TICKET_SIZE - low_count

        consecutive_count = sum(
            right - left == 1
            for left, right in zip(
                draw,
                draw[1:],
            )
        )

        repeated_from_previous = (
            len(
                set(draw)
                & set(old_previous)
            )
            if old_previous
            else 0
        )

        self.previous_draw_context = (
            sum(draw) / 273.0,
            (sum(draw) / TICKET_SIZE) / NUMBER_MAX,
            (draw[-1] - draw[0]) / 38.0,
            odd_count / max(even_count, 1) / TICKET_SIZE,
            low_count / max(high_count, 1) / TICKET_SIZE,
            consecutive_count / 6.0,
            repeated_from_previous / TICKET_SIZE,
        )

        self.previous_draw = draw
        self.draw_count += 1

    def current_gap(
        self,
        number: int,
    ) -> int:
        """Return draws elapsed since the last appearance."""
        if number not in self.last_seen:
            return self.draw_count

        return (
            self.draw_count
            - 1
            - self.last_seen[number]
        )

    def gap_distribution_features(
        self,
        number: int,
    ) -> tuple[float, ...]:
        """Return gap distribution, survival, hazard and entropy."""
        current_gap = self.current_gap(number)
        gaps = self.completed_gaps[number]

        if not gaps:
            return (
                float(current_gap),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

        gap_array = np.asarray(
            gaps,
            dtype=float,
        )

        q10, q25, q50, q75, q90 = np.quantile(
            gap_array,
            [0.10, 0.25, 0.50, 0.75, 0.90],
        )

        empirical_cdf = float(
            np.mean(
                gap_array <= current_gap
            )
        )

        empirical_survival = float(
            np.mean(
                gap_array >= current_gap
            )
        )

        events_at_gap = float(
            np.sum(
                gap_array == current_gap
            )
        )

        gaps_at_risk = float(
            np.sum(
                gap_array >= current_gap
            )
        )

        empirical_hazard = (
            events_at_gap + 1.0
        ) / (
            gaps_at_risk + 2.0
        )

        gap_entropy = distribution_entropy(
            Counter(gaps).values()
        )

        return (
            float(current_gap),
            float(q10),
            float(q25),
            float(q50),
            float(q75),
            float(q90),
            empirical_cdf,
            empirical_survival,
            empirical_hazard,
            gap_entropy,
        )

    def transition_deviation(
        self,
        source_number: int,
        candidate: int,
    ) -> float:
        """Return conditional transition deviation from 7/39."""
        transition_count = self.transition_counts[
            (
                source_number,
                candidate,
            )
        ]

        source_count = self.source_opportunities[
            source_number
        ]

        conditional_probability = (
            transition_count
            + BASE_PROBABILITY * PRIOR_STRENGTH
        ) / (
            source_count + PRIOR_STRENGTH
        )

        return float(
            conditional_probability
            - BASE_PROBABILITY
        )

    def transition_distribution_features(
        self,
        candidate: int,
    ) -> tuple[float, ...]:
        """Return seven sorted conditional transition deviations."""
        if not self.previous_draw:
            return (0.0,) * TICKET_SIZE

        deviations = [
            self.transition_deviation(
                source_number,
                candidate,
            )
            for source_number in self.previous_draw
        ]

        deviations.sort()

        return tuple(deviations)

    def cooccurrence_deviation(
        self,
        source_number: int,
        candidate: int,
    ) -> float:
        """Return normalized same-draw co-occurrence deviation."""
        if source_number == candidate:
            return 0.0

        pair = tuple(
            sorted(
                (
                    source_number,
                    candidate,
                )
            )
        )

        pair_count = self.cooccurrence_counts[pair]

        source_count = self.appearance_opportunities[
            source_number
        ]

        conditional_probability = (
            pair_count
            + BASE_PROBABILITY * PRIOR_STRENGTH
        ) / (
            source_count + PRIOR_STRENGTH
        )

        return float(
            conditional_probability
            - BASE_PROBABILITY
        )

    def cooccurrence_distribution_features(
        self,
        candidate: int,
    ) -> tuple[float, ...]:
        """Return seven sorted graph co-occurrence deviations."""
        if not self.previous_draw:
            return (0.0,) * TICKET_SIZE

        deviations = [
            self.cooccurrence_deviation(
                source_number,
                candidate,
            )
            for source_number in self.previous_draw
        ]

        deviations.sort()

        return tuple(deviations)

    def features(
        self,
        number: int,
    ) -> tuple[float, ...]:
        """Return the fixed 33-feature candidate vector."""
        gap_features = self.gap_distribution_features(
            number
        )

        transition_features = (
            self.transition_distribution_features(
                number
            )
        )

        transition_entropy = distribution_entropy(
            np.abs(transition_features)
        )

        cooccurrence_features = (
            self.cooccurrence_distribution_features(
                number
            )
        )

        cooccurrence_entropy = distribution_entropy(
            np.abs(cooccurrence_features)
        )

        output = (
            *gap_features,
            *transition_features,
            transition_entropy,
            *cooccurrence_features,
            cooccurrence_entropy,
            *self.previous_draw_context,
        )

        if len(output) != 33:
            raise RuntimeError(
                f"Expected 33 features, got "
                f"{len(output)}"
            )

        return output


def build_dataset(
    draws: list[tuple[int, ...]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    DistributionState,
]:
    """Build leakage-safe feature blocks and continuous targets."""
    state = DistributionState()
    blocks: list[np.ndarray] = []

    for draw in draws:
        block = np.asarray(
            [
                state.features(number)
                for number in range(
                    NUMBER_MIN,
                    NUMBER_MAX + 1,
                )
            ],
            dtype=float,
        )

        if block.shape != (NUMBER_MAX, 33):
            raise RuntimeError(
                f"Invalid block shape: {block.shape}"
            )

        blocks.append(block)

        # Add the target draw only after its features exist.
        state.add_draw(draw)

    feature_blocks = np.stack(blocks)

    draw_count = len(draws)

    waits = np.full(
        (draw_count, NUMBER_MAX),
        np.nan,
        dtype=float,
    )

    resolution_indices = np.full(
        (draw_count, NUMBER_MAX),
        draw_count,
        dtype=int,
    )

    next_occurrence: list[int | None] = [
        None
    ] * NUMBER_MAX

    for target_index in range(
        draw_count - 1,
        -1,
        -1,
    ):
        for number in draws[target_index]:
            next_occurrence[number - 1] = target_index

        for number_index in range(NUMBER_MAX):
            resolved_at = next_occurrence[number_index]

            if resolved_at is None:
                continue

            waits[
                target_index,
                number_index,
            ] = (
                resolved_at
                - target_index
                + 1
            )

            resolution_indices[
                target_index,
                number_index,
            ] = resolved_at

    continuous_targets = np.log(
        BASE_WAIT / waits
    )

    return (
        feature_blocks,
        continuous_targets,
        resolution_indices,
        state,
    )


def parameter_candidates(
    model_name: str,
) -> list[tuple[Any, ...]]:
    """Return the approved temporal tuning candidates."""
    if model_name in {
        "RANDOM_FOREST",
        "EXTRA_TREES",
    }:
        return [
            (8, 10, "sqrt"),
            (12, 5, 0.7),
        ]

    if model_name == "GRADIENT_BOOSTING":
        return [
            (100, 2, 0.05),
            (150, 3, 0.05),
        ]

    if model_name == "SVR":
        return [
            (1.0, 0.1),
            (10.0, 0.1),
        ]

    if model_name == "KNN":
        return [
            (15, "distance"),
            (25, "distance"),
        ]

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def create_model(
    model_name: str,
    parameters: tuple[Any, ...],
):
    """Create one approved regressor."""
    if model_name == "RANDOM_FOREST":
        (
            max_depth,
            min_samples_leaf,
            max_features,
        ) = parameters

        return RandomForestRegressor(
            n_estimators=50,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=SEED,
            n_jobs=1,
        )

    if model_name == "EXTRA_TREES":
        (
            max_depth,
            min_samples_leaf,
            max_features,
        ) = parameters

        return ExtraTreesRegressor(
            n_estimators=50,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=SEED,
            n_jobs=1,
        )

    if model_name == "GRADIENT_BOOSTING":
        (
            n_estimators,
            max_depth,
            learning_rate,
        ) = parameters

        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=10,
            random_state=SEED,
        )

    if model_name == "SVR":
        c_value, epsilon = parameters

        return make_pipeline(
            StandardScaler(),
            SVR(
                C=c_value,
                epsilon=epsilon,
                gamma="scale",
            ),
        )

    if model_name == "KNN":
        n_neighbors, weights = parameters

        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                p=2,
                n_jobs=1,
            ),
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def fit_models_before_target(
    model_name: str,
    parameters: tuple[Any, ...],
    feature_blocks: np.ndarray,
    continuous_targets: np.ndarray,
    resolution_indices: np.ndarray,
    target_index: int,
) -> list[Any]:
    """Fit one leakage-safe regressor for each number."""
    models: list[Any] = []

    origin_indices = np.arange(
        1,
        target_index,
    )

    for number_index in range(NUMBER_MAX):
        resolved_mask = (
            resolution_indices[
                1:target_index,
                number_index,
            ]
            < target_index
        )

        rows = origin_indices[resolved_mask]

        model = create_model(
            model_name,
            parameters,
        )

        model.fit(
            feature_blocks[
                rows,
                number_index,
                :,
            ],
            continuous_targets[
                rows,
                number_index,
            ],
        )

        models.append(model)

    return models


def predict_all_numbers(
    models: list[Any],
    feature_block: np.ndarray,
) -> np.ndarray:
    """Predict one continuous score for every number."""
    return np.asarray(
        [
            models[number_index].predict(
                feature_block[
                    number_index : number_index + 1
                ]
            )[0]
            for number_index in range(NUMBER_MAX)
        ],
        dtype=float,
    )


def evaluate_period(
    model_name: str,
    parameters: tuple[Any, ...],
    feature_blocks: np.ndarray,
    continuous_targets: np.ndarray,
    resolution_indices: np.ndarray,
    draws: list[tuple[int, ...]],
    start_index: int,
    end_index: int,
) -> dict[str, object]:
    """Evaluate one temporal period with periodic refitting."""
    hit_counts: list[int] = []
    absolute_errors: list[float] = []

    models: list[Any] | None = None

    for target_index in range(
        start_index,
        end_index,
    ):
        should_refit = (
            models is None
            or (
                target_index - start_index
            ) % REFIT_INTERVAL == 0
        )

        if should_refit:
            models = fit_models_before_target(
                model_name=model_name,
                parameters=parameters,
                feature_blocks=feature_blocks,
                continuous_targets=continuous_targets,
                resolution_indices=resolution_indices,
                target_index=target_index,
            )

        predicted_scores = predict_all_numbers(
            models,
            feature_blocks[target_index],
        )

        ranking = np.argsort(
            -predicted_scores,
            kind="stable",
        )

        predicted_numbers = {
            int(index + 1)
            for index in ranking[:TICKET_SIZE]
        }

        actual_numbers = set(
            draws[target_index]
        )

        hit_counts.append(
            len(
                predicted_numbers
                & actual_numbers
            )
        )

        known_mask = ~np.isnan(
            continuous_targets[target_index]
        )

        absolute_errors.extend(
            np.abs(
                predicted_scores[known_mask]
                - continuous_targets[
                    target_index,
                    known_mask,
                ]
            ).tolist()
        )

    distribution = Counter(hit_counts)

    return {
        "draws": len(hit_counts),
        "average_matches": round(
            sum(hit_counts) / len(hit_counts),
            6,
        ),
        "rate_3_plus": round(
            sum(value >= 3 for value in hit_counts)
            / len(hit_counts),
            6,
        ),
        "rate_4_plus": round(
            sum(value >= 4 for value in hit_counts)
            / len(hit_counts),
            6,
        ),
        "maximum_matches": max(hit_counts),
        "target_mae": round(
            float(np.mean(absolute_errors)),
            6,
        ),
        "match_distribution": dict(
            sorted(distribution.items())
        ),
    }


def metric_key(
    metrics: dict[str, object],
) -> tuple[float, float, float, float]:
    """Return the approved tuning priority."""
    return (
        float(metrics["average_matches"]),
        float(metrics["rate_3_plus"]),
        float(metrics["rate_4_plus"]),
        -float(metrics["target_mae"]),
    )


def tune_model(
    model_name: str,
    feature_blocks: np.ndarray,
    continuous_targets: np.ndarray,
    resolution_indices: np.ndarray,
    draws: list[tuple[int, ...]],
    tuning_start: int,
    tuning_end: int,
) -> tuple[
    tuple[Any, ...],
    dict[str, dict[str, object]],
]:
    """Select parameters on the chronological tuning period."""
    candidates = parameter_candidates(model_name)

    results: dict[
        str,
        dict[str, object],
    ] = {}

    for parameters in candidates:
        metrics = evaluate_period(
            model_name=model_name,
            parameters=parameters,
            feature_blocks=feature_blocks,
            continuous_targets=continuous_targets,
            resolution_indices=resolution_indices,
            draws=draws,
            start_index=tuning_start,
            end_index=tuning_end,
        )

        results[str(parameters)] = metrics

    selected = max(
        candidates,
        key=lambda parameters: metric_key(
            results[str(parameters)]
        ),
    )

    return selected, results


def generate_next(
    model_name: str,
    selected_parameters: tuple[Any, ...],
    feature_blocks: np.ndarray,
    continuous_targets: np.ndarray,
    resolution_indices: np.ndarray,
    state: DistributionState,
) -> tuple[
    tuple[int, ...],
    list[tuple[int, float]],
]:
    """Fit on all resolved history and rank the next draw."""
    models = fit_models_before_target(
        model_name=model_name,
        parameters=selected_parameters,
        feature_blocks=feature_blocks,
        continuous_targets=continuous_targets,
        resolution_indices=resolution_indices,
        target_index=len(feature_blocks),
    )

    future_features = np.asarray(
        [
            state.features(number)
            for number in range(
                NUMBER_MIN,
                NUMBER_MAX + 1,
            )
        ],
        dtype=float,
    )

    predicted_scores = predict_all_numbers(
        models,
        future_features,
    )

    ranking = np.argsort(
        -predicted_scores,
        kind="stable",
    )

    ranked_numbers = [
        int(index + 1)
        for index in ranking
    ]

    next_prediction = tuple(
        sorted(
            ranked_numbers[:TICKET_SIZE]
        )
    )

    selected_scores = [
        (
            number,
            float(
                predicted_scores[number - 1]
            ),
        )
        for number in ranked_numbers[:TICKET_SIZE]
    ]

    return next_prediction, selected_scores


def analyze_model(
    model_name: str,
    feature_blocks: np.ndarray,
    continuous_targets: np.ndarray,
    resolution_indices: np.ndarray,
    draws: list[tuple[int, ...]],
    state: DistributionState,
    tuning_start: int,
    holdout_start: int,
) -> dict[str, object]:
    """Tune, hold out and generate one model result."""
    (
        selected_parameters,
        tuning_results,
    ) = tune_model(
        model_name=model_name,
        feature_blocks=feature_blocks,
        continuous_targets=continuous_targets,
        resolution_indices=resolution_indices,
        draws=draws,
        tuning_start=tuning_start,
        tuning_end=holdout_start,
    )

    holdout_metrics = evaluate_period(
        model_name=model_name,
        parameters=selected_parameters,
        feature_blocks=feature_blocks,
        continuous_targets=continuous_targets,
        resolution_indices=resolution_indices,
        draws=draws,
        start_index=holdout_start,
        end_index=len(draws),
    )

    (
        next_prediction,
        selected_scores,
    ) = generate_next(
        model_name=model_name,
        selected_parameters=selected_parameters,
        feature_blocks=feature_blocks,
        continuous_targets=continuous_targets,
        resolution_indices=resolution_indices,
        state=state,
    )

    return {
        "model": model_name,
        "selected_parameters": selected_parameters,
        "tuning_results": tuning_results,
        "holdout_metrics": holdout_metrics,
        "next_prediction": next_prediction,
        "selected_scores": selected_scores,
    }


def analyze_csv(
    lottery_name: str,
    path: Path,
) -> dict[str, object]:
    """Run all five regressors for one chronological CSV."""
    draws = load_draws(path)

    (
        feature_blocks,
        continuous_targets,
        resolution_indices,
        state,
    ) = build_dataset(draws)

    tuning_start = (
        len(draws)
        - TUNING_DRAWS
        - HOLDOUT_DRAWS
    )

    holdout_start = (
        len(draws)
        - HOLDOUT_DRAWS
    )

    models = {
        model_name: analyze_model(
            model_name=model_name,
            feature_blocks=feature_blocks,
            continuous_targets=continuous_targets,
            resolution_indices=resolution_indices,
            draws=draws,
            state=state,
            tuning_start=tuning_start,
            holdout_start=holdout_start,
        )
        for model_name in MODEL_NAMES
    }

    return {
        "lottery": lottery_name,
        "csv_path": str(path),
        "rows": len(draws),
        "seed": SEED,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "models": models,
    }


def format_percentage(
    value: float,
) -> str:
    """Format a decimal rate as Serbian percentage text."""
    return (
        f"{value * 100:.1f}%"
        .replace(".", ",")
    )


def format_average(
    value: float,
) -> str:
    """Format a decimal value with Serbian decimal comma."""
    return (
        f"{value:.3f}"
        .replace(".", ",")
    )


def print_next_predictions(
    analyses: list[dict[str, object]],
) -> None:
    """Print all ten NEXT predictions."""
    print()
    print("NEXT PREDIKCIJE")
    print("=" * 72)

    for analysis in analyses:
        print()
        print(analysis["lottery"])

        for model_name, result in analysis["models"].items():
            prediction = ", ".join(
                f"{number:02d}"
                for number in result["next_prediction"]
            )

            print(
                f"{DISPLAY_NAMES[model_name]:30s} "
                f"{prediction}"
            )


def print_holdout_table(
    analyses: list[dict[str, object]],
) -> None:
    """Print the approved Serbian holdout table and comment."""
    separator = (
        "╠════════════╬═══════════════════════════════"
        "╬═════════╬════════╬════════╬══════════╣"
    )

    print()
    print(
        "╔════════════╦═══════════════════════════════"
        "╦═════════╦════════╦════════╦══════════╗"
    )
    print(
        "║ IGRA       ║ REGRESOR                      "
        "║ PROSEK  ║ 3+     ║ 4+     ║ MAKSIMUM ║"
    )
    print(separator)

    first_game = True

    for analysis in analyses:
        if not first_game:
            print(separator)

        first_game = False
        lottery_name = str(analysis["lottery"])

        for model_name, result in analysis["models"].items():
            metrics = result["holdout_metrics"]

            print(
                f"║ {lottery_name:<10s} "
                f"║ {DISPLAY_NAMES[model_name]:<29s} "
                f"║ {format_average(metrics['average_matches']):<7s} "
                f"║ {format_percentage(metrics['rate_3_plus']):<6s} "
                f"║ {format_percentage(metrics['rate_4_plus']):<6s} "
                f"║ {metrics['maximum_matches']:<8d} ║"
            )

    print(
        "╚════════════╩═══════════════════════════════"
        "╩═════════╩════════╩════════╩══════════╝"
    )

    print()
    print("KOMENTAR:")
    print(
        "Rezultati su dobri jer su u očekivanim granicama "
        "za loto, koji je slučajan proces."
    )


def main() -> None:
    """Run both CSV files and print ten predictions."""
    loto_analysis = analyze_csv(
        lottery_name="Loto",
        path=LOTO_CSV,
    )

    loto_plus_analysis = analyze_csv(
        lottery_name="Loto Plus",
        path=LOTO_PLUS_CSV,
    )

    analyses = [
        loto_analysis,
        loto_plus_analysis,
    ]

    print_next_predictions(analyses)
    print_holdout_table(analyses)


if __name__ == "__main__":
    main()



"""
NEXT PREDIKCIJE
========================================================================

Loto
RandomForestRegressor          08, x, 19, y, 24, z, 39
ExtraTreesRegressor            07, x, 15, y, 26, z, 37
GradientBoostingRegressor      07, x, 23, y, 31, z, 37
SVR                            03, x, 15, y, 31, z, 39
KNeighborsRegressor            10, x, 15, y, 31, z, 39

Loto Plus
RandomForestRegressor          02, x, 32, y, 36, z, 39
ExtraTreesRegressor            04, x, 29, y, 33, z, 39
GradientBoostingRegressor      05, x, 23, y, 36, z, 39
SVR                            11, x, 29, y, 32, z, 36
KNeighborsRegressor            08, x, 31, y, 35, z, 38



Za slučajan proces 7/39 očekivani broj preseka između dve sedmočlane kombinacije je:

E[X] = (7*7)/39 = 49/7 = 1,256

Holdout

╔════════════╦═══════════════════════════════╦═════════╦════════╦════════╦══════════╗
║ IGRA       ║ REGRESOR                      ║ PROSEK  ║ 3+     ║ 4+     ║ MAKSIMUM ║
╠════════════╬═══════════════════════════════╬═════════╬════════╬════════╬══════════╣
║ Loto       ║ RandomForestRegressor         ║ 1,252   ║ 8,8%   ║ 1,2%   ║ 4        ║
║ Loto       ║ ExtraTreesRegressor           ║ 1,196   ║ 7,6%   ║ 0,0%   ║ 3        ║
║ Loto       ║ GradientBoostingRegressor     ║ 1,280   ║ 8,0%   ║ 1,6%   ║ 4        ║
║ Loto       ║ SVR                           ║ 1,264   ║ 9,6%   ║ 2,4%   ║ 4        ║
║ Loto       ║ KNeighborsRegressor           ║ 1,256   ║ 9,2%   ║ 0,8%   ║ 4        ║
╠════════════╬═══════════════════════════════╬═════════╬════════╬════════╬══════════╣
║ Loto Plus  ║ RandomForestRegressor         ║ 1,172   ║ 8,8%   ║ 0,8%   ║ 4        ║
║ Loto Plus  ║ ExtraTreesRegressor           ║ 1,256   ║ 10,0%  ║ 1,2%   ║ 4        ║
║ Loto Plus  ║ GradientBoostingRegressor     ║ 1,188   ║ 8,4%   ║ 0,0%   ║ 3        ║
║ Loto Plus  ║ SVR                           ║ 1,296   ║ 10,0%  ║ 2,0%   ║ 5        ║
║ Loto Plus  ║ KNeighborsRegressor           ║ 1,368   ║ 10,8%  ║ 0,8%   ║ 4        ║
╚════════════╩═══════════════════════════════╩═════════╩════════╩════════╩══════════╝

KOMENTAR:
Rezultati su dobri jer su u očekivanim granicama za loto, koji je slučajan proces.
"""



"""
- pet regresora;
- poseban regresor za svaki broj 1-39;
- oba kompletna CSV fajla;
- ukupno 10 NEXT predikcija;
- kontinuirana time-to-event meta;
- gap raspodela, survival, hazard i entropija;
- uslovne raspodele prelaza;
- normalizovano ko-pojavljivanje i njegova entropija;
- sedam strukturnih osobina prethodnog izvlačenja;
- vremensko podešavanje i odvojeni holdout;
- stabilno rangiranje;
- StandardScaler samo za SVR i KNN;
- seed=39;
"""
