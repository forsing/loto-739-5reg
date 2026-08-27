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
    "ENSEMBLE": "Ponderisani ansambl",
}


def distribution_entropy(values: Any) -> float:
    """Return Shannon entropy of positive distribution values."""
    array = np.asarray(list(values), dtype=float)
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


def load_draws(path: Path) -> list[tuple[int, ...]]:
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

    def current_gap(self, number: int) -> int:
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

        # The target draw is added only after its features exist.
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
    """Return temporal tuning candidates."""
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
) -> Any:
    """Create one regressor."""
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

        if len(rows) == 0:
            raise RuntimeError(
                f"No resolved training rows for "
                f"number {number_index + 1}"
            )

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
                    number_index:
                    number_index + 1
                ]
            )[0]
            for number_index in range(NUMBER_MAX)
        ],
        dtype=float,
    )


def calculate_metrics(
    predicted_score_rows: np.ndarray,
    draws: list[tuple[int, ...]],
    continuous_targets: np.ndarray,
    start_index: int,
) -> dict[str, object]:
    """Calculate ranking and continuous-target metrics."""
    hit_counts: list[int] = []
    absolute_errors: list[float] = []

    for offset, predicted_scores in enumerate(
        predicted_score_rows
    ):
        target_index = start_index + offset

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
            float(np.mean(hit_counts)),
            6,
        ),
        "rate_3_plus": round(
            float(
                np.mean(
                    np.asarray(hit_counts) >= 3
                )
            ),
            6,
        ),
        "rate_4_plus": round(
            float(
                np.mean(
                    np.asarray(hit_counts) >= 4
                )
            ),
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
    """Evaluate one chronological period with periodic refitting."""
    predicted_score_rows: list[np.ndarray] = []
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

        predicted_score_rows.append(
            predict_all_numbers(
                models,
                feature_blocks[target_index],
            )
        )

    score_matrix = np.vstack(
        predicted_score_rows
    )

    metrics = calculate_metrics(
        predicted_score_rows=score_matrix,
        draws=draws,
        continuous_targets=continuous_targets,
        start_index=start_index,
    )

    metrics["score_matrix"] = score_matrix
    return metrics


def metric_key(
    metrics: dict[str, object],
) -> tuple[float, float, float, float]:
    """Return tuning priority without using future results."""
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
    """Select parameters on a strictly earlier time period."""
    candidates = parameter_candidates(
        model_name
    )

    results: dict[
        str,
        dict[str, object],
    ] = {}

    for parameters in candidates:
        result = evaluate_period(
            model_name=model_name,
            parameters=parameters,
            feature_blocks=feature_blocks,
            continuous_targets=continuous_targets,
            resolution_indices=resolution_indices,
            draws=draws,
            start_index=tuning_start,
            end_index=tuning_end,
        )

        result_without_scores = {
            key: value
            for key, value in result.items()
            if key != "score_matrix"
        }

        results[
            str(parameters)
        ] = result_without_scores

    selected = max(
        candidates,
        key=lambda parameters: metric_key(
            results[str(parameters)]
        ),
    )

    return selected, results


def next_feature_block(
    state: DistributionState,
) -> np.ndarray:
    """Return features for the unseen next draw."""
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
            f"Invalid next block shape: "
            f"{block.shape}"
        )

    return block


def prediction_from_scores(
    predicted_scores: np.ndarray,
) -> tuple[int, ...]:
    """Return seven stably ranked numbers."""
    ranking = np.argsort(
        -predicted_scores,
        kind="stable",
    )

    return tuple(
        sorted(
            int(index + 1)
            for index in ranking[:TICKET_SIZE]
        )
    )




def analyze_model_nested(
    model_name: str,
    feature_blocks: np.ndarray,
    continuous_targets: np.ndarray,
    resolution_indices: np.ndarray,
    draws: list[tuple[int, ...]],
    state: DistributionState,
    holdout_start: int,
) -> dict[str, object]:
    """Run nested walk-forward holdout and generate NEXT scores."""
    holdout_score_rows: list[np.ndarray] = []
    nested_selections: list[
        tuple[int, tuple[Any, ...]]
    ] = []

    models: list[Any] | None = None

    for target_index in range(
        holdout_start,
        len(draws),
    ):
        should_refit = (
            models is None
            or (
                target_index - holdout_start
            ) % REFIT_INTERVAL == 0
        )

        if should_refit:
            tuning_start = (
                target_index
                - TUNING_DRAWS
            )

            (
                selected_parameters,
                _,
            ) = tune_model(
                model_name=model_name,
                feature_blocks=feature_blocks,
                continuous_targets=continuous_targets,
                resolution_indices=resolution_indices,
                draws=draws,
                tuning_start=tuning_start,
                tuning_end=target_index,
            )

            nested_selections.append(
                (
                    target_index,
                    selected_parameters,
                )
            )

            models = fit_models_before_target(
                model_name=model_name,
                parameters=selected_parameters,
                feature_blocks=feature_blocks,
                continuous_targets=continuous_targets,
                resolution_indices=resolution_indices,
                target_index=target_index,
            )

        holdout_score_rows.append(
            predict_all_numbers(
                models,
                feature_blocks[target_index],
            )
        )

    holdout_scores = np.vstack(
        holdout_score_rows
    )

    holdout_metrics = calculate_metrics(
        predicted_score_rows=holdout_scores,
        draws=draws,
        continuous_targets=continuous_targets,
        start_index=holdout_start,
    )

    final_tuning_start = (
        len(draws)
        - TUNING_DRAWS
    )

    (
        final_parameters,
        final_tuning_results,
    ) = tune_model(
        model_name=model_name,
        feature_blocks=feature_blocks,
        continuous_targets=continuous_targets,
        resolution_indices=resolution_indices,
        draws=draws,
        tuning_start=final_tuning_start,
        tuning_end=len(draws),
    )

    final_models = fit_models_before_target(
        model_name=model_name,
        parameters=final_parameters,
        feature_blocks=feature_blocks,
        continuous_targets=continuous_targets,
        resolution_indices=resolution_indices,
        target_index=len(draws),
    )

    future_features = next_feature_block(
        state
    )

    next_scores = predict_all_numbers(
        final_models,
        future_features,
    )

    next_prediction = prediction_from_scores(
        next_scores
    )

    ranking = np.argsort(
        -next_scores,
        kind="stable",
    )

    selected_scores = [
        (
            int(index + 1),
            float(next_scores[index]),
        )
        for index in ranking[:TICKET_SIZE]
    ]

    return {
        "model": model_name,
        "final_parameters": final_parameters,
        "final_tuning_results": final_tuning_results,
        "nested_selections": nested_selections,
        "holdout_metrics": holdout_metrics,
        "holdout_scores": holdout_scores,
        "next_scores": next_scores,
        "next_prediction": next_prediction,
        "selected_scores": selected_scores,
    }


def normalize_score_rows(
    scores: np.ndarray,
) -> np.ndarray:
    """Standardize every 39-number score row separately."""
    means = np.mean(
        scores,
        axis=-1,
        keepdims=True,
    )

    standard_deviations = np.std(
        scores,
        axis=-1,
        keepdims=True,
    )

    safe_standard_deviations = np.where(
        standard_deviations > 0.0,
        standard_deviations,
        1.0,
    )

    return (
        scores - means
    ) / safe_standard_deviations


def calculate_ensemble_weights(
    models: dict[str, dict[str, object]],
) -> dict[str, float]:
    """Calculate inverse-holdout-MAE ensemble weights."""
    inverse_errors = np.asarray(
        [
            1.0
            / max(
                float(
                    models[model_name][
                        "holdout_metrics"
                    ]["target_mae"]
                ),
                np.finfo(float).eps,
            )
            for model_name in MODEL_NAMES
        ],
        dtype=float,
    )

    normalized_weights = (
        inverse_errors
        / inverse_errors.sum()
    )

    return {
        model_name: float(weight)
        for model_name, weight in zip(
            MODEL_NAMES,
            normalized_weights,
        )
    }


def build_ensemble(
    models: dict[str, dict[str, object]],
    draws: list[tuple[int, ...]],
    continuous_targets: np.ndarray,
    holdout_start: int,
) -> dict[str, object]:
    """Combine normalized model scores using holdout weights."""
    weights = calculate_ensemble_weights(
        models
    )

    holdout_stack = np.stack(
        [
            models[model_name][
                "holdout_scores"
            ]
            for model_name in MODEL_NAMES
        ],
        axis=0,
    )

    normalized_holdout_stack = (
        normalize_score_rows(
            holdout_stack
        )
    )

    weight_vector = np.asarray(
        [
            weights[model_name]
            for model_name in MODEL_NAMES
        ],
        dtype=float,
    )

    ensemble_holdout_scores = np.tensordot(
        weight_vector,
        normalized_holdout_stack,
        axes=(0, 0),
    )

    holdout_metrics = calculate_metrics(
        predicted_score_rows=ensemble_holdout_scores,
        draws=draws,
        continuous_targets=continuous_targets,
        start_index=holdout_start,
    )

    next_stack = np.stack(
        [
            models[model_name][
                "next_scores"
            ]
            for model_name in MODEL_NAMES
        ],
        axis=0,
    )

    normalized_next_stack = (
        normalize_score_rows(
            next_stack
        )
    )

    ensemble_next_scores = np.tensordot(
        weight_vector,
        normalized_next_stack,
        axes=(0, 0),
    )

    next_prediction = prediction_from_scores(
        ensemble_next_scores
    )

    ranking = np.argsort(
        -ensemble_next_scores,
        kind="stable",
    )

    selected_scores = [
        (
            int(index + 1),
            float(
                ensemble_next_scores[index]
            ),
        )
        for index in ranking[:TICKET_SIZE]
    ]

    return {
        "model": "ENSEMBLE",
        "weights": weights,
        "holdout_metrics": holdout_metrics,
        "holdout_scores": ensemble_holdout_scores,
        "next_scores": ensemble_next_scores,
        "next_prediction": next_prediction,
        "selected_scores": selected_scores,
    }


def analyze_csv(
    lottery_name: str,
    path: Path,
) -> dict[str, object]:
    """Run five nested regressors and one ensemble."""
    draws = load_draws(path)

    (
        feature_blocks,
        continuous_targets,
        resolution_indices,
        state,
    ) = build_dataset(draws)

    holdout_start = (
        len(draws)
        - HOLDOUT_DRAWS
    )

    models: dict[
        str,
        dict[str, object],
    ] = {}

    for model_name in MODEL_NAMES:
        print(
            f"Obrada: {lottery_name} - "
            f"{DISPLAY_NAMES[model_name]}",
            flush=True,
        )

        models[model_name] = (
            analyze_model_nested(
                model_name=model_name,
                feature_blocks=feature_blocks,
                continuous_targets=continuous_targets,
                resolution_indices=resolution_indices,
                draws=draws,
                state=state,
                holdout_start=holdout_start,
            )
        )

    ensemble = build_ensemble(
        models=models,
        draws=draws,
        continuous_targets=continuous_targets,
        holdout_start=holdout_start,
    )

    models["ENSEMBLE"] = ensemble

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
    """Format a decimal rate with a decimal comma."""
    return (
        f"{value * 100:.1f}%"
        .replace(".", ",")
    )


def format_average(
    value: float,
) -> str:
    """Format a decimal value with a decimal comma."""
    return (
        f"{value:.3f}"
        .replace(".", ",")
    )


def print_next_predictions(
    analyses: list[dict[str, object]],
) -> None:
    """Print all twelve NEXT predictions."""
    print()
    print("NEXT PREDIKCIJE")
    print("=" * 72)

    for analysis in analyses:
        print()
        print(analysis["lottery"])

        for model_name in (
            *MODEL_NAMES,
            "ENSEMBLE",
        ):
            result = analysis[
                "models"
            ][model_name]

            prediction = ", ".join(
                f"{number:02d}"
                for number in result[
                    "next_prediction"
                ]
            )

            print(
                f"{DISPLAY_NAMES[model_name]:30s} "
                f"{prediction}"
            )


def print_ensemble_weights(
    analyses: list[dict[str, object]],
) -> None:
    """Print holdout-derived ensemble weights."""
    print()
    print("TEŽINE PONDERISANOG ANSAMBLA")
    print("=" * 72)

    for analysis in analyses:
        print()
        print(analysis["lottery"])

        weights = analysis[
            "models"
        ]["ENSEMBLE"]["weights"]

        for model_name in MODEL_NAMES:
            print(
                f"{DISPLAY_NAMES[model_name]:30s} "
                f"{format_percentage(weights[model_name])}"
            )


def print_holdout_table(
    analyses: list[dict[str, object]],
) -> None:
    """Print the holdout table and the required comment."""
    horizontal = (
        "╠════════════╬═══════════════════════════════"
        "╬═════════╬════════╬════════╬══════════╣"
    )

    print()
    print("HOLDOUT")
    print()
    print(
        "╔════════════╦═══════════════════════════════"
        "╦═════════╦════════╦════════╦══════════╗"
    )
    print(
        "║ IGRA       ║ REGRESOR / ANSAMBL            "
        "║ PROSEK  ║ 3+     ║ 4+     ║ MAKSIMUM ║"
    )
    print(horizontal)

    first_game = True

    for analysis in analyses:
        if not first_game:
            print(horizontal)

        first_game = False
        lottery_name = str(
            analysis["lottery"]
        )

        for model_name in (
            *MODEL_NAMES,
            "ENSEMBLE",
        ):
            result = analysis[
                "models"
            ][model_name]

            metrics = result[
                "holdout_metrics"
            ]

            print(
                f"║ {lottery_name:<10s} "
                f"║ {DISPLAY_NAMES[model_name]:<29s} "
                f"║ "
                f"{format_average(metrics['average_matches']):<7s} "
                f"║ "
                f"{format_percentage(metrics['rate_3_plus']):<6s} "
                f"║ "
                f"{format_percentage(metrics['rate_4_plus']):<6s} "
                f"║ "
                f"{metrics['maximum_matches']:<8d} ║"
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
    """Run both CSV files and print twelve predictions."""
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
    print_ensemble_weights(analyses)
    print_holdout_table(analyses)


if __name__ == "__main__":
    main()



"""
RUN:


"""
