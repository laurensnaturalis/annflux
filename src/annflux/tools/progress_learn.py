import logging
from typing import Tuple

import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor

from annflux.tools.core import AnnFluxState

time_estimators = []
states = []


def estimate_duration(annflux_state: AnnFluxState, step_state: Tuple[str, int, int], logger: logging.Logger):
    global time_estimators, states
    if len(time_estimators) == 0:
        timings = pandas.read_csv(annflux_state.timings_path)
        timings.dropna(inplace=True)
        timings.reset_index(drop=True, inplace=True)
        states = []
        transitions = []
        durations = []
        for r, row in timings.iterrows():
            if (
                r > 0
                and row.status != "idle"
                and timings.loc[r - 1, "status"] != "idle"
            ):
                # state = f"{timings.loc[r - 1, 'status']}-{row.status}"
                state = timings.loc[r - 1, "status"]
                if state not in states:
                    states.append(state)
                transitions.append(
                    (
                        states.index(state),
                        float(row.num_total),
                        float(row.num_labeled),
                    )
                )
                durations.append(
                    row.timestamp - timings.loc[r - 1, "timestamp"],
                )
        logger.info(f"estimate_duration: transitions={transitions[:-10]}")
        x = np.vstack(transitions)

        # time_estimators = []
        for _ in range(10):
            rf = RandomForestRegressor()
            rf.fit(x, np.array(durations))
            time_estimators.append(rf)

    if step_state[0] in states:
        in_features = (
            np.array(
                [
                    states.index(step_state[0]),
                    float(step_state[1]),
                    float(step_state[2]),
                ]
            )
            .reshape(-1, 1)
            .T
        )
        results = []
        for rf_ in time_estimators:
            results.append(rf_.predict(in_features)[0])
        result = np.mean(results), np.std(results)
    else:
        result = 0, 0
    return result
