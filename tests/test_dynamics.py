import numpy as np
import numpy.typing as npt
import pytest

import diffmp


@pytest.mark.parametrize(
    "dynamics", [("unicycle1_v0"), ("unicycle1_v1"), ("unicycle2_v0")]
)
def test_implementations(dynamics: str):
    assert isinstance(
        diffmp.dynamics.get_dynamics(dynamics), diffmp.dynamics.DynamicsBase
    )


@pytest.mark.parametrize(
    "dynamics, action, expected",
    [
        ("unicycle1_v0", np.array([[-0.5, 0.0]]), np.array([[-0.05, 0.0, 0.0]])),
        ("unicycle1_v1", np.array([[0.5, 0.0]]), np.array([[0.05, 0.0, 0.0]])),
        (
            "unicycle2_v0",
            np.array([[0.25, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 0.025, 0.0]]),
        ),
    ],
)
def test_step(
    dynamics: str, action: npt.NDArray[np.floating], expected: npt.NDArray[np.floating]
):
    dyn = diffmp.dynamics.get_dynamics(dynamics)
    q0 = np.zeros(len(dyn.q))
    q1 = dyn.step(q0, action)
    assert np.allclose(q1, np.atleast_2d(expected))
