import numpy as np

from cps_maze.calibration.charuco import charuco_ids_to_maze_points_mm


def test_charuco_ids_map_into_maze_millimeters():
    ids = np.array([[0], [3], [12], [15]])

    maze_points = charuco_ids_to_maze_points_mm(ids)

    assert np.allclose(
        maze_points,
        np.array(
            [
                [-41.0, 155.0],
                [-5.0, 155.0],
                [-41.0, 119.0],
                [-5.0, 119.0],
            ]
        ),
    )