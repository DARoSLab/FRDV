"""Check motion chunk boundaries, overlap blending, and quaternion conventions."""

import numpy as np
import pytest

from mocap_retargeting.kinematic_retargeting.motion_chunking.chunk_retargeting import (
    plan_chunks,
)
from mocap_retargeting.kinematic_retargeting.motion_chunking.trajectory_stitching import (
    combine_chunks,
)
from mocap_retargeting.kinematic_retargeting.motion_chunking.trajectory_stitching import (
    blend_trajectories,
)


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("stitch", [False, True])
@pytest.mark.parametrize(
    "frames,overlap", [(97, 5), (12, 15), (5, 20), (1, 20), (12, 0)]
)
def test_overlapping_chunks_reconstruct_every_frame_in_order(
    workers, stitch, frames, overlap
):
    positions = np.arange(frames * 6, dtype=float).reshape(frames, 2, 3)
    chunks, size = plan_chunks(
        positions, workers, overlap, 11, "motion.bvh", [("a", "b")], ["a", "b"], fps=60
    )
    results = []
    for chunk_id, data, first_frame, path, skeleton, names, fps in chunks:
        assert fps == 60
        assert np.array_equal(data[0], positions[first_frame - 11])
        q = np.zeros((len(data), 10))
        q[:, :3] = data[:, 0]
        q[:, 6] = 1.0
        q[:, 7:] = data[:, 1]
        results.append((chunk_id, q, data))
    q, mocap = combine_chunks(list(reversed(results)), size, frames, overlap, stitch)
    q = np.asarray(q)
    np.testing.assert_array_equal(mocap, positions)
    np.testing.assert_allclose(q[:, :3], positions[:, 0], atol=1e-12)
    np.testing.assert_allclose(q[:, 7:], positions[:, 1], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(q[:, 3:7], axis=1), 1.0)


@pytest.mark.parametrize("overlap", [2, 6, 15])
def test_earlier_worker_skips_covered_sections_and_blends_directly_into_next_contributor(
    overlap,
):
    positions = np.zeros((12, 1, 3))
    positions[:, 0, 0] = np.arange(12)
    chunks, size = plan_chunks(
        positions, 3, overlap, 0, "motion.npz", [], ["Pelvis"], fps=120
    )
    results = []
    for chunk_id, mocap, *_ in chunks:
        q = np.zeros((len(mocap), 8))
        q[:, 0] = mocap[:, 0, 0]
        q[:, 6] = 1
        q[:, 7] = [10, 100, 20][chunk_id]
        results.append((chunk_id, q, mocap))
    q, mocap = combine_chunks(results, size, 12, overlap, True)
    q = np.asarray(q)
    np.testing.assert_array_equal(np.asarray(mocap), positions)
    np.testing.assert_array_equal(q[:, 0], np.arange(12))
    if overlap == 2:
        # Normal-size overlaps retain the existing A/B and B/C transitions.
        expected = [10] * 5 + [100] * 4 + [20] * 3
    elif overlap == 6:
        # A owns frames 1-8, A/C blend on 9-10, C owns 11-12. B is unused.
        expected = [10] * 9 + [20] * 3
    else:
        # A already covers the whole motion, so no handoff is needed.
        expected = [10] * 12
    np.testing.assert_allclose(q[:, 7], expected)


@pytest.mark.parametrize("stitch", [False, True])
@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "empty", "truncated", "nonfinite"]
)
def test_combining_rejects_missing_or_invalid_results(stitch, failure):
    positions = np.zeros((12, 1, 3))
    chunks, size = plan_chunks(
        positions, 3, 2, 0, "motion.npz", [], ["Pelvis"], fps=120
    )
    results = []
    for chunk_id, mocap, *_ in chunks:
        q = np.zeros((len(mocap), 8))
        q[:, 6] = 1
        results.append((chunk_id, q, mocap))
    if failure == "missing":
        results.pop(1)
    elif failure == "duplicate":
        results.append(results[1])
    elif failure == "empty":
        results[1] = (1, [], [])
    elif failure == "truncated":
        results[1] = (1, results[1][1][:-1], results[1][2][:-1])
    else:
        results[1][1][0, 7] = np.nan
    with pytest.raises(ValueError, match="chunk IDs|Chunk 1"):
        combine_chunks(results, size, 12, 2, stitch)


def test_blending_antipodal_quaternions_preserves_the_rotation():
    left = np.zeros((9, 10))
    right = np.ones((9, 10))
    left[:, 3:7] = [0, 0, 0, 1]
    right[:, 3:7] = [0, 0, 0, -1]
    blended = blend_trajectories(left, right)
    np.testing.assert_allclose(blended[:, 3:7], np.tile([0, 0, 0, 1], (9, 1)))
    np.testing.assert_allclose(blended[0, :3], left[0, :3])
    np.testing.assert_allclose(blended[-1, :3], right[-1, :3])
