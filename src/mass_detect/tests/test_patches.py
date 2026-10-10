"""Unit tests (run with: pytest tests)."""
import SimpleITK as sitk
import numpy as np
from src.config import AIR_HU, CENTRE_INDEX, PATCH_SIZE
from src.patches import extract_patch

BACKGROUND_HU = -1000
BLOB_SIGMA_MM = 4.0

def rotation(angle_x=0.0, angle_y=0.0, angle_z=0.0): # Rotation matrix from three angles in degrees.
    ax, ay, az = np.radians([angle_x, angle_y, angle_z])
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx

def make_ct(blob_mm, spacing=(0.7, 0.7, 2.5), size=(200, 200, 80), direction=None, volume_centre_mm=None): # A CT with a Gaussian blob of high HU at the physical point blob_mm.
    # The grid is placed so that the middle of the volume is at volume_centre_mm (default: the blob).
    # The array is filled by computing the physical position of every voxel, so the geometry is exact.
    
    direction = np.eye(3) if direction is None else direction
    spacing = np.array(spacing, dtype=float)
    size_xyz = np.array(size)
    middle = np.asarray(blob_mm if volume_centre_mm is None else volume_centre_mm, dtype=float)
    origin = middle - direction @ (spacing * (size_xyz - 1) / 2)       # index (size-1)/2 sits at 'middle'

    k, j, i = np.meshgrid(np.arange(size[2]), np.arange(size[1]), np.arange(size[0]), indexing="ij")
    index = np.stack([i, j, k], axis=-1) * spacing                      # shape (z, y, x, 3)
    physical = origin + index @ direction.T
    distance_squared = ((physical - np.asarray(blob_mm)) ** 2).sum(axis=-1)
    values = BACKGROUND_HU + 1500 * np.exp(-distance_squared / (2 * BLOB_SIGMA_MM ** 2))

    image = sitk.GetImageFromArray(values.astype(np.int16))
    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(origin))
    image.SetDirection(tuple(direction.ravel()))
    return image

def peak_index(patch): # Voxel (z, y, x) of the brightest voxel.
    return np.array(np.unravel_index(np.argmax(patch), patch.shape))

def test_blob_at_the_centre_lands_on_voxel_32_with_anisotropic_spacing():
    blob = np.array([20.3, 100.7, 200.2])
    image = make_ct(blob, spacing=(0.7, 0.7, 2.5), volume_centre_mm=blob + [3, -4, 6])
    patch, outside = extract_patch(image, blob)
    assert patch.shape == (PATCH_SIZE,) * 3 and patch.dtype == np.int16
    assert np.all(np.abs(peak_index(patch) - CENTRE_INDEX) <= 1)
    assert outside == 0.0

def test_axis_order_is_z_y_x(): # A blob displaced by (+5, -3, +7) mm in (x, y, z) must appear at index (32+7, 32-3, 32+5).
    centre = np.array([20.0, 100.0, 200.0])
    image = make_ct(centre + [5, -3, 7], spacing=(0.7, 0.7, 2.5), volume_centre_mm=centre)
    patch, _ = extract_patch(image, centre)
    expected = np.array([CENTRE_INDEX + 7, CENTRE_INDEX - 3, CENTRE_INDEX + 5])
    assert np.all(np.abs(peak_index(patch) - expected) <= 1), peak_index(patch)

def test_tilted_direction_is_handled(): # CTs with gantry tilt have a direction that is not the identity. The patch must still be aligned with the world.
    centre = np.array([-30.0, 60.0, -500.0])
    for tilt in (rotation(angle_x=13), rotation(angle_y=-8, angle_z=5), rotation(angle_z=30)):
        offset = np.array([4, -6, 5])                                    # displacement in world axes
        image = make_ct(centre + offset, spacing=(0.65, 0.65, 1.0), size=(220, 220, 200),
                        direction=tilt, volume_centre_mm=centre)
        patch, outside = extract_patch(image, centre)
        expected = np.array([CENTRE_INDEX + offset[2], CENTRE_INDEX + offset[1], CENTRE_INDEX + offset[0]])
        assert np.all(np.abs(peak_index(patch) - expected) <= 1), (peak_index(patch), expected)
        assert outside == 0.0

def test_naive_conversion_would_fail_on_a_tilted_ct(): # Documents why the direction matters: ignoring it puts the blob tens of mm away.
    centre = np.array([-30.0, 60.0, -500.0])
    tilt = rotation(angle_x=13)
    image = make_ct(centre, spacing=(0.65, 0.65, 1.0), size=(220, 220, 200), direction=tilt,
                    volume_centre_mm=centre + [60, 60, 0])
    correct = np.array(image.TransformPhysicalPointToContinuousIndex(tuple(centre)))
    naive = (centre - np.array(image.GetOrigin())) / np.array(image.GetSpacing())
    assert np.linalg.norm((correct - naive) * np.array(image.GetSpacing())) > 10

def test_points_outside_the_ct_are_filled_with_air_and_counted():
    blob = np.array([0.0, 0.0, 0.0])
    image = make_ct(blob, spacing=(1.0, 1.0, 1.0), size=(100, 100, 100))      # extends about 50 mm around the blob
    inside, outside_inside = extract_patch(image, blob)
    near_border, outside_border = extract_patch(image, blob + [40, 0, 0])      # patch sticks out by about 22 mm in x
    assert outside_inside == 0.0
    assert 0.2 < outside_border < 0.45
    assert near_border.min() == AIR_HU
    assert inside.min() >= BACKGROUND_HU - 1

def test_nearest_neighbour_keeps_integer_labels():
    centre = np.array([10.0, 20.0, 30.0])
    labels = make_ct(centre, spacing=(0.7, 0.7, 2.5), size=(200, 200, 80))
    labels = sitk.Cast(labels > 0, sitk.sitkUInt8) * 7                    # every voxel above 0 HU becomes label 7
    patch, _ = extract_patch(labels, centre, interpolator=sitk.sitkNearestNeighbor, outside_value=0)
    assert set(np.unique(patch)) <= {0, 7}
