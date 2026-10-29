
from math import ceil, floor
from pathlib import Path

import nrrd
import numpy as np
from PIL import Image


def load_nrrd_grayscale(path):
    """Load a 3D NRRD volume and scale its intensities to 8-bit grayscale."""
    data, header = nrrd.read(path, index_order="F")
    if header.get("dimension") != 3 or data.ndim != 3:
        raise ValueError("input NRRD must contain a 3D scalar volume")
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(
        data.dtype, np.complexfloating
    ):
        raise ValueError("input NRRD must contain numeric scalar values")
    if not np.isfinite(data).all():
        raise ValueError("input NRRD must contain only finite values")

    # Convert NRRD's x, y, z axes to the projection's z, y, x matrix layout.
    volume = np.transpose(data, (2, 1, 0)).astype(np.float64)
    minimum = volume.min()
    maximum = volume.max()
    if minimum == maximum:
        return np.zeros(volume.shape, dtype=np.uint8)

    grayscale = np.rint((volume - minimum) * (255.0 / (maximum - minimum)))
    return grayscale.astype(np.uint8)


def _sample_trilinear(matrix, point):
    x, y, z = point
    depth = len(matrix)
    height = len(matrix[0])
    width = len(matrix[0][0])
    bounds_tolerance = 1e-8
    if not (
        -bounds_tolerance <= z <= depth - 1 + bounds_tolerance
        and -bounds_tolerance <= y <= height - 1 + bounds_tolerance
        and -bounds_tolerance <= x <= width - 1 + bounds_tolerance
    ):
        return 0
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    z = min(max(z, 0), depth - 1)

    x0, y0, z0 = floor(x), floor(y), floor(z)
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    z1 = min(z0 + 1, depth - 1)
    dx, dy, dz = x - x0, y - y0, z - z0

    value = 0.0
    for zi, z_weight in ((z0, 1 - dz), (z1, dz)):
        for yi, y_weight in ((y0, 1 - dy), (y1, dy)):
            for xi, x_weight in ((x0, 1 - dx), (x1, dx)):
                value += matrix[zi][yi][xi] * x_weight * y_weight * z_weight
    return floor(value + 0.5)


def project_plane(matrix, a, b, c):
    """Project the part of the plane through three points inside the volume."""
    try:
        volume = np.asarray(matrix)
    except ValueError as error:
        raise ValueError("matrix must be a rectangular 3D volume") from error
    if volume.ndim != 3 or not all(volume.shape):
        raise ValueError("matrix must be a non-empty 3D list")
    if not np.issubdtype(volume.dtype, np.integer) or np.issubdtype(
        volume.dtype, np.bool_
    ):
        raise ValueError("matrix values must be integers from 0 to 255")
    invalid = np.argwhere((volume < 0) | (volume > 255))
    if invalid.size:
        z, y, x = (int(index) for index in invalid[0])
        raise ValueError(
            "matrix values must be integers from 0 to 255; "
            f"invalid value at matrix[{z}][{y}][{x}]"
        )

    points = []
    for label, point in zip(("a", "b", "c"), (a, b, c)):
        try:
            coordinates = np.asarray(point, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label} must contain three numeric coordinates"
            ) from error
        if coordinates.shape != (3,) or not np.isfinite(coordinates).all():
            raise ValueError(
                f"{label} must contain three finite coordinates (x, y, z)"
            )
        points.append(coordinates)
    a, b, c = points

    ab_vector = b - a
    ac_vector = c - a
    normal = np.cross(ab_vector, ac_vector)
    normal_length = np.linalg.norm(normal)
    if normal_length <= 1e-12 * max(
        1.0, np.linalg.norm(ab_vector) * np.linalg.norm(ac_vector)
    ):
        raise ValueError("a, b, and c must define a non-degenerate plane")
    normal /= normal_length
    horizontal = ab_vector / np.linalg.norm(ab_vector)
    vertical = np.cross(normal, horizontal)

    depth, height, width = volume.shape
    corners = [
        np.asarray((x, y, z), dtype=np.float64)
        for x in (0, width - 1)
        for y in (0, height - 1)
        for z in (0, depth - 1)
    ]
    intersection_points = []
    distance_tolerance = 1e-8 * max(width, height, depth)
    for first_index, first in enumerate(corners):
        for second_index in range(first_index + 1, len(corners)):
            second = corners[second_index]
            if sum(first[axis] != second[axis] for axis in range(3)) != 1:
                continue

            first_distance = np.dot(first - a, normal)
            second_distance = np.dot(second - a, normal)
            candidates = []
            if abs(first_distance) <= distance_tolerance:
                candidates.append(first)
            if abs(second_distance) <= distance_tolerance:
                candidates.append(second)
            if first_distance * second_distance < 0:
                fraction = first_distance / (first_distance - second_distance)
                candidates.append(first + fraction * (second - first))

            for candidate in candidates:
                if not any(
                    np.linalg.norm(candidate - existing) <= distance_tolerance
                    for existing in intersection_points
                ):
                    intersection_points.append(candidate)

    if len(intersection_points) < 3:
        raise ValueError("the plane does not intersect the volume in a 2D area")

    plane_coordinates = np.asarray(
        [
            (
                np.dot(point - a, horizontal),
                np.dot(point - a, vertical),
            )
            for point in intersection_points
        ]
    )
    origin = plane_coordinates[0]
    relative_coordinates = plane_coordinates - origin
    has_area = any(
        abs(
            relative_coordinates[i, 0] * relative_coordinates[j, 1]
            - relative_coordinates[i, 1] * relative_coordinates[j, 0]
        )
        > distance_tolerance**2
        for i in range(1, len(relative_coordinates))
        for j in range(i + 1, len(relative_coordinates))
    )
    if not has_area:
        raise ValueError("the plane does not intersect the volume in a 2D area")

    minimum_u, minimum_v = plane_coordinates.min(axis=0)
    maximum_u, maximum_v = plane_coordinates.max(axis=0)
    columns = ceil(maximum_u - minimum_u) + 1
    rows = ceil(maximum_v - minimum_v) + 1
    result = [[0 for _ in range(columns)] for _ in range(rows)]

    for y in range(rows):
        v = maximum_v - y
        for x in range(columns):
            u = minimum_u + x
            point = a + u * horizontal + v * vertical
            result[y][x] = _sample_trilinear(matrix, point)

    return result


def write_projection_png(image, path):
    """Write a row-major grayscale image as an 8-bit PNG."""
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path, format="PNG")


def load_coordinate_series(path):
    """Read blank-line-separated groups of three comma-separated xyz points."""
    series = []
    current_points = []
    current_start_line = None

    with Path(path).open(encoding="utf-8") as coordinate_file:
        for line_number, line in enumerate(coordinate_file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                if current_points:
                    if len(current_points) != 3:
                        raise ValueError(
                            f"{path}: lines {current_start_line}-{line_number - 1} "
                            f"must contain exactly 3 coordinate rows; "
                            f"found {len(current_points)}"
                        )
                    series.append(tuple(current_points))
                    current_points = []
                    current_start_line = None
                continue

            if current_start_line is None:
                current_start_line = line_number
            fields = stripped_line.split(",")
            if len(fields) != 3:
                raise ValueError(
                    f"{path}: line {line_number} must contain x, y, z"
                )
            try:
                point = tuple(float(field.strip()) for field in fields)
            except ValueError as error:
                raise ValueError(
                    f"{path}: line {line_number} contains a non-numeric coordinate"
                ) from error
            if not all(np.isfinite(point)):
                raise ValueError(
                    f"{path}: line {line_number} coordinates must be finite"
                )
            current_points.append(point)

    if current_points:
        if len(current_points) != 3:
            raise ValueError(
                f"{path}: lines {current_start_line}-end must contain exactly "
                f"3 coordinate rows; found {len(current_points)}"
            )
        series.append(tuple(current_points))

    if not series:
        raise ValueError(f"{path}: no coordinate series found")
    return series


def select_input_files():
    """Prompt for the volume NRRD and coordinate text file."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    try:
        nrrd_path = filedialog.askopenfilename(
            parent=root,
            title="Select the 3D NRRD volume",
            filetypes=(("NRRD files", "*.nrrd *.nhdr"), ("All files", "*.*")),
        )
        if not nrrd_path:
            return None

        coordinates_path = filedialog.askopenfilename(
            parent=root,
            title="Select the coordinate series file",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
        )
        if not coordinates_path:
            return None

        return Path(nrrd_path), Path(coordinates_path)
    finally:
        root.destroy()


def main():
    selected_files = select_input_files()
    if selected_files is None:
        print("Input selection cancelled; no projections were generated.")
        return

    input_path, coordinates_path = selected_files
    output_dir = Path(__file__).with_name("projections")
    matrix = load_nrrd_grayscale(input_path)
    coordinate_series = load_coordinate_series(coordinates_path)
    output_dir.mkdir(exist_ok=True)

    for index, (a, b, c) in enumerate(coordinate_series, start=1):
        image = project_plane(matrix, a, b, c)
        output_path = output_dir / f"projection_{index:03d}.png"
        write_projection_png(image, output_path)
        print(f"Saved projection {index} to {output_path}")


if __name__ == "__main__":
    main()