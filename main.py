
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


def project_coordinates(matrix, points):
    """Project a triangle or isosceles trapezoid from its defining points."""
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
    if volume.dtype != np.uint8:
        invalid = np.argwhere((volume < 0) | (volume > 255))
        if invalid.size:
            z, y, x = (int(index) for index in invalid[0])
            raise ValueError(
                "matrix values must be integers from 0 to 255; "
                f"invalid value at matrix[{z}][{y}][{x}]"
            )

    if len(points) not in (3, 4):
        raise ValueError("each coordinate series must contain 3 or 4 points")

    parsed_points = []
    for label, point in zip(("a", "b", "c", "d"), points):
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
        parsed_points.append(coordinates)

    if len(parsed_points) == 3:
        a, b, c = parsed_points
        ab_vector = b - a
        ac_vector = c - a
        ab = np.linalg.norm(ab_vector)
        ac = np.linalg.norm(ac_vector)
        bc = np.linalg.norm(c - b)
        tolerance = 1e-6 * max(1.0, ab, ac)
        if ab == 0 or ac == 0 or bc == 0:
            raise ValueError("triangle points must be distinct")
        if abs(ab - ac) > tolerance:
            raise ValueError("triangle must be isosceles with the first point as apex")
        if np.linalg.norm(np.cross(ab_vector, ac_vector)) <= tolerance:
            raise ValueError("triangle points must define a non-degenerate plane")
        row_vector = ab_vector
        opposite_row_vector = ac_vector
        row_extent = ab
        column_extent = bc
    else:
        a, b, c, d = parsed_points
        ab_vector = b - a
        dc_vector = c - d
        ad_vector = d - a
        bc_vector = c - b
        ab = np.linalg.norm(ab_vector)
        dc = np.linalg.norm(dc_vector)
        ad = np.linalg.norm(ad_vector)
        bc = np.linalg.norm(bc_vector)
        tolerance = 1e-6 * max(1.0, ab, dc, ad, bc)
        if min(ab, dc, ad, bc) == 0:
            raise ValueError("trapezoid points must be distinct")
        if np.linalg.norm(np.cross(ab_vector, dc_vector)) > tolerance * max(ab, dc):
            raise ValueError("trapezoid sides AB and CD must be parallel")
        plane_normal = np.cross(ab_vector, ad_vector)
        normal_length = np.linalg.norm(plane_normal)
        if normal_length <= tolerance * max(ab, ad):
            raise ValueError("trapezoid points must define a non-degenerate plane")
        if abs(np.dot(c - a, plane_normal / normal_length)) > tolerance:
            raise ValueError("trapezoid points must lie in the same plane")
        if abs(ad - bc) > tolerance:
            raise ValueError("trapezoid legs AD and BC must have equal lengths")
        row_start = a
        row_start_vector = ad_vector
        opposite_start = b
        opposite_start_vector = bc_vector
        row_extent = max(ad, bc)
        column_extent = max(ab, dc)

    if len(parsed_points) == 3:
        row_start = a
        row_start_vector = ab_vector
        opposite_start = a
        opposite_start_vector = ac_vector

    rows = ceil(row_extent) + 1
    columns = ceil(column_extent) + 1
    result = [[0 for _ in range(columns)] for _ in range(rows)]

    for y in range(rows):
        fraction = y / (rows - 1)
        start = row_start + fraction * row_start_vector
        end = opposite_start + fraction * opposite_start_vector
        span = end - start
        row_length = np.linalg.norm(span)
        x_offset = ((columns - 1) - row_length) / 2

        for x in range(columns):
            if row_length == 0:
                if x != floor((columns - 1) / 2 + 0.5):
                    continue
                along_row = 0
            else:
                along_row = (x - x_offset) / row_length
                if not 0 <= along_row <= 1:
                    continue
            point = start + along_row * span
            result[y][x] = _sample_trilinear(matrix, point)

    return result


def write_projection_png(image, path):
    """Write a row-major grayscale image as an 8-bit PNG."""
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path, format="PNG")


def load_coordinate_series(path):
    """Read groups of three or four comma-separated xyz points."""
    series = []
    current_points = []
    current_start_line = None

    with Path(path).open(encoding="utf-8") as coordinate_file:
        for line_number, line in enumerate(coordinate_file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                if current_points:
                    if len(current_points) not in (3, 4):
                        raise ValueError(
                            f"{path}: lines {current_start_line}-{line_number - 1} "
                            f"must contain 3 or 4 coordinate rows; "
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
            if len(current_points) > 4:
                raise ValueError(
                    f"{path}: coordinate series starting on line "
                    f"{current_start_line} may contain at most 4 rows"
                )

    if current_points:
        if len(current_points) not in (3, 4):
            raise ValueError(
                f"{path}: lines {current_start_line}-end must contain exactly "
                f"3 or 4 coordinate rows; found {len(current_points)}"
            )
        series.append(tuple(current_points))

    if not series:
        raise ValueError(f"{path}: no coordinate series found")
    return series


def select_input_files():
    """Prompt for source files and the desired projection output."""
    from tkinter import Tk, filedialog, messagebox, simpledialog

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

        make_video = messagebox.askyesno(
            "Projection output",
            "Create an MP4 video instead of individual PNG images?",
            parent=root,
        )
        if make_video:
            frame_duration = simpledialog.askfloat(
                "Video frame duration",
                "How long should each image be displayed? (seconds)",
                initialvalue=1.0,
                minvalue=0.01,
                parent=root,
            )
            if frame_duration is None:
                return None

            output_path = filedialog.asksaveasfilename(
                parent=root,
                title="Save projection video",
                defaultextension=".mp4",
                filetypes=(("MP4 video", "*.mp4"),),
            )
            if not output_path:
                return None
            return (
                Path(nrrd_path),
                Path(coordinates_path),
                "mp4",
                frame_duration,
                Path(output_path),
            )

        return Path(nrrd_path), Path(coordinates_path), "png", None, None
    finally:
        root.destroy()


def write_projection_video(images, path, frame_duration):
    """Write grayscale projection images to an MP4 with a fixed frame duration."""
    import imageio_ffmpeg

    if not images:
        raise ValueError("cannot create a video without projection images")
    if not np.isfinite(frame_duration) or frame_duration <= 0:
        raise ValueError("frame duration must be a finite positive number")

    frames = [np.asarray(image, dtype=np.uint8) for image in images]
    height = max(frame.shape[0] for frame in frames)
    width = max(frame.shape[1] for frame in frames)
    height += height % 2
    width += width % 2
    fps = 1.0 / frame_duration
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart", "-bf", "0"],
    )
    writer.send(None)
    try:
        for frame in frames:
            canvas = np.zeros((height, width), dtype=np.uint8)
            y_offset = (height - frame.shape[0]) // 2
            x_offset = (width - frame.shape[1]) // 2
            canvas[
                y_offset : y_offset + frame.shape[0],
                x_offset : x_offset + frame.shape[1],
            ] = frame
            writer.send(np.repeat(canvas[:, :, np.newaxis], 3, axis=2).tobytes())
    finally:
        writer.close()


def main():
    selected_files = select_input_files()
    if selected_files is None:
        print("Selection cancelled; no projections were generated.")
        return

    input_path, coordinates_path, output_format, frame_duration, video_path = (
        selected_files
    )
    matrix = load_nrrd_grayscale(input_path)
    coordinate_series = load_coordinate_series(coordinates_path)

    images = []
    for points in coordinate_series:
        image = project_coordinates(matrix, points)
        images.append(np.asarray(image, dtype=np.uint8))

    if output_format == "mp4":
        write_projection_video(images, video_path, frame_duration)
        print(f"Saved {len(images)} projections to video {video_path}")
    else:
        output_dir = Path(__file__).with_name("projections")
        output_dir.mkdir(exist_ok=True)
        for index, image in enumerate(images, start=1):
            output_path = output_dir / f"projection_{index:03d}.png"
            write_projection_png(image, output_path)
            print(f"Saved projection {index} to {output_path}")


if __name__ == "__main__":
    main()