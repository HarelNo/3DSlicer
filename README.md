# 3DSlicer
The problem:
  NRRD files can be easily read to generate images of slices parallel to each axis but generating an image from a sloped plane is a bit harder.
Idea:
  Using 3 coordinates a plane could be generated and used to "slice" the "3d shape" provided in the NRRD file and generate an image based on overlap in coordinates.
Problem with the idea:
  Generating a 2d plane and calculating the overlap takes alot of effort and slows down the frame generation speed immensly.
Final steps:
  Using the algorithm given by the task an isosceles triangle or an isosceles trapezoid is generated and the image "trapped" within it is then saved and potentially saved as a video



Reads a 3D NRRD volume and a coordinate file, projects each shape, and
writes one 2D grayscale PNG per coordinate series.

Install the dependencies and run the program:

```powershell
python -m pip install -r requirements.txt
python main.py
```

When run, the script prompts you to select a 3D NRRD volume and then a
coordinate text file. You can then choose to save separate PNGs or combine the
projections into an MP4. For MP4 output, choose how many seconds each image is
displayed and where to save the video. Differently sized projections are
centered on a common black canvas in the video. PNGs are saved as
`projection_001.png`, `projection_002.png`, and so on in a `projections`
folder beside `main.py`. Both formats use 8-bit grayscale, where 0 is black
and 255 is white. The MP4 is finalized for early playback and uses frames
without B-frame reordering to improve compatibility with players and editors.
Canceling a dialog stops the run.

`Coordinates.txt` can contain any number of coordinate series. Each series
consists of either three or four rows, one `(x, y, z)` voxel coordinate per
row, with blank lines between series:

- Three points define an isosceles triangle. The first point is the apex and
  is equally distant from the other two.
- Four points are listed around the perimeter as `A, B, C, D`, defining an
  isosceles trapezoid. `AB` and `CD` are parallel, and legs `AD` and `BC` are
  equal in length.

Each image contains only its triangle or trapezoid, with areas outside the
shape or the volume black. For example:

```text
128, 40, 32
68, 160, 32
188, 160, 32

128, 48, 64
48, 208, 64
208, 208, 64

4, 4, 32
12, 4, 32
10, 10, 32
6, 10, 32
```

`load_nrrd_grayscale(path)` accepts a 3D scalar NRRD and scales its complete
intensity range linearly to integers from `0` (black) to `255` (white). The
triangle projection uses `(x, y, z)` voxel coordinates and trilinear
interpolation to sample fractional positions between voxels; it rounds the
interpolated value to the nearest 8-bit grayscale level. Samples outside the
volume are set to `0`.
