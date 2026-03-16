# Zenith Star Trail Latitude Finder

Determine your geographic latitude purely from a long-exposure star trail photograph taken with the camera pointed straight up (toward the zenith). No GPS, no horizon reference, no star catalogue required — just geometry.

---

## How It Works

When a camera is aimed at the zenith, the image centre corresponds to the zenith point on the celestial sphere. Star trails appear as circular arcs, all concentric around the celestial pole. The pole is offset from the zenith by an angular distance of `(90° − latitude)`.

The algorithm:

1. Enhances the image (CLAHE + bilateral filter + unsharp mask)
2. Detects edges (Canny + Sobel, combined)
3. Extracts arc contours above a minimum length threshold
4. Fits circles to each arc via linear least squares, refined with nonlinear optimisation
5. Finds the common centre of all fitted circles → the celestial pole pixel
6. Measures the pixel distance from the pole to the image centre (= zenith)
7. Converts to degrees using the plate scale: `latitude = 90° − angular_distance`

---

## Requirements

```
python >= 3.8
opencv-python
numpy
scipy
matplotlib
```

Install dependencies:

```bash
pip install opencv-python numpy scipy matplotlib
```

---

## Usage

### Jupyter Notebook (recommended)

Open `Zenith_star_trails.ipynb` and edit the settings cell at the top, then run all cells. Plots render inline.

### Command line

1. Edit the settings block at the top of the script:

```python
IMAGE_PATH      = "Zenith_star_trails.jpg"   # path to your image
FOCAL_LENGTH_MM = 10                          # lens focal length in mm
PIXEL_SIZE_UM   = 5.50                        # sensor pixel pitch in µm
HEMISPHERE      = 'north'                     # 'north' or 'south'
ACTUAL_LATITUDE = 48.0                        # known latitude for comparison (optional)
```


2. The script prints results to the console and opens a three-panel diagnostic plot:
   - Original image
   - Edge map
   - Fitted circles overlaid, with zenith (cyan), pole (red), and the connecting line (orange)

### Example output

```
Image:        1232 x 922 px
Plate scale:  113.25 arcsec/px
Zenith pixel: (616, 461)

Pole at (621.3, 94.7) px  +/-18.4 px
Zenith-to-pole : 366.5 px  =  11.523 deg  (→  latitude ≈ 78.477°)  [example only]

========== RESULTS ==========
Zenith pixel      : (616, 461)
Pole pixel        : (621.3, 94.7)
Zenith-to-pole    : 366.5 px  =  11.523 deg
Calculated Latitude : 78.477 deg
Actual Latitude     : 48.000 deg
Difference          : 0.477 deg
Uncertainty       : +/- 0.578 deg  (~+/- 64 km)
Zone              : Temperate
Arcs used         : 47
==============================
```

---

## Camera Setup

| Parameter | Requirement |
|-----------|-------------|
| Pointing  | Camera aimed **straight up** — image centre must equal the zenith |
| Levelling | Use a hot-shoe spirit level or tripod bubble |
| Exposure  | Long enough to produce arcs ≥ 150 px; typically 30 min – several hours |
| Lens      | Wide-angle recommended so the pole falls within the frame at mid-latitudes |
| Sky       | Dark site preferred; light pollution reduces trail contrast |

> **Critical:** Any tilt of the camera from vertical introduces a direct error in the zenith position and therefore in the computed latitude. 1° of tilt ≈ 1° of latitude error.

---

## Accuracy & Limitations

| Source | Typical effect |
|--------|---------------|
| Camera tilt from vertical | ~1° per degree of tilt — dominant error |
| Focal length / pixel size uncertainty | Scales linearly; 1% error → ~0.4° latitude error |
| Short or clipped arcs | Arcs cut by the image boundary constrain the circle centre poorly |
| Few arcs | Minimum 3 valid fits required; more arcs → lower pole uncertainty |

Realistic accuracy on a well-levelled real photograph: **±0.5° – 2°** depending on exposure length and sky quality.

---

## Tested Hardware

| Camera | Pixel size |
|--------|-----------|
| Nikon D90 | 5.50 µm |

Common pixel sizes for other cameras:

| Camera | Pixel size (µm) |
|--------|----------------|
| Canon 6D | 6.54 |
| Sony A7 III | 5.97 |
| Nikon D3500 | 5.87 |
| Canon 90D | 3.20 |

---

## File Structure

```
.
├── Zenith_star_trails.ipynb   # Jupyter notebook (recommended entry point)
├── Zenith_star_trails.jpg     # Sample image (Nikon D90, 10mm, ~48°N)
└── README.md
```

The included sample image is a long-exposure zenith shot suitable for testing the pipeline out of the box. Open `Zenith_star_trails.ipynb` to run interactively with inline plots, or run `zenith_latitude.py` directly from the command line.

---

## License

MIT
