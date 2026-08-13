import tifffile
import numpy as np

# ----------------------------
# CHANGE THIS PATH IF NEEDED
# ----------------------------
tiff_path = "Data_Larynx/20260129 Larynx Section B_037.tiff"

print("=" * 60)
print("Opening TIFF...")
print("=" * 60)

with tifffile.TiffFile(tiff_path) as tif:

    print("\nNumber of series:")
    print(len(tif.series))

    print("\nNumber of pages:")
    print(len(tif.pages))

    print("\nSeries information:")

    for i, series in enumerate(tif.series):
        print(f"\nSeries {i}")
        print("Shape :", series.shape)
        print("Axes  :", series.axes)
        print("Dtype :", series.dtype)

    print("\n" + "=" * 60)
    print("Metadata")
    print("=" * 60)

    try:
        print(tif.imagej_metadata)
    except Exception:
        print("No ImageJ metadata")

    print("\n" + "=" * 60)
    print("OME Metadata")
    print("=" * 60)

    try:
        print(tif.ome_metadata)
    except Exception:
        print("No OME metadata")

print("\nDone.")