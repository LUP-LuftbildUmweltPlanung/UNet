import glob
from pathlib import Path
import numpy as np
from osgeo import gdal
from tqdm import tqdm


def merge_label_tiles_windowed(
    tiles_folder,
    output_file,
    window_px=4096,
    recursive=False,
    show_progress=True,
):
    """
    Merge single-band class-label tiles (Byte) with overlap.
    First tile wins in overlap areas.
    """

    # -------------------------------------------------------
    # Resolve tiles
    # -------------------------------------------------------
    p = Path(tiles_folder)
    if recursive:
        tiles = sorted(p.rglob("*.tif"))
    else:
        tiles = sorted(p.glob("*.tif"))

    tiles = [str(t) for t in tiles if Path(t).is_file()]
    if not tiles:
        raise FileNotFoundError("No tiles found.")

    # -------------------------------------------------------
    # Read reference info from first tile
    # -------------------------------------------------------
    ds0 = gdal.Open(tiles[0], gdal.GA_ReadOnly)
    gt0 = ds0.GetGeoTransform()
    proj0 = ds0.GetProjection()
    xres, yres = gt0[1], gt0[5]
    ds0 = None

    # -------------------------------------------------------
    # Determine full extent
    # -------------------------------------------------------
    ulx_list, uly_list, lrx_list, lry_list = [], [], [], []

    for t in tiles:
        ds = gdal.Open(t)
        gt = ds.GetGeoTransform()
        ulx = gt[0]
        uly = gt[3]
        lrx = gt[0] + ds.RasterXSize * gt[1]
        lry = gt[3] + ds.RasterYSize * gt[5]

        ulx_list.append(ulx)
        uly_list.append(uly)
        lrx_list.append(lrx)
        lry_list.append(lry)
        ds = None

    full_ulx = min(ulx_list)
    full_uly = max(uly_list)
    full_lrx = max(lrx_list)
    full_lry = min(lry_list)

    xsize = int(round((full_lrx - full_ulx) / xres))
    ysize = int(round((full_lry - full_uly) / yres))

    # -------------------------------------------------------
    # Create output raster
    # -------------------------------------------------------
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(
        str(output_file),
        xsize,
        ysize,
        1,
        gdal.GDT_Byte,
        options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )

    dst.SetProjection(proj0)
    dst.SetGeoTransform((full_ulx, xres, 0.0, full_uly, 0.0, yres))

    # -------------------------------------------------------
    # Build spatial index
    # -------------------------------------------------------
    index = []
    for t in tiles:
        ds = gdal.Open(t)
        gt = ds.GetGeoTransform()

        index.append({
            "path": t,
            "ulx": gt[0],
            "uly": gt[3],
            "lrx": gt[0] + ds.RasterXSize * gt[1],
            "lry": gt[3] + ds.RasterYSize * gt[5],
            "w": ds.RasterXSize,
            "h": ds.RasterYSize
        })
        ds = None

    # -------------------------------------------------------
    # Windowed merging
    # -------------------------------------------------------
    win = int(window_px)
    n_rows = (ysize + win - 1) // win
    n_cols = (xsize + win - 1) // win

    iterator = range(n_rows * n_cols)
    if show_progress:
        iterator = tqdm(iterator, desc="Merging", unit="window")

    for k in iterator:
        row = k // n_cols
        col = k % n_cols

        y0 = row * win
        x0 = col * win

        h = min(win, ysize - y0)
        w = min(win, xsize - x0)

        win_ulx = full_ulx + x0 * xres
        win_uly = full_uly + y0 * yres

        # buffer for this window
        out_buf = np.zeros((h, w), dtype=np.uint8)
        written = np.zeros((h, w), dtype=bool)

        for t in index:

            win_lrx = win_ulx + w * xres
            win_lry = win_uly + h * yres

            ov_ulx = max(win_ulx, t["ulx"])
            ov_uly = min(win_uly, t["uly"])
            ov_lrx = min(win_lrx, t["lrx"])
            ov_lry = max(win_lry, t["lry"])

            if not (ov_ulx < ov_lrx and ov_lry < ov_uly):
                continue

            nx = int(round((ov_lrx - ov_ulx) / xres))
            ny = int(round((ov_uly - ov_lry) / abs(yres)))
            if nx <= 0 or ny <= 0:
                continue

            win_x_off = int(round((ov_ulx - win_ulx) / xres))
            win_y_off = int(round((win_uly - ov_uly) / abs(yres)))
            tile_x_off = int(round((ov_ulx - t["ulx"]) / xres))
            tile_y_off = int(round((t["uly"] - ov_uly) / abs(yres)))

            ds = gdal.Open(t["path"])
            arr = ds.GetRasterBand(1).ReadAsArray(tile_x_off, tile_y_off, nx, ny)
            ds = None

            target = out_buf[win_y_off:win_y_off+ny, win_x_off:win_x_off+nx]
            mask   = written[win_y_off:win_y_off+ny, win_x_off:win_x_off+nx]

            write_pixels = ~mask
            target[write_pixels] = arr[write_pixels]
            mask[write_pixels] = True

        dst.GetRasterBand(1).WriteArray(out_buf, xoff=x0, yoff=y0)

    dst.FlushCache()
    dst = None

    print(f"✅ Merge finished: {output_file}")


merge_label_tiles_windowed(
    r"Path\to\tiles",
    r"output\merged_tiles.tif",
    window_px=1024
)
