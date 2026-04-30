"""
Boundary extraction and per-contour masks for the LLC tiled grid.

Two functions:

    grid_boundaries_from_mask(grid, mask)
        Given a global C-grid mask, return closed boundary contours as
        (face, j, i) corner indices and (lon, lat) coordinates. Uses
        spherical Delaunay so it handles the dateline, poles, and tile seams.
        Output convention matches the original regionate function: indices
        closed, coordinates open.

    mask_from_grid_boundaries(lons_c, lats_c, grid, ...)
        Drop-in analog of regionate's mask_from_grid_boundaries with the
        same three-branch structure (along_boundary / small region / circumpolar).
        The circumpolar branch uses an antipodal-centroid stereographic
        projection (generalization of the original's south-pole-only
        projection) so it works for either hemisphere.
"""

import numpy as np
import xarray as xr
import matplotlib.tri as mtri
from matplotlib import _tri
from scipy.spatial import ConvexHull

import geopandas as gpd
from shapely.geometry import Polygon
import regionmask


# ---------------------------------------------------------------------------
# Boundary extraction (spherical Delaunay, returns closed contours)
# ---------------------------------------------------------------------------

def grid_boundaries_from_mask(grid, mask):
    """Use spherical Delaunay to get indices of cell corners that bound the mask
    on a tiled (LLC-style) grid. Returns lists with a common length that is
    the number of discrete contours that bound the mask.

    Drop-in analog of `regionate.grid_conform.grid_boundaries_from_mask`,
    extended with a face index for tiled grids. Output convention matches
    the original:
      - face_c_list, i_c_list, j_c_list are CLOSED (last entry == first)
      - lons_c_list, lats_c_list are OPEN (closure stripped)

    ARGUMENTS
    ---------
    grid : `xgcm.Grid` instance with face_connections
    mask : `xr.DataArray` instance of type `bool`, dims (face, j, i)

    RETURNS
    -------
    face_c_list, i_c_list, j_c_list, lons_c_list, lats_c_list
    """
    ds = grid._ds
    x_dim = grid.axes["X"].coords["left"]   # i_g
    y_dim = grid.axes["Y"].coords["left"]   # j_g
    lon_c = ds["geolon_c"]
    lat_c = ds["geolat_c"]
    face_dim = [d for d in lon_c.dims if d not in {y_dim, x_dim}][0]

    # Mask is on C-points (face, j, i); corners are on (face, j_g, i_g).
    # Interpolate mask to corners so it shares dims with geolon_c/geolat_c.
    mask_on_corners = grid.interp(
        grid.interp(mask.astype(float), "X", boundary="extend"),
        "Y", boundary="extend",
    )

    pts = xr.Dataset(
        {"lon_c": lon_c, "lat_c": lat_c, "mask": mask_on_corners}
    ).stack(point=(face_dim, y_dim, x_dim))

    valid = (
        np.isfinite(pts["lon_c"].values).ravel()
        & np.isfinite(pts["lat_c"].values).ravel()
        & np.isfinite(pts["mask"].values).ravel()
    )
    pts = pts.isel(point=xr.DataArray(valid, dims="point"))

    lon = pts["lon_c"].values
    lat = pts["lat_c"].values
    z = pts["mask"].values

    # Spherical Delaunay triangulation of the corner point cloud, via the
    # convex hull of the points projected onto the unit sphere.
    #
    # Why this works: a convex hull's facets are triangles whose vertices are
    # mutually closest in 3D. When all points lie on the unit sphere, "closest
    # in 3D" is monotone in great-circle distance, so the hull's facets are
    # exactly the spherical Delaunay triangles. We don't need actual Earth
    # radius — only the directions matter, so unit vectors are correct and
    # numerically well-behaved.
    #
    # What this buys us: the resulting triangulation has no edges and no
    # singularities. Triangles bridge across the dateline (a point at +179.9
    # and one at -179.9 are nearly identical 3D vectors and get connected),
    # across the poles, and across LLC tile seams (corners on either side of
    # a seam are the same physical point, so they get connected). Marching
    # triangles run on this connectivity therefore produces contours that
    # cross all those features correctly, with no special-case handling.
    xyz = _lonlat_to_xyz(lon, lat)
    triangles = ConvexHull(xyz).simplices

    tri = mtri.Triangulation(lon, lat, triangles=triangles)
    gen = _tri.TriContourGenerator(tri.get_cpp_triangulation(), z)
    segs = gen.create_contour(0.5)[0]

    pidx = pts.indexes["point"]
    f_all = np.asarray(pidx.get_level_values(face_dim))
    j_all = np.asarray(pidx.get_level_values(y_dim))
    i_all = np.asarray(pidx.get_level_values(x_dim))

    face_c_list, i_c_list, j_c_list = [], [], []
    lons_c_list, lats_c_list = [], []

    for seg in segs:
        seg = np.asarray(seg)
        if seg.ndim != 2 or seg.shape[0] < 3:
            continue

        nn = _nearest(_lonlat_to_xyz(seg[:, 0], seg[:, 1]), xyz)
        idx = np.column_stack([f_all[nn], j_all[nn], i_all[nn]])
        idx = _remove_consecutive_duplicates(idx)
        idx = _remove_backtracks(idx)
        idx = _close_loop(idx)
        if idx.shape[0] < 4:
            continue

        f, j, i = idx[:, 0].astype(np.int64), idx[:, 1].astype(np.int64), idx[:, 2].astype(np.int64)
        sel = {
            face_dim: xr.DataArray(f, dims="pt"),
            y_dim: xr.DataArray(j, dims="pt"),
            x_dim: xr.DataArray(i, dims="pt"),
        }
        face_c_list.append(f)
        i_c_list.append(i)
        j_c_list.append(j)
        lons_c_list.append(lon_c.isel(sel).values[:-1])
        lats_c_list.append(lat_c.isel(sel).values[:-1])

    return face_c_list, i_c_list, j_c_list, lons_c_list, lats_c_list


# ---------------------------------------------------------------------------
# Per-contour mask via stereographic projection
# ---------------------------------------------------------------------------

def mask_from_grid_boundaries(
    lons_c,
    lats_c,
    grid,
    along_boundary=False,
    coordnames={'h': ('geolon', 'geolat')},
):
    """Find mask bounded by a sequence of cell corner coordinates on LLC.

    Same intent and branching as `regionate.grid_conform.mask_from_grid_boundaries`,
    but adapted to a tiled grid where C-point coords are 3D (face, j, i):
      1. `along_boundary=True` -- strict polygon test, no longitude wrapping.
      2. |Δlon| < 180  -- small region; shift to 0..360 longitude convention.
      3. |Δlon| >= 180 -- circumpolar; project stereographically. The original
         hardcoded the south pole and closed the polygon to lat=-90; here the
         projection pole is the antipode of the contour's centroid, which
         generalizes to either hemisphere.

    All three branches stack the 3D (face, j, i) coords to 1D for regionmask
    and unstack the result back to (face, j, i).

    ARGUMENTS
    ---------
    lons_c [list or np.ndarray] -- cell corner longitudes
    lats_c [list or np.ndarray] -- cell corner latitudes
    grid [xgcm.Grid] -- ocean model grid
    along_boundary [bool]
    coordnames [dict] -- names of C-point lon/lat in grid._ds.
        Default {'h': ('geolon', 'geolat')}.

    RETURNS
    -------
    region_grid_mask : xr.DataArray of bool, dims (face, j, i)
    """
    lons_c = np.asarray(lons_c)
    lats_c = np.asarray(lats_c)

    lon_h = grid._ds[coordnames['h'][0]]
    lat_h = grid._ds[coordnames['h'][1]]
    crs = 'epsg:4326'

    Δlon = np.sum(np.diff(lons_c)[np.abs(np.diff(lons_c)) < 180])

    if along_boundary:
        polygon_geom = Polygon(zip(lons_c, lats_c))
        polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])
        region_grid_mask = _llc_mask_geopandas(
            polygon, lon_h, lat_h, wrap_lon=False
        )

    elif np.abs(Δlon) < 180.:
        lons = _loop(lons_c)
        lats = _loop(lats_c)
        wrapped_lons = _wrap_continuously(lons)
        minlon = np.min(wrapped_lons)
        polygon_geom = Polygon(
            zip(np.mod(wrapped_lons - minlon, 360.), lats)
        )
        polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])
        region_grid_mask = _llc_mask_geopandas(
            polygon,
            np.mod(lon_h - minlon, 360.),
            lat_h,
            wrap_lon='360',
        )

    else:
        # Circumpolar: project stereographically from the antipode of the
        # contour centroid so the loop becomes a simple planar polygon.
        # Generalization of the south-pole stereographic in the original.
        cxyz = _lonlat_to_xyz(lons_c, lats_c).mean(axis=0)
        cxyz /= np.linalg.norm(cxyz)
        pole_lon = np.rad2deg(np.arctan2(-cxyz[1], -cxyz[0]))
        pole_lat = -np.rad2deg(np.arcsin(cxyz[2]))

        cx, cy = _stereographic(lons_c, lats_c, pole_lon, pole_lat)
        poly_xy = list(zip(cx, cy))
        if poly_xy[0] != poly_xy[-1]:
            poly_xy.append(poly_xy[0])
        polygon = gpd.GeoDataFrame(
            index=[0], crs=None, geometry=[Polygon(poly_xy)]
        )

        gx, gy = _stereographic(lon_h.values, lat_h.values, pole_lon, pole_lat)
        gx_da = xr.DataArray(gx, dims=lon_h.dims, coords=lon_h.coords)
        gy_da = xr.DataArray(gy, dims=lat_h.dims, coords=lat_h.coords)
        region_grid_mask = _llc_mask_geopandas(
            polygon, gx_da, gy_da, wrap_lon=False
        )

    return region_grid_mask


def _llc_mask_geopandas(polygon, lon_da, lat_da, wrap_lon):
    """regionmask.mask_geopandas wrapper that handles 3D LLC coords by
    stacking to 1D and unstacking back.
    """
    if not isinstance(lon_da, xr.DataArray):
        lon_da = xr.DataArray(lon_da)
    if not isinstance(lat_da, xr.DataArray):
        lat_da = xr.DataArray(lat_da)

    stack_dims = list(lon_da.dims)
    lon_flat = lon_da.stack(_pt=stack_dims)
    lat_flat = lat_da.stack(_pt=stack_dims)

    mask_flat = ~np.isnan(
        regionmask.mask_geopandas(
            polygon, lon_flat, lat=lat_flat, wrap_lon=wrap_lon
        )
    )
    return mask_flat.unstack("_pt").transpose(*stack_dims)


def _loop(x):
    """Append x[0] to the end if not already closed."""
    x = np.asarray(x)
    if len(x) and x[0] != x[-1]:
        x = np.append(x, x[0])
    return x


def _wrap_continuously(x, limit_discontinuity=180.):
    new_x = x.copy().astype(float)
    for i in range(len(new_x) - 1):
        if new_x[i + 1] - new_x[i] >= 180.:
            new_x[i + 1] -= 360.
        elif new_x[i + 1] - new_x[i] < -180:
            new_x[i + 1] += 360.
    return new_x


def _stereographic(lon, lat, pole_lon, pole_lat):
    """Stereographic projection from the point (pole_lon, pole_lat).

    The projection pole maps to infinity; its antipode maps to the origin.
    Returns planar (x, y) coordinates. Inputs in degrees.
    """
    lon = np.deg2rad(np.asarray(lon))
    lat = np.deg2rad(np.asarray(lat))
    plon = np.deg2rad(pole_lon)
    plat = np.deg2rad(pole_lat)

    # Unit vectors
    cx = np.cos(lat) * np.cos(lon)
    cy = np.cos(lat) * np.sin(lon)
    cz = np.sin(lat)
    px = np.cos(plat) * np.cos(plon)
    py = np.cos(plat) * np.sin(plon)
    pz = np.sin(plat)

    # Stereographic from pole P: project ray from P through point onto the
    # plane through the antipode -P perpendicular to P. Using the standard
    # formula in a basis aligned with -P.
    #
    # Build an orthonormal frame (e1, e2) on the plane perpendicular to P.
    # e1 chosen as east-direction at -P, e2 as north-direction at -P.
    a_lat = -plat
    a_lon = plon + np.pi  # antipode lon
    e1 = np.array([-np.sin(a_lon), np.cos(a_lon), 0.0])
    e2 = np.array([
        -np.sin(a_lat) * np.cos(a_lon),
        -np.sin(a_lat) * np.sin(a_lon),
         np.cos(a_lat),
    ])

    # Stereographic scale factor: r = 2 * tan((pi - angle_to_P) / 2)
    cos_to_P = px * cx + py * cy + pz * cz
    cos_to_P = np.clip(cos_to_P, -1.0, 1.0)
    angle_to_P = np.arccos(cos_to_P)
    angle_to_antipode = np.pi - angle_to_P
    r = 2.0 * np.tan(angle_to_antipode / 2.0)

    # Direction in the plane: project point onto plane perpendicular to P,
    # then take the unit vector and scale by r.
    proj_x = cx - cos_to_P * px
    proj_y = cy - cos_to_P * py
    proj_z = cz - cos_to_P * pz
    proj_norm = np.sqrt(proj_x**2 + proj_y**2 + proj_z**2)
    proj_norm = np.where(proj_norm == 0, 1.0, proj_norm)  # avoid /0 at antipode
    ux = proj_x / proj_norm
    uy = proj_y / proj_norm
    uz = proj_z / proj_norm

    x = r * (ux * e1[0] + uy * e1[1] + uz * e1[2])
    y = r * (ux * e2[0] + uy * e2[1] + uz * e2[2])
    return x, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lonlat_to_xyz(lon, lat):
    lon = np.deg2rad(np.asarray(lon))
    lat = np.deg2rad(np.asarray(lat))
    c = np.cos(lat)
    return np.column_stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)])


def _nearest(xyz, xyz_ref):
    return np.argmax(xyz @ xyz_ref.T, axis=1)


def _remove_consecutive_duplicates(idx):
    if len(idx) == 0:
        return idx
    keep = np.ones(len(idx), dtype=bool)
    keep[1:] = np.any(idx[1:] != idx[:-1], axis=1)
    return idx[keep]


def _remove_backtracks(idx):
    if len(idx) < 3:
        return idx
    out = [idx[0]]
    for k in range(1, len(idx)):
        if len(out) >= 2 and np.array_equal(idx[k], out[-2]):
            out.pop()
        else:
            out.append(idx[k])
    return np.asarray(out)


def _close_loop(idx):
    if len(idx) and not np.array_equal(idx[0], idx[-1]):
        idx = np.vstack([idx, idx[0]])
    return idx