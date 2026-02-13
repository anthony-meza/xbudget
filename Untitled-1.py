
import numpy as np
import xarray as xr


grid = xgcm.Grid(ds_ecco..,,, face)

grid.diff_2d_vector()


def diff_2d_flux_llc90_ghost(grid, Fx, Fy, face_dim=None):
    if face_dim is None:
        keys = list(grid._face_connections.keys())
        if len(keys) != 1:
            raise ValueError("Provide face_dim when grid has multiple face-connection keys.")
        face_dim = keys[0]

    ds = grid._ds

    def _center_dim(ax):
        return grid.axes[ax].coords["center"]

    def _stag_dim(ax, da):
        c = [c for pos, c in grid.axes[ax].coords.items() if pos != "center" and c in da.dims]
        if len(c) != 1:
            raise ValueError("Flux difference inconsistent with finite volume discretization.")
        return c[0]

    Xc, Yc = _center_dim("X"), _center_dim("Y")
    Xs, Ys = _stag_dim("X", Fx), _stag_dim("Y", Fy)

    xs_new = int(Fx[Xs].isel({Xs: -1}).values) + 1
    ys_new = int(Fy[Ys].isel({Ys: -1}).values) + 1

    tiles = Fx[face_dim].values

    def ghost_X(t):
        t = int(t)
        if 0 <= t <= 2:
            g = Fx.sel({face_dim: t + 3}).isel({Xs: 0})
        elif 3 <= t <= 5:
            g = Fy.sel({face_dim: 12 - t}).isel({Ys: 0, Xc: slice(None, None, -1)}).rename({Xc: Yc}).assign_coords({Yc: ds[Yc]})
        elif t == 6:
            g = Fx.sel({face_dim: 7}).isel({Xs: 0})
        elif 7 <= t <= 8:
            g = Fx.sel({face_dim: t + 1}).isel({Xs: 0})
        elif 10 <= t <= 11:
            g = Fx.sel({face_dim: t + 1}).isel({Xs: 0})
        else:
            g = xr.full_like(Fx.sel({face_dim: t}).isel({Xs: 0}), np.nan)
        return g.expand_dims({face_dim: [t], Xs: [xs_new]})

    def ghost_Y(t):
        t = int(t)
        if 0 <= t <= 1:
            g = Fy.sel({face_dim: t + 1}).isel({Ys: 0})
        elif t == 2:
            g = Fx.sel({face_dim: 6}).isel({Xs: 0, Yc: slice(None, None, -1)}).rename({Yc: Xc}).assign_coords({Xc: ds[Xc]})
        elif 3 <= t <= 5:
            g = Fy.sel({face_dim: t + 1}).isel({Ys: 0})
        elif t == 6:
            g = Fx.sel({face_dim: 10}).isel({Xs: 0, Yc: slice(None, None, -1)}).rename({Yc: Xc}).assign_coords({Xc: ds[Xc]})
        elif 7 <= t <= 9:
            g = Fy.sel({face_dim: t + 3}).isel({Ys: 0})
        elif 10 <= t <= 12:
            g = Fx.sel({face_dim: 12 - t}).isel({Xs: 0, Yc: slice(None, None, -1)}).rename({Yc: Xc}).assign_coords({Xc: ds[Xc]})
        else:
            g = xr.full_like(Fy.sel({face_dim: t}).isel({Ys: 0}), np.nan)
        return g.expand_dims({face_dim: [t], Ys: [ys_new]})

    gx = xr.concat([ghost_X(t) for t in tiles], dim=face_dim, coords="minimal", compat="override", join="override")
    gy = xr.concat([ghost_Y(t) for t in tiles], dim=face_dim, coords="minimal", compat="override", join="override")

    Fx_p = xr.concat([Fx, gx], dim=Xs, coords="minimal", compat="override", join="override").chunk({Xs: -1})
    Fy_p = xr.concat([Fy, gy], dim=Ys, coords="minimal", compat="override", join="override").chunk({Ys: -1})

    return {
        "X": Fx_p.diff(Xs).rename({Xs: Xc}).assign_coords({Xc: ds[Xc]}),
        "Y": Fy_p.diff(Ys).rename({Ys: Yc}).assign_coords({Yc: ds[Yc]}),
    }
