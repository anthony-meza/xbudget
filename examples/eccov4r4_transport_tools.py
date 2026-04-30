import numpy as np 
import xarray as xr
import xgcm 
from xbudget.llc90 import * 

def calc_uv_bolus(ds, grid, allow_rechunk=True):
    psix = ds["GM_PsiX"].where(np.isfinite(ds["GM_PsiX"]), 0.0)
    psiy = ds["GM_PsiY"].where(np.isfinite(ds["GM_PsiY"]), 0.0)

    psix_u = psix * ds["dyG"]
    psiy_v = psiy * ds["dxG"]

    UBOL_TRSP = grid.diff(psix_u.chunk({"k_l": -1}), "Z", boundary="fill")
    VBOL_TRSP = grid.diff(psiy_v.chunk({"k_l": -1}), "Z", boundary="fill")

    psi_diff = diff_2d_flux_llc90(
        grid,
        psix_u.unify_chunks(),
        psiy_v.unify_chunks(),
        allow_rechunk=allow_rechunk,
    )

    WBOL_TRSP = psi_diff["X"] + psi_diff["Y"]

    UBOL_TRSP = UBOL_TRSP.where(ds["maskW"]).fillna(0.0)
    VBOL_TRSP = VBOL_TRSP.where(ds["maskS"]).fillna(0.0)
    WBOL_TRSP = WBOL_TRSP.fillna(0.0)

    UVELSTAR = UBOL_TRSP / (ds["drF"] * ds["dyG"])
    VVELSTAR = VBOL_TRSP / (ds["drF"] * ds["dxG"])
    WVELSTAR = WBOL_TRSP / ds["rA"]

    return xr.Dataset(
        {
            "UVELSTAR": UVELSTAR.rename("UVELSTAR"),
            "VVELSTAR": VVELSTAR.rename("VVELSTAR"),
            "WVELSTAR": WVELSTAR.rename("WVELSTAR"),
        }
    )

def remap_transport(ds, trsp, coord_name, grid, coord_edges, method="histogram"):
    coord_edges = np.asarray(coord_edges)
    coord_centers = 0.5 * (coord_edges[:-1] + coord_edges[1:])

    if "i_g" in trsp.dims:
        face_axis = "X"
        mask = ds["maskW"]
    elif "j_g" in trsp.dims:
        face_axis = "Y"
        mask = ds["maskS"]
    else:
        raise ValueError("Expected transport on `i_g` or `j_g`.")

    coord_face = grid.interp(
        ds[coord_name].chunk({"tile": -1, "i": -1, "j": -1, "k": -1}),
        face_axis,
    )

    coord_face = coord_face.where(mask)
    trsp = trsp.where(mask).fillna(0.0)

    if method == "histogram":
        trsp_remapped = histogram(coord_face, bins=[coord_edges], weights=trsp, dim=["k"])
        trsp_remapped = trsp_remapped.rename({f"{coord_name}_bin": coord_name})
        trsp_remapped = trsp_remapped.assign_coords({coord_name: coord_centers})
        return trsp_remapped

    if method == "conservative":
        coord_face_p1 = grid.interp(
            coord_face.chunk({"k": -1}),
            "Z", to="outer", boundary="extend")

        trsp_remapped = grid.transform(
            trsp.chunk({"k": -1}),
            axis="Z",
            target=coord_edges,
            target_data=coord_face_p1.chunk({"k_p1": -1}),
            method="conservative")

        if "remapped" in trsp_remapped.dims:
            trsp_remapped = trsp_remapped.rename({"remapped": coord_name})

        trsp_remapped = trsp_remapped.assign_coords({coord_name: coord_centers})
        return trsp_remapped

    raise ValueError("method must be either 'histogram' or 'conservative'")


def remap_transports(
    ds,
    grid,
    URES_TRSP,
    VRES_TRSP,
    coord_edges,
    coord_name="SIG2",
    method="histogram",
):
    remap_func = lambda uv_trsp: remap_transport(ds, uv_trsp, coord_name, 
                                        grid, coord_edges, method=method)
    U_remapped = remap_func(URES_TRSP)
    V_remapped = remap_func(VRES_TRSP)

    return U_remapped, V_remapped


def calc_psi_from_remapped_transports(
    ds,
    grid,
    U_remapped,
    V_remapped,
    coord_name="SIG2",
    allow_rechunk=True,
):
    diff_UV_RES = diff_2d_flux_llc90(
        grid,
        U_remapped.fillna(0.0),
        V_remapped.fillna(0.0),
        allow_rechunk=allow_rechunk,
    )

    UVRES_CONV = -(diff_UV_RES["X"].fillna(0.0) + diff_UV_RES["Y"].fillna(0.0))
    UVRES_CONV = UVRES_CONV.mean("time").compute()

    lat_vals = np.arange(-88, 88)

    lats_da = xr.DataArray(
        lat_vals,
        coords={"lat": lat_vals},
        dims=("lat",),
    )

    psi = xr.zeros_like(UVRES_CONV[coord_name] * lats_da)
    psi.coords[coord_name] = UVRES_CONV[coord_name]

    def reverse_cumsum(da, dim):
        return da.isel({dim: slice(None, None, -1)}).cumsum(dim).isel({dim: slice(None, None, -1)})

    for lat in lat_vals:
        latmask = ds["YC"] < lat
        masked_conv = (UVRES_CONV * latmask).sum(["tile", "i", "j"])

        psi.loc[{"lat": lat}] = reverse_cumsum(masked_conv, coord_name).compute()

    return 1e-6 * psi