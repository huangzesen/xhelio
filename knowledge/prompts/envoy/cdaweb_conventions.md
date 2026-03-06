## CDAWeb Dataset ID Conventions

- Some CDAWeb datasets use `@N` suffixes (e.g., `PSP_FLD_L2_RFS_LFR@2`, `WI_H0_MFI@0`).
  These are **valid sub-datasets** that split large datasets into manageable parts.
  Treat them exactly like regular dataset IDs — pass them to `fetch_data` and `list_parameters` as-is.
- Attitude datasets (`_AT_`), orbit datasets (`_ORBIT_`, `_OR_`), and key-parameter
  datasets (`_K0_`, `_K1_`, `_K2_`) are all valid CDAWeb datasets that can be fetched normally.
- Cross-mission datasets like `OMNI_COHO1HR_MERGED_MAG_PLASMA` or `SOLO_HELIO1HR_POSITION`
  are merged products from COHOWeb/HelioWeb — also valid for fetch_data.