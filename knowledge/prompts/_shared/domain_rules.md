## Domain Rules

- **Data labels**: Fetched data is stored as `DATASET.PARAM` (e.g., `PSP_FLD_L2_MAG_RTN.psp_fld_l2_mag_RTN`). Never invent labels — always use labels reported by agents or shown in `xhelio__assets`.
- **Variable naming**: In `xhelio__run_code`, input data is staged as files. Read DataFrames with `pd.read_parquet('LABEL.parquet')` and xarray DataArrays with `xr.open_dataarray('LABEL.nc')`. Available libraries: `pandas`, `numpy`, `xarray`, `scipy`, `pywt`, `numba`, `sklearn`, `statsmodels`, `astropy`, `lmfit`, `sympy`.
- **Dataset conventions**: Some datasets use `@N` sub-dataset suffixes (e.g., `DATASET_RFS_LFR@2`). These are valid — pass them as-is. Time ranges use `" to "` separator, never `"/"`.
- **NaN handling**: If a parameter returns all NaN, skip it and try the next candidate. Do not retry the same parameter.
- **3D data**: xarray DataArray entries (3D+) must be reduced to 2D via DataOps before visualization. The viz agent cannot handle 3D data directly.
- **Log scale**: Apply log transforms in DataOps (`np.log10`), not in visualization. The viz agent has no log-z capability.
- **Shape/annotation limit**: Plotly figures are limited to 30 shapes + annotations total. For many xhelio__events, show only the most significant.
- **Data fetching**: Data fetching is routed through envoy agents. Describe the physical quantity instead of specifying parameter names.