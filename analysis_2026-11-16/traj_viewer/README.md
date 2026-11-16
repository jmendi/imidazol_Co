# 3D trajectory viewer

Also published as a Claude artifact (private link, owner only).

Contents:
- `index.html`: the page. It uses 3Dmol.js and reads its data from `data/<system>_<T>.json`.
- `export_viewer_data.py`: writes those data files from `cif_runs/<system>/nptaniso_<T>`.

What the export keeps:
- every 40th frame (one frame per 40 fs);
- molecules kept whole and continuous across the periodic boundaries;
- the cell of every frame;
- the watched Co–N bonds (100 fs mean ending above 2.4 Å, or peaking above 2.45 Å).

To view it locally:
1. Write the data:
   `python export_viewer_data.py data`
2. Serve the folder (opening `index.html` directly from disk will not work, because the page loads its data with `fetch`):
   `python -m http.server`
