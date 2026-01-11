import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat


def as_list_str(x):
    x = np.atleast_1d(x)
    out = []
    for v in x.tolist():
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out


def polys_in_entry(entry) -> int:
    """
    Cuenta cuántos polígonos válidos hay en una celda:
    - entry puede ser vacío, ndarray object (cell), o ndarray numérico Nx2
    - un polígono válido: Nx2 y N>=3
    """
    if entry is None:
        return 0

    # MATLAB cell -> ndarray dtype=object
    if isinstance(entry, np.ndarray) and entry.dtype == object:
        total = 0
        for e in entry.ravel():
            total += polys_in_entry(e)
        return total

    # numeric polygon Nx2
    if isinstance(entry, np.ndarray) and entry.dtype != object:
        arr = np.asarray(entry)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3:
            return 1
        return 0

    return 0


def row_has_any_polygon(row_obj: np.ndarray) -> bool:
    """
    row_obj: shape (C,) dtype=object, una fila de labelPolys.
    True si alguna clase tiene al menos 1 polígono válido.
    """
    for entry in row_obj:
        if polys_in_entry(entry) > 0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_mat", required=True, help="gTruth_py_flat.mat de entrada")
    ap.add_argument("--out_mat", required=True, help="Salida .mat filtrada (ej: gTruth_py_flat_filtered.mat)")
    args = ap.parse_args()

    in_mat = Path(args.in_mat)
    out_mat = Path(args.out_mat)

    m = loadmat(in_mat, squeeze_me=True, struct_as_record=False)

    for k in ("imageFiles", "labelNames", "labelPolys"):
        if k not in m:
            raise RuntimeError(f"Falta '{k}' en {in_mat}. Asegúrate que sea gTruth_py_flat.mat.")

    image_files = as_list_str(m["imageFiles"])
    label_names = as_list_str(m["labelNames"])
    label_polys = np.array(m["labelPolys"], dtype=object)

    if label_polys.ndim != 2:
        raise RuntimeError(f"labelPolys debe ser 2D. Recibido: {label_polys.ndim}D")

    n_images, n_classes = label_polys.shape
    if len(image_files) != n_images:
        raise RuntimeError(f"Mismatch: imageFiles={len(image_files)} vs labelPolys rows={n_images}")

    keep_idx = []
    for i in range(n_images):
        if row_has_any_polygon(label_polys[i, :]):
            keep_idx.append(i)

    keep_idx = np.array(keep_idx, dtype=int)

    image_files_f = [image_files[i] for i in keep_idx.tolist()]
    label_polys_f = label_polys[keep_idx, :]

    print(f"Total imágenes: {n_images}")
    print(f"Con polígonos:  {len(image_files_f)}")
    print(f"Eliminadas:     {n_images - len(image_files_f)}")

    # Guardar mat filtrado (v7)
    out = {
        "imageFiles": np.array(image_files_f, dtype=object),
        "labelNames": np.array(label_names, dtype=object),
        "labelPolys": label_polys_f,
    }
    savemat(out_mat, out, do_compression=False)

    print(f"Guardado: {out_mat}")


if __name__ == "__main__":
    main()
