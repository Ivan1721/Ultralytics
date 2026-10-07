import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import loadmat
import yaml


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_list_str(arr):
    arr = np.atleast_1d(arr)
    out = []
    for x in arr.tolist():
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return out


def load_flat(mat_path: Path):
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    for k in ("imageFiles", "labelNames", "labelPolys"):
        if k not in m:
            raise RuntimeError(f"Falta '{k}' en {mat_path}. Revisa el export MATLAB.")

    image_files = to_list_str(m["imageFiles"])
    label_names = to_list_str(m["labelNames"])
    label_polys = np.array(m["labelPolys"], dtype=object)  # (N, C)

    if label_polys.ndim != 2:
        raise RuntimeError(f"labelPolys debe ser 2D, recibido {label_polys.ndim}D")

    if len(image_files) != label_polys.shape[0]:
        raise RuntimeError(f"Mismatch: imageFiles={len(image_files)} vs labelPolys rows={label_polys.shape[0]}")

    return image_files, label_names, label_polys


def normalize_polys_cell(cell_entry):
    """
    cell_entry esperado: puede ser
    - vacío
    - ndarray dtype=object (cell array) con polígonos Nx2
    - un solo polígono Nx2
    Retorna list of (Nx2 float) arrays.
    """
    if cell_entry is None:
        return []

    if isinstance(cell_entry, np.ndarray) and cell_entry.dtype == object:
        polys = []
        for e in cell_entry.ravel():
            polys.extend(normalize_polys_cell(e))
        return polys

    if isinstance(cell_entry, np.ndarray) and cell_entry.dtype != object:
        arr = np.array(cell_entry, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3:
            return [arr]
        return []

    return []


def convert(mat_path: Path, labels_out: Path, yaml_out: Path, dataset_root: str, train_rel: str, val_rel: str):
    image_files, label_names, label_polys = load_flat(mat_path)

    ensure_dir(labels_out)

    n_images, n_classes = label_polys.shape
    n_written, n_empty = 0, 0

    for i in range(n_images):
        img_path = Path(image_files[i])
        if not img_path.is_file():
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        out_txt = labels_out / f"{img_path.stem}.txt"
        lines = []

        for c in range(n_classes):
            class_id = c
            entry = label_polys[i, c]
            polys = normalize_polys_cell(entry)
            if not polys:
                continue

            for pts in polys:
                pts = np.asarray(pts, dtype=float)
                if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
                    continue

                pts[:, 0] = np.clip(pts[:, 0], 1, W)
                pts[:, 1] = np.clip(pts[:, 1], 1, H)

                x = pts[:, 0] / float(W)
                y = pts[:, 1] / float(H)

                coords = np.empty((pts.shape[0] * 2,), dtype=float)
                coords[0::2] = x
                coords[1::2] = y

                line = str(class_id) + " " + " ".join(f"{v:.6f}" for v in coords.tolist())
                lines.append(line)

        with open(out_txt, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        n_written += 1
        if not lines:
            n_empty += 1

    # data.yaml
    y = {}
    if dataset_root:
        y["path"] = dataset_root.replace("\\", "/")
    y["train"] = train_rel
    y["val"] = val_rel
    y["names"] = {i: name for i, name in enumerate(label_names)}

    ensure_dir(Path(yaml_out).parent)
    with open(yaml_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(y, f, sort_keys=False, allow_unicode=True)

    print("OK")
    print(f"Labels escritos: {n_written}")
    print(f"Labels vacíos:   {n_empty}")
    print(f"Labels dir:      {labels_out}")
    print(f"data.yaml:       {yaml_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True, help="Ruta a gTruth_py_flat.mat (export MATLAB plano)")
    ap.add_argument("--labels_out", required=True, help="Carpeta salida labels")
    ap.add_argument("--yaml_out", required=True, help="Ruta salida data.yaml")
    ap.add_argument("--dataset_root", default="", help="Root para 'path:' en YAML (opcional)")
    ap.add_argument("--train_rel", default="images/train", help="train relativo YAML")
    ap.add_argument("--val_rel", default="images/val", help="val relativo YAML")
    args = ap.parse_args()

    convert(
        mat_path=Path(args.mat),
        labels_out=Path(args.labels_out),
        yaml_out=Path(args.yaml_out),
        dataset_root=args.dataset_root,
        train_rel=args.train_rel,
        val_rel=args.val_rel,
    )


if __name__ == "__main__":
    main()
