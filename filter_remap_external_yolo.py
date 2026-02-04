import argparse
import shutil
from pathlib import Path
import yaml


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(p: Path, obj):
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def list_images(img_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in img_dir.glob("*") if p.suffix.lower() in exts]


def filter_and_remap_label(label_path: Path, src_id_to_tgt_id: dict[int, int]):
    if not label_path.exists():
        return None  # no label -> omitimos

    out_lines = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cid = int(float(parts[0]))
        except:
            continue

        if cid not in src_id_to_tgt_id:
            continue

        new_cid = src_id_to_tgt_id[cid]
        out_lines.append(" ".join([str(new_cid)] + parts[1:]))

    return out_lines


def copy_pair(img_path: Path, out_img_dir: Path, out_lbl_dir: Path, new_stem: str, label_lines: list[str]):
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    dst_img = out_img_dir / f"{new_stem}{img_path.suffix.lower()}"
    dst_lbl = out_lbl_dir / f"{new_stem}.txt"

    shutil.copy2(img_path, dst_img)
    with open(dst_lbl, "w", encoding="utf-8") as f:
        for l in label_lines:
            f.write(l + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external_root", required=True, help="Raíz del dataset externo (donde está data.yaml)")
    ap.add_argument("--out_root", required=True, help="Raíz del dataset final unificado")
    ap.add_argument("--tag", default="external", help="Prefijo para renombrar archivos (evita colisiones)")
    ap.add_argument("--drop_empty", action="store_true", help="Descartar imágenes que queden sin instancias tras filtrar")
    args = ap.parse_args()

    external_root = Path(args.external_root)
    out_root = Path(args.out_root)

    cfg = read_yaml(external_root / "data.yaml")
    names = cfg["names"]  # lista de nombres
    if isinstance(names, dict):
        # por si viene como dict
        names = [names[i] for i in sorted(names.keys(), key=lambda x: int(x))]

    # localizar IDs de las clases fuente
    def find_id(class_name: str) -> int:
        try:
            return names.index(class_name)
        except ValueError:
            raise RuntimeError(f"No encontré la clase '{class_name}' en names del data.yaml externo.")

    src_green = find_id("green_apple")
    src_red   = find_id("red_apple")

    # remapeo: dataset externo -> dataset final
    # green_apple -> apple_green (0)
    # red_apple   -> apple_red   (1)
    src_to_tgt = {src_green: 0, src_red: 1}

    print("Mapping externo -> final:", src_to_tgt)

    # splits típicos Roboflow (ojo: tu yaml usa ../train/images etc.)
    # Asumimos que external_root apunta al folder "data.yaml" y que train/valid/test están como carpetas hermanas.
    # Ej: external_root = dataset/data/  y train está en dataset/train/images
    # Por eso resolvemos rutas relativas desde external_root.
    splits = [
        ("train", cfg.get("train")),
        ("val",   cfg.get("val")),
        ("test",  cfg.get("test")),
    ]

    kept_total = dropped_total = scanned_total =_
