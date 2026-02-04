import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def color_for_class(cls_id: int):
    # Colores fijos y distintos por clase (BGR)
    palette = [
        (50, 205, 50),   # green-ish
        (0, 0, 255),     # red
        (255, 200, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    return palette[cls_id % len(palette)]


def overlay_masks(image_bgr, masks_xy, class_ids, scores, names, alpha=0.45, draw_contours=True):
    """
    image_bgr: np.ndarray HxWx3
    masks_xy: list of np.ndarray (N_i x 2) in pixel coords
    class_ids: list[int]
    scores: list[float]
    """
    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()

    for poly, cls_id, sc in zip(masks_xy, class_ids, scores):
        if poly is None or len(poly) < 3:
            continue

        poly_int = np.round(poly).astype(np.int32)
        poly_int[:, 0] = np.clip(poly_int[:, 0], 0, w - 1)
        poly_int[:, 1] = np.clip(poly_int[:, 1], 0, h - 1)

        color = color_for_class(int(cls_id))

        # Relleno de la máscara
        cv2.fillPoly(overlay, [poly_int], color)

        # Contorno (más legible en póster)
        if draw_contours:
            cv2.polylines(overlay, [poly_int], True, (0, 0, 0), 2, lineType=cv2.LINE_AA)

        # Etiqueta
        label = f"{names[int(cls_id)]} {sc:.2f}"
        x0, y0 = int(poly_int[0, 0]), int(poly_int[0, 1])
        y0 = max(20, y0)
        cv2.putText(overlay, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Alpha blend
    out = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Ruta al .pt (best.pt) de tu YOLO11-seg")
    ap.add_argument("--input", required=True, help="Imagen o carpeta con imágenes")
    ap.add_argument("--out", required=True, help="Carpeta de salida")
    ap.add_argument("--imgsz", type=int, default=960, help="Tamaño de inferencia (mayor = más detalle, más lento)")
    ap.add_argument("--conf", type=float, default=0.05, help="Confidence bajo => más instancias (más cobertura)")
    ap.add_argument("--iou", type=float, default=0.6, help="IoU NMS (más alto => menos supresión)")
    ap.add_argument("--max_det", type=int, default=300, help="Máx detecciones por imagen")
    ap.add_argument("--alpha", type=float, default=0.45, help="Transparencia de máscara")
    ap.add_argument("--dpi", type=int, default=300, help="Solo para nombre; la imagen sale en resolución original")
    args = ap.parse_args()

    weights = Path(args.weights)
    inp = Path(args.input)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    model = YOLO(str(weights))
    names = model.names  # dict id->name

    if inp.is_dir():
        images = [p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        images = [inp]

    if not images:
        raise SystemExit("No se encontraron imágenes en la ruta de entrada.")

    for img_path in images:
        # Inference: parámetros para “máxima cobertura”
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            retina_masks=True,  # máscaras más nítidas
            agnostic_nms=False,
            verbose=False,
        )

        r = results[0]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        if r.masks is None or r.boxes is None:
            # Sin detecciones: guarda original (útil para mostrar fallos si quieres)
            out_path = out_dir / f"{img_path.stem}_pred.png"
            cv2.imwrite(str(out_path), img_bgr)
            continue

        # Polígonos en pixeles
        masks_xy = r.masks.xy  # list of Nx2 float
        cls_ids = r.boxes.cls.cpu().numpy().astype(int).tolist()
        scores = r.boxes.conf.cpu().numpy().tolist()

        out_img = overlay_masks(
            image_bgr=img_bgr,
            masks_xy=masks_xy,
            class_ids=cls_ids,
            scores=scores,
            names=names,
            alpha=args.alpha,
            draw_contours=True,
        )

        out_path = out_dir / f"{img_path.stem}_pred.png"
        cv2.imwrite(str(out_path), out_img)

    print(f"Listo. Resultados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
