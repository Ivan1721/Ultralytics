# Occlusion-Aware Fruit Instance Segmentation

Investigación comparativa de segmentación de instancias para percepción de frutas en robótica agrícola bajo oclusión, comparando **YOLO11-seg**, **YOLO26-seg** y **Mask2Former** sobre un dataset propio capturado con un brazo robótico 6DOF en un huerto real (manzana verde, manzana roja, durazno, palta, pera y naranja).

El trabajo está escrito como el artículo *"Occlusion-Aware Fruit Instance Segmentation for Agricultural Robotics: A Comparative Study of YOLO11, YOLO26, and Mask2Former"*, enviado a **ARTIIS 2026** (submission #9311, en revisión — ver [Estado del paper](#estado-del-paper)).

## Ecosistema de repositorios

Este proyecto vive en tres carpetas hermanas bajo `C:\Users\garci\repos\`, cada una con su propio rol:

```
C:\Users\garci\repos\
├── Ultralytics\      (este repo) — pipeline YOLO11/YOLO26, análisis, y el manuscrito
├── mmdetection\       fork de open-mmlab/mmdetection — entrenamiento/eval de Mask2Former, DETR, Mask-RCNN
└── dataset\
    ├── yolo\          dataset en formato YOLO (usado por este repo)
    └── coco\          el mismo dataset en formato COCO (usado por mmdetection)
```

`dataset/` es compartido entre los dos repos y no pertenece a ninguno de los dos en git — ambos lo referencian con rutas relativas. Si alguna de estas tres carpetas se mueve, hay que revisar las rutas relativas que cruzan entre ellas (ver la nota en `CLAUDE.md`).

## Estructura de este repo

```
Ultralytics/
├── notebooks/              los 3 notebooks del pipeline (InstanceSeg_Code.ipynb es el vigente)
├── scripts/                conversión MATLAB groundTruth -> dataset YOLO
├── groundtruth_exports/    exports .mat crudos/filtrados desde MATLAB
├── results/                figuras y tablas generadas para el paper
├── paper/                  manuscrito LaTeX (ver paper/Articulo/sn-articleOK.tex)
├── CLAUDE.md               guía técnica detallada del pipeline (para trabajar con Claude Code)
└── Manzana/, runs/, ...    datos crudos y salidas de entrenamiento (ignorados por git)
```

Para el detalle técnico de cada paso (comandos exactos, convenciones de nombres de carpetas, bugs conocidos de los scripts), ver **[CLAUDE.md](CLAUDE.md)** — está pensado para que cualquiera (humano o agente) retome el proyecto sin tener que releer todo el historial de conversación.

## Pipeline, en una línea por paso

1. Etiquetar en MATLAB Image Labeler (polígonos por instancia + nivel de visibilidad por imagen, asignado manualmente).
2. Exportar el `groundTruth` a un `.mat` plano compatible con Python (`scripts/export_gtruth_for_python.m`).
3. Filtrar filas vacías (`scripts/filter_gtruth_flat_no_empty.py`).
4. Convertir a formato YOLO (`scripts/gtruth_flat_to_yolo.py`) → `dataset/yolo/`.
5. Entrenar YOLO11-seg y YOLO26-seg (n/s/m/l/x) desde `notebooks/InstanceSeg_Code.ipynb`; entrenar Mask2Former desde `mmdetection/` usando `dataset/coco/`.
6. Validar por nivel de visibilidad (25/50/75/100%, definidos manualmente durante el etiquetado) para medir robustez ante oclusión.
7. Generar las figuras/tablas comparativas → `results/` y `paper/Articulo/`.

## Entorno

```powershell
conda env create -f yolo.yml      # entorno YOLO (ver nota en CLAUDE.md sobre el nombre del archivo)
conda activate yolo
```

Mask2Former corre en un entorno `openmmlab` separado dentro de `mmdetection/` (ver ese repo). En esta máquina hay dos variantes (`openmmlab` y `openmmlab241fix`) con distinta compatibilidad de extensiones compiladas de `mmcv` — si una falla con `DLL load failed` al importar `mmcv`, probar con la otra.

## Estado del paper

Recibió 2 revisiones en ARTIIS 2026 (accept / weak accept) con observaciones metodológicas, ya incorporadas al manuscrito:

- Protocolo de visibilidad documentado explícitamente (criterio manual + conteos por nivel).
- GPU exacta especificada (RTX 4060 Ti).
- Split real (71%/29%) corregido en la Figura 1, que decía 80%/20% por error.
- Ausencia de test set independiente reconocida como limitación.
- Afirmaciones de "tiempo real"/embebido moderadas (la latencia reportada es solo inferencia del modelo, no el pipeline robótico completo).
- Criterio explícito para la afirmación de "mejor trade-off" de YOLO11m.

Pendiente: revisión general de inglés/ortografía, y priorizar referencias publicadas sobre preprints donde exista una versión formal.

## Dataset (resumen)

6 clases, 1224 imágenes de entrenamiento / 491 de validación. Ver `paper/Articulo/sn-articleOK.tex` (Tabla 1) para la distribución completa por clase, y `results/validation_by_visibility/` para los resultados desglosados por nivel de oclusión.
