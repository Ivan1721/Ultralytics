function export_gtruth_for_python(inMat, outMat, varargin)
% Exporta un groundTruth (Image Labeler) a un .mat "plano" (v7) compatible con SciPy:
% - imageFiles: cellstr (N x 1)
% - labelNames: cellstr (C x 1) (clases, en orden 0..C-1)
% - labelPolys: cell (N x C), cada celda contiene una cell array de polígonos
%              cada polígono es Nx2 double en pixeles [x y]
%
% Uso:
%   export_gtruth_for_python("gTruthV1.mat","gTruth_py_flat.mat")
%
% Opcional:
%   export_gtruth_for_python(..., 'dropEmptyLabels', true)
%   -> elimina labels globalmente vacíos (sin instancias en todo el dataset)
%
% Nota: Esto maneja PolygonROI / images.roi.Polygon / polyshape y variantes.

    p = inputParser;
    addParameter(p, 'dropEmptyLabels', false, @(x)islogical(x) && isscalar(x));
    parse(p, varargin{:});
    dropEmptyLabels = p.Results.dropEmptyLabels;

    S = load(inMat);

    % encontrar objeto groundTruth
    gTruth = [];
    fn = fieldnames(S);
    for i = 1:numel(fn)
        if isa(S.(fn{i}), "groundTruth")
            gTruth = S.(fn{i});
            break;
        end
    end
    if isempty(gTruth)
        error("No se encontró un objeto groundTruth en %s", inMat);
    end

    imageFiles = gTruth.DataSource.Source;
    labelNames = gTruth.LabelDefinitions.Name;  % orden de clases
    T = gTruth.LabelData;                       % tabla: filas=imagenes, columnas=clases

    % (Opcional) eliminar labels globalmente vacíos ANTES de exportar
    if dropEmptyLabels
        keepCols = true(1, width(T));
        for c = 1:width(T)
            colHasAny = false;
            for r = 1:height(T)
                polys = normalizeEntryToVertexCells(T{r,c});
                if ~isempty(polys)
                    colHasAny = true;
                    break;
                end
            end
            keepCols(c) = colHasAny;
        end

        removed = labelNames(~keepCols);
        if any(~keepCols)
            fprintf("Eliminando labels vacíos globalmente (%d):\n", sum(~keepCols));
            disp(removed);
        end

        T = T(:, keepCols);
        labelNames = labelNames(keepCols);
    end

    N = height(T);
    C = width(T);

    labelPolys = cell(N, C);

    for r = 1:N
        for c = 1:C
            entry = T{r,c};
            polys = normalizeEntryToVertexCells(entry);
            labelPolys{r,c} = polys;
        end
    end

    % Guardar PLANO (v7)
    save(outMat, "imageFiles", "labelNames", "labelPolys", "-v7");
    fprintf("OK: guardado %s (N=%d, C=%d)\n", outMat, N, C);
end


function polys = normalizeEntryToVertexCells(entry)
% Devuelve:
%   polys = { P1, P2, ... } donde Pi es Nx2 double [x y]
%
% Soporta:
% - empty
% - cell anidada
% - polyshape  (boundary)
% - images.roi.Polygon / PolygonROI / ROI con propiedad Position (Nx2)
% - objetos con propiedad Vertices (Nx2)
% - struct con campo Vertices (Nx2)
% - numeric Nx2

    polys = {};

    if isempty(entry)
        return;
    end

    % Cell (posiblemente anidada)
    if iscell(entry)
        if numel(entry) == 1 && iscell(entry{1})
            entry = entry{1};
        end
        for i = 1:numel(entry)
            polys_i = normalizeEntryToVertexCells(entry{i});
            if ~isempty(polys_i)
                polys = [polys, polys_i]; %#ok<AGROW>
            end
        end
        return;
    end

    % polyshape
    if isa(entry, "polyshape")
        try
            [x,y] = boundary(entry);
            pts = [x(:) y(:)];
            if isValidPts(pts)
                polys = {double(pts)};
            end
        catch
        end
        return;
    end

    % numeric Nx2
    if isnumeric(entry)
        pts = double(entry);
        if isValidPts(pts)
            polys = {pts};
        end
        return;
    end

    % struct con Vertices
    if isstruct(entry) && isfield(entry, "Vertices")
        pts = double(entry.Vertices);
        if isValidPts(pts)
            polys = {pts};
        end
        return;
    end

    % Objetos ROI: images.roi.Polygon, vision.internal... etc.
    % Caso típico: propiedad Position (Nx2)
    if isobject(entry)
        % 1) Position
        pts = tryGetProp(entry, 'Position');
        if ~isempty(pts)
            pts = double(pts);
            if isValidPts(pts)
                polys = {pts};
                return;
            end
        end

        % 2) Vertices
        pts = tryGetProp(entry, 'Vertices');
        if ~isempty(pts)
            pts = double(pts);
            if isValidPts(pts)
                polys = {pts};
                return;
            end
        end
    end

    % Si llega aquí: tipo no soportado → devuelve vacío (seguro)
end


function v = tryGetProp(obj, propName)
% Obtiene propiedad si existe sin reventar.
    v = [];
    try
        if isprop(obj, propName)
            v = obj.(propName);
        end
    catch
        v = [];
    end
end


function ok = isValidPts(pts)
% Valida Nx2 y N>=3, y que no sea NaN/Inf.
    ok = isnumeric(pts) && ndims(pts) == 2 && size(pts,2) == 2 && size(pts,1) >= 3 ...
         && all(isfinite(pts(:)));
end
