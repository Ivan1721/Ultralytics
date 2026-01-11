function export_gtruth_for_python(inMat, outMat)
% Exporta un groundTruth (Image Labeler) a un .mat "plano" (v7) compatible con SciPy:
% - imageFiles: cellstr (N x 1)
% - labelNames: cellstr (C x 1) (clases, en orden 0..C-1)
% - labelPolys: cell (N x C), cada celda contiene una cell array de polígonos
%              cada polígono es Nx2 double en pixeles [x y]
%
% Uso:
%   export_gtruth_for_python("gTruthV1.mat","gTruth_py_flat.mat")

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

    T = gTruth.LabelData; % tabla: filas=imagenes, columnas=clases

    N = height(T);
    C = width(T);

    labelPolys = cell(N, C);

    for r = 1:N
        for c = 1:C
            entry = T{r,c};

            % Normalizamos a: cell array de polígonos Nx2 double
            polys = normalizeEntryToVertexCells(entry);

            labelPolys{r,c} = polys;
        end
    end

    % Guardar PLANO (v7)
    save(outMat, "imageFiles", "labelNames", "labelPolys", "-v7");
    fprintf("OK: guardado %s (N=%d, C=%d)\n", outMat, N, C);
end


function polys = normalizeEntryToVertexCells(entry)
% Convierte el contenido de una celda de LabelData a:
%   polys = { P1, P2, ... } donde Pi es Nx2 double [x y]
% Soporta:
% - empty
% - polyshape
% - numeric Nx2
% - cell anidada de lo anterior
% - struct con campo Vertices

    polys = {};

    if isempty(entry)
        return;
    end

    if iscell(entry)
        % si viene doble celda: { { ... } }
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

    if isa(entry, "polyshape")
        [x,y] = boundary(entry);
        pts = [x(:) y(:)];
        if size(pts,1) >= 3
            polys = {pts};
        end
        return;
    end

    if isnumeric(entry) && size(entry,2) == 2 && size(entry,1) >= 3
        polys = {double(entry)};
        return;
    end

    if isstruct(entry) && isfield(entry, "Vertices")
        v = entry.Vertices;
        if isnumeric(v) && size(v,2) == 2 && size(v,1) >= 3
            polys = {double(v)};
        end
        return;
    end
end
