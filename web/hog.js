export const HOG = {
  width: 450,
  height: 600,
  orientations: 9,
  cell: 16,
  block: 2,
  epsilon: 1e-5,
  dimensions: 34992,
};

const RAD_TO_DEG = 180 / Math.PI;

function areaSpans(source, target) {
  const scale = source / target;
  const table = new Array(target);
  for (let index = 0; index < target; index += 1) {
    const start = index * scale;
    const end = start + scale;
    const cellWidth = Math.min(scale, source - start);
    const first = Math.ceil(start);
    const last = Math.floor(end);
    const entries = [];
    if (first - start > 1e-3) entries.push([first - 1, Math.fround((first - start) / cellWidth)]);
    for (let pixel = first; pixel < last; pixel += 1) entries.push([pixel, Math.fround(1 / cellWidth)]);
    if (end - last > 1e-3) entries.push([last, Math.fround((end - last) / cellWidth)]);
    table[index] = entries;
  }
  return table;
}

function roundHalfEven(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction > 0.5) return floor + 1;
  if (fraction < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

export function areaResize(rgba, sourceWidth, sourceHeight, width, height) {
  const output = new Uint8ClampedArray(width * height * 3);
  if (sourceWidth < width || sourceHeight < height) {
    for (let y = 0; y < height; y += 1) {
      const sourceY = Math.min(sourceHeight - 1, Math.floor((y * sourceHeight) / height));
      for (let x = 0; x < width; x += 1) {
        const sourceX = Math.min(sourceWidth - 1, Math.floor((x * sourceWidth) / width));
        const from = (sourceY * sourceWidth + sourceX) * 4;
        const to = (y * width + x) * 3;
        output[to] = rgba[from];
        output[to + 1] = rgba[from + 1];
        output[to + 2] = rgba[from + 2];
      }
    }
    return output;
  }

  const columns = areaSpans(sourceWidth, width);
  const rows = areaSpans(sourceHeight, height);
  for (let y = 0; y < height; y += 1) {
    const rowSpans = rows[y];
    for (let x = 0; x < width; x += 1) {
      const columnSpans = columns[x];
      let red = 0;
      let green = 0;
      let blue = 0;
      for (let r = 0; r < rowSpans.length; r += 1) {
        const [sourceY, weightY] = rowSpans[r];
        const base = sourceY * sourceWidth;
        for (let c = 0; c < columnSpans.length; c += 1) {
          const [sourceX, weightX] = columnSpans[c];
          const weight = Math.fround(weightX * weightY);
          const offset = (base + sourceX) * 4;
          red = Math.fround(red + Math.fround(rgba[offset] * weight));
          green = Math.fround(green + Math.fround(rgba[offset + 1] * weight));
          blue = Math.fround(blue + Math.fround(rgba[offset + 2] * weight));
        }
      }
      const target = (y * width + x) * 3;
      output[target] = roundHalfEven(red);
      output[target + 1] = roundHalfEven(green);
      output[target + 2] = roundHalfEven(blue);
    }
  }
  return output;
}

export function toGray(rgb, width, height) {
  const pixels = width * height;
  const gray = new Uint8Array(pixels);
  for (let i = 0; i < pixels; i += 1) {
    const offset = i * 3;
    const value = (rgb[offset] * 4899 + rgb[offset + 1] * 9617 + rgb[offset + 2] * 1868 + 8192) >> 14;
    gray[i] = value > 255 ? 255 : value;
  }
  return gray;
}

export function extractCells(gray, width, height) {
  const pixels = width * height;
  const values = new Float64Array(pixels);
  for (let i = 0; i < pixels; i += 1) values[i] = gray[i] / 255;

  const gradientRow = new Float64Array(pixels);
  const gradientColumn = new Float64Array(pixels);
  for (let y = 1; y < height - 1; y += 1) {
    const row = y * width;
    const above = row - width;
    const below = row + width;
    for (let x = 0; x < width; x += 1) {
      gradientRow[row + x] = values[below + x] - values[above + x];
    }
  }
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 1; x < width - 1; x += 1) {
      gradientColumn[row + x] = values[row + x + 1] - values[row + x - 1];
    }
  }

  const cell = HOG.cell;
  const orientations = HOG.orientations;
  const rows = Math.floor(height / cell);
  const columns = Math.floor(width / cell);
  const histogram = new Float64Array(rows * columns * orientations);
  const binWidth = 180 / orientations;

  for (let y = 0; y < rows * cell; y += 1) {
    const cellRow = Math.floor(y / cell);
    const row = y * width;
    for (let x = 0; x < columns * cell; x += 1) {
      const index = row + x;
      const angle = ((Math.atan2(gradientRow[index], gradientColumn[index]) * RAD_TO_DEG) % 180 + 180) % 180;
      let bin = Math.floor(angle / binWidth);
      if (bin >= orientations) bin = orientations - 1;
      const magnitude = Math.hypot(gradientColumn[index], gradientRow[index]);
      const cellColumn = Math.floor(x / cell);
      histogram[(cellRow * columns + cellColumn) * orientations + bin] += magnitude;
    }
  }

  const area = cell * cell;
  for (let i = 0; i < histogram.length; i += 1) histogram[i] /= area;
  return { values: histogram, rows, columns };
}

export function normalizeBlocks(cells) {
  const { values, rows, columns } = cells;
  const block = HOG.block;
  const orientations = HOG.orientations;
  const epsilon = HOG.epsilon;
  const blockRows = rows - block + 1;
  const blockColumns = columns - block + 1;
  const output = new Float32Array(blockRows * blockColumns * block * block * orientations);
  const patch = new Float64Array(block * block * orientations);
  let cursor = 0;

  for (let blockRow = 0; blockRow < blockRows; blockRow += 1) {
    for (let blockColumn = 0; blockColumn < blockColumns; blockColumn += 1) {
      let index = 0;
      for (let r = 0; r < block; r += 1) {
        for (let c = 0; c < block; c += 1) {
          const source = ((blockRow + r) * columns + blockColumn + c) * orientations;
          for (let o = 0; o < orientations; o += 1) patch[index + o] = values[source + o];
          index += orientations;
        }
      }
      let sum = epsilon * epsilon;
      for (let i = 0; i < patch.length; i += 1) sum += patch[i] * patch[i];
      let norm = Math.sqrt(sum);
      for (let i = 0; i < patch.length; i += 1) patch[i] = Math.min(patch[i] / norm, 0.2);
      sum = epsilon * epsilon;
      for (let i = 0; i < patch.length; i += 1) sum += patch[i] * patch[i];
      norm = Math.sqrt(sum);
      for (let i = 0; i < patch.length; i += 1) output[cursor + i] = patch[i] / norm;
      cursor += patch.length;
    }
  }
  return output;
}

export function featuresFromImage(rgba, sourceWidth, sourceHeight) {
  const rgb = areaResize(rgba, sourceWidth, sourceHeight, HOG.width, HOG.height);
  const gray = toGray(rgb, HOG.width, HOG.height);
  const cells = extractCells(gray, HOG.width, HOG.height);
  return { features: normalizeBlocks(cells), cells, gray };
}
