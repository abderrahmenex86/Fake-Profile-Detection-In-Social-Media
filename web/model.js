export async function loadModel(base = "./assets") {
  const response = await fetch(`${base}/model.json`);
  if (!response.ok) throw new Error(`model.json ${response.status}`);
  const model = await response.json();
  const scalerResponse = await fetch(`${base}/${model.scaler.file}`);
  if (!scalerResponse.ok) throw new Error(`${model.scaler.file} ${scalerResponse.status}`);
  const buffer = await scalerResponse.arrayBuffer();
  const scaler = new Float32Array(buffer);
  const features = model.scaler.features;
  return {
    classes: model.classes,
    trees: model.trees,
    offsets: model.offsets,
    mean: scaler.subarray(0, features),
    scale: scaler.subarray(features, features * 2),
    feature: model.feature,
    metrics: model.metrics,
    size: model.size,
  };
}

export function standardize(features, mean, scale) {
  const output = new Float32Array(features.length);
  for (let i = 0; i < features.length; i += 1) {
    output[i] = (features[i] - mean[i]) / scale[i];
  }
  return output;
}

function traverse(nodes, row) {
  let node = 0;
  for (;;) {
    const [feature, threshold, yes, no] = nodes[node];
    if (feature < 0) return threshold;
    node = row[feature] < threshold ? yes : no;
  }
}

export function predict(model, features) {
  const row = standardize(features, model.mean, model.scale);
  const margins = new Float64Array(model.classes.length);
  for (let index = 0; index < model.trees.length; index += 1) {
    margins[index % margins.length] += traverse(model.trees[index], row);
  }
  const scores = new Float64Array(margins.length);
  for (let i = 0; i < margins.length; i += 1) scores[i] = margins[i] + model.offsets[i];
  const peak = Math.max(...scores);
  let total = 0;
  for (let i = 0; i < scores.length; i += 1) {
    scores[i] = Math.exp(scores[i] - peak);
    total += scores[i];
  }
  for (let i = 0; i < scores.length; i += 1) scores[i] /= total;
  let winner = 0;
  for (let i = 1; i < scores.length; i += 1) if (scores[i] > scores[winner]) winner = i;
  return { label: model.classes[winner], index: winner, probabilities: Array.from(scores) };
}
