import { readFileSync } from "node:fs";
import { join } from "node:path";

import { featuresFromImage } from "../web/hog.js";
import { predict } from "../web/model.js";

const directory = process.argv[2] ?? "/tmp/opencode/web-validation";
const assets = process.argv[3] ?? "web/assets";

function readFloat32(path, count) {
  const buffer = readFileSync(path);
  const length = count ?? buffer.length / 4;
  const values = new Float32Array(length);
  for (let i = 0; i < length; i += 1) values[i] = buffer.readFloatLE(i * 4);
  return values;
}

const manifest = JSON.parse(readFileSync(join(directory, "manifest.json"), "utf8"));
const images = readFileSync(join(directory, "images.bin"));
const referenceFeatures = readFloat32(join(directory, "features.bin"));
const referenceGray = readFileSync(join(directory, "gray.bin"));

const modelJson = JSON.parse(readFileSync(join(assets, "model.json"), "utf8"));
const scaler = readFloat32(join(assets, modelJson.scaler.file));
const featureCount = modelJson.scaler.features;
const model = {
  classes: modelJson.classes,
  trees: modelJson.trees,
  offsets: modelJson.offsets,
  mean: scaler.subarray(0, featureCount),
  scale: scaler.subarray(featureCount, featureCount * 2),
};

const featureSize = 34992;
const graySize = 450 * 600;
let featureMax = 0;
let featureMean = 0;
let probabilityMax = 0;
let grayMismatch = 0;
let grayMax = 0;
let agreed = 0;
let elapsed = 0;

for (let index = 0; index < manifest.entries.length; index += 1) {
  const entry = manifest.entries[index];
  const rgba = new Uint8Array(images.buffer, images.byteOffset + entry.offset, entry.length);
  const start = performance.now();
  const { features, gray } = featuresFromImage(rgba, manifest.width, manifest.height);
  const result = predict(model, features);
  elapsed += performance.now() - start;

  const reference = referenceFeatures.subarray(index * featureSize, (index + 1) * featureSize);
  for (let i = 0; i < featureSize; i += 1) {
    const diff = Math.abs(features[i] - reference[i]);
    if (diff > featureMax) featureMax = diff;
    featureMean += diff;
  }

  for (let i = 0; i < graySize; i += 1) {
    const diff = Math.abs(gray[i] - referenceGray[index * graySize + i]);
    if (diff > grayMax) grayMax = diff;
    if (diff > 0) grayMismatch += 1;
  }

  let referenceIndex = 0;
  for (let c = 1; c < entry.probabilities.length; c += 1) {
    if (entry.probabilities[c] > entry.probabilities[referenceIndex]) referenceIndex = c;
  }
  if (result.index === referenceIndex) agreed += 1;
  for (let c = 0; c < model.classes.length; c += 1) {
    probabilityMax = Math.max(probabilityMax, Math.abs(result.probabilities[c] - entry.probabilities[c]));
  }
  if (index % 25 === 0) process.stdout.write(`[js] ${index + 1}/${manifest.entries.length}\n`);
}

const total = manifest.entries.length;
console.log("");
console.log(`images               ${total}`);
console.log(`feature max diff     ${featureMax.toExponential(2)}`);
console.log(`feature mean diff    ${(featureMean / (total * featureSize)).toExponential(2)}`);
console.log(`gray mismatch        ${grayMismatch} pixels, max ${grayMax}`);
console.log(`probability max diff ${probabilityMax.toExponential(2)}`);
console.log(`label agreement      ${agreed}/${total}`);
console.log(`mean latency         ${(elapsed / total).toFixed(1)} ms/image`);
