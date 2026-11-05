import { HOG, featuresFromImage } from "./hog.js";
import { loadModel, predict } from "./model.js";

const source = document.getElementById("source");
const sourceMeta = document.getElementById("source-meta");
const chooser = document.getElementById("chooser");
const placeholder = document.querySelector(".placeholder");
const pick = document.getElementById("pick");
const fileInput = document.getElementById("file");
const gradient = document.getElementById("gradient");
const bars = Array.from(document.querySelectorAll(".bar"));
const status = document.getElementById("status");
const bench = document.getElementById("bench");
const sampleButtons = Array.from(document.querySelectorAll("[data-sample]"));

const modelPromise = loadModel();
let currentUrl = null;
let job = 0;

function setStatus(message) {
  status.textContent = message;
}

function isImage(file) {
  return Boolean(file) && file.type.startsWith("image/");
}

async function toPixels(file) {
  const bitmap = await createImageBitmap(file, {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(bitmap, 0, 0);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  bitmap.close();
  return { pixels, width: canvas.width, height: canvas.height };
}

function drawGradient(cells) {
  const context = gradient.getContext("2d");
  const scaleX = gradient.width / HOG.width;
  const scaleY = gradient.height / HOG.height;
  context.fillStyle = "#141610";
  context.fillRect(0, 0, gradient.width, gradient.height);
  context.save();
  context.scale(scaleX, scaleY);
  context.strokeStyle = "#e8e8de";
  context.lineWidth = 0.9;

  let peak = 0;
  for (const value of cells.values) if (value > peak) peak = value;
  if (peak <= 0) peak = 1;

  const radius = HOG.cell / 2 - 1;
  for (let row = 0; row < cells.rows; row += 1) {
    for (let column = 0; column < cells.columns; column += 1) {
      const centerY = row * HOG.cell + HOG.cell / 2;
      const centerX = column * HOG.cell + HOG.cell / 2;
      for (let orientation = 0; orientation < HOG.orientations; orientation += 1) {
        const index = (row * cells.columns + column) * HOG.orientations + orientation;
        const value = cells.values[index] / peak;
        if (value < 0.045) continue;
        const angle = (Math.PI * (orientation + 0.5)) / HOG.orientations;
        const dr = radius * Math.sin(angle);
        const dc = radius * Math.cos(angle);
        context.globalAlpha = 0.12 + 0.88 * value;
        context.beginPath();
        context.moveTo(centerX + dr, centerY - dc);
        context.lineTo(centerX - dr, centerY + dc);
        context.stroke();
      }
    }
  }
  context.restore();
  context.globalAlpha = 1;
}

function render(result, width, height, elapsed) {
  bars.forEach((bar, index) => {
    const probability = result.probabilities[index];
    bar.classList.toggle("is-winner", index === result.index);
    bar.querySelector(".fill").style.width = `${Math.min(1, probability) * 100}%`;
    bar.querySelector(".value").textContent = probability.toFixed(3);
  });
  sourceMeta.textContent = `${width} × ${height}`;
  const confidence = result.probabilities[result.index];
  setStatus(`${result.label} · ${(confidence * 100).toFixed(1)}% confidence · ${Math.round(elapsed)} ms`);
}

async function classify(file) {
  if (!isImage(file)) {
    setStatus("That file is not an image. Choose a PNG, JPEG, or WebP.");
    return;
  }
  const ticket = (job += 1);
  setStatus("Reading the screenshot…");

  try {
    const { pixels, width, height } = await toPixels(file);
    const model = await modelPromise;
    const start = performance.now();
    const { features, cells } = featuresFromImage(pixels, width, height);
    const result = predict(model, features);
    const elapsed = performance.now() - start;
    if (ticket !== job) return;

    if (currentUrl) URL.revokeObjectURL(currentUrl);
    currentUrl = URL.createObjectURL(file);
    source.src = currentUrl;
    source.hidden = false;
    chooser.hidden = true;
    placeholder.hidden = true;
    drawGradient(cells);
    render(result, width, height, elapsed);
  } catch (error) {
    console.error(error);
    if (ticket === job) setStatus("That image could not be read. Try a PNG or JPEG.");
  }
}

function openPicker() {
  fileInput.click();
}

pick.addEventListener("click", openPicker);
chooser.addEventListener("click", openPicker);

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) classify(file);
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((type) => {
  bench.addEventListener(type, (event) => {
    event.preventDefault();
    bench.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((type) => {
  bench.addEventListener(type, (event) => {
    event.preventDefault();
    if (type === "dragleave" && bench.contains(event.relatedTarget)) return;
    bench.classList.remove("is-dragging");
  });
});

bench.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) classify(file);
});

window.addEventListener("paste", (event) => {
  const item = Array.from(event.clipboardData?.items ?? []).find((entry) => entry.type.startsWith("image/"));
  if (!item) return;
  const file = item.getAsFile();
  if (file) {
    event.preventDefault();
    classify(file);
  }
});

sampleButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    const name = button.dataset.sample;
    setStatus("Loading the sample…");
    try {
      const response = await fetch(`samples/${name}.png`);
      if (!response.ok) throw new Error(`${response.status}`);
      const blob = await response.blob();
      classify(new File([blob], `${name}.png`, { type: blob.type || "image/png" }));
    } catch (error) {
      console.error(error);
      setStatus("The sample could not be loaded.");
    }
  });
});

modelPromise.catch(() => setStatus("The model could not be loaded. Reload the page."));

const requested = new URLSearchParams(window.location.search).get("sample");
if (requested) sampleButtons.find((button) => button.dataset.sample === requested)?.click();
