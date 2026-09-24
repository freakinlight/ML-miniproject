import {
  FEATURES,
  evaluateModel,
  inferTaskType,
  parsePenguinsCsv,
  stratifiedSplit,
  trainKnn,
} from "./model.js";

const SPECIES_META = {
  Adelie: {
    color: "#ff765f",
    className: "adelie-result",
    description: "A compact penguin with a shorter bill and a classic black-and-white profile.",
  },
  Chinstrap: {
    color: "#ffc95c",
    className: "chinstrap-result",
    description: "Often recognized by the narrow dark line beneath its chin and a longer bill.",
  },
  Gentoo: {
    color: "#57d2c4",
    className: "gentoo-result",
    description: "Typically the heaviest of the three, with especially long flippers.",
  },
};

const inputs = {
  billLength: document.querySelector("#bill-length"),
  billDepth: document.querySelector("#bill-depth"),
  flipperLength: document.querySelector("#flipper-length"),
  bodyMass: document.querySelector("#body-mass"),
};

const outputs = {
  billLength: document.querySelector("#bill-length-output"),
  billDepth: document.querySelector("#bill-depth-output"),
  flipperLength: document.querySelector("#flipper-length-output"),
  bodyMass: document.querySelector("#body-mass-output"),
};

const state = {
  rows: [],
  trainRows: [],
  testRows: [],
  model: null,
  evaluation: null,
  prediction: null,
  sample: null,
  k: 7,
};

function formatMeasurement(key, value) {
  if (key === "bodyMass") return `${Number(value).toLocaleString()} g`;
  return `${value} mm`;
}

function updateRange(input, key) {
  const minimum = Number(input.min);
  const maximum = Number(input.max);
  const position = ((Number(input.value) - minimum) / (maximum - minimum)) * 100;
  input.style.setProperty("--range-position", `${position}%`);
  outputs[key].textContent = formatMeasurement(key, input.value);
}

function currentSample() {
  return Object.fromEntries(
    FEATURES.map(({ key }) => [key, Number(inputs[key].value)]),
  );
}

function percent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function renderSummary() {
  document.querySelector("#dataset-size").textContent = state.rows.length;
  document.querySelector("#accuracy").textContent = percent(state.evaluation.accuracy);
  document.querySelector("#macro-f1").textContent = state.evaluation.macroF1.toFixed(3);
  document.querySelector("#test-count").textContent = `${state.evaluation.testSize} test rows`;
  document.querySelector("#header-status").textContent = `Model ready · ${state.trainRows.length} training rows`;

  const task = inferTaskType(state.rows.map((row) => row.species));
  document.querySelector("#task-reason").textContent = task.reason;
  renderConfusionMatrix();
}

function renderConfusionMatrix() {
  const { labels, confusion } = state.evaluation;
  const table = document.createElement("table");
  table.className = "confusion-table";
  table.innerHTML = `
    <caption class="sr-only">Confusion matrix for the held-out test set</caption>
    <thead>
      <tr><th scope="col">Actual ↓</th>${labels.map((label) => `<th scope="col">${label.slice(0, 3)}</th>`).join("")}</tr>
    </thead>
    <tbody>
      ${labels.map((actual) => `
        <tr>
          <th scope="row">${actual}</th>
          ${labels.map((predicted) => {
            const correct = actual === predicted ? " correct" : "";
            return `<td class="${correct}">${confusion[actual][predicted]}</td>`;
          }).join("")}
        </tr>
      `).join("")}
    </tbody>
  `;
  const container = document.querySelector("#confusion-matrix");
  container.replaceChildren(table);
}

function renderProbabilities(prediction) {
  const container = document.querySelector("#probability-list");
  container.hidden = false;
  container.innerHTML = Object.entries(prediction.probabilities)
    .sort(([, first], [, second]) => second - first)
    .map(([species, probability]) => `
      <div class="probability-row">
        <div><span>${species}</span><strong>${percent(probability, 0)}</strong></div>
        <div class="probability-track"><i style="width:${probability * 100}%; background:${SPECIES_META[species].color}"></i></div>
      </div>
    `).join("");
}

function renderNeighbours(prediction) {
  const container = document.querySelector("#neighbour-list");
  container.innerHTML = prediction.neighbours.map((neighbour, index) => `
    <article class="neighbour-row">
      <span class="neighbour-rank">${String(index + 1).padStart(2, "0")}</span>
      <i style="background:${SPECIES_META[neighbour.species].color}"></i>
      <div>
        <strong>${neighbour.species}</strong>
        <small>${neighbour.island} Island</small>
      </div>
      <span class="distance">d = ${neighbour.distance.toFixed(2)}</span>
    </article>
  `).join("");
}

function renderPrediction() {
  const prediction = state.prediction;
  const meta = SPECIES_META[prediction.label];
  const panel = document.querySelector("#prediction-card");
  panel.classList.remove(...Object.values(SPECIES_META).map((item) => item.className));
  panel.classList.add(meta.className, "has-result");

  document.querySelector("#result-avatar").textContent = prediction.label.charAt(0);
  document.querySelector("#result-kicker").textContent = "Most likely species";
  document.querySelector("#result-species").textContent = prediction.label;
  document.querySelector("#result-description").textContent = meta.description;
  const note = document.querySelector("#confidence-note");
  note.hidden = false;
  note.innerHTML = `<strong>${percent(prediction.confidence, 0)} neighbour support</strong><span>Distance-weighted vote using k = ${state.k}</span>`;

  renderProbabilities(prediction);
  renderNeighbours(prediction);
  drawScatterplot();
}

function drawScatterplot() {
  const canvas = document.querySelector("#scatterplot");
  const wrapper = canvas.parentElement;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(wrapper.clientWidth, 320);
  const height = Math.min(Math.max(width * 0.56, 280), 420);
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);

  const padding = { top: 22, right: 22, bottom: 48, left: 58 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const xMin = 30;
  const xMax = 62;
  const yMin = 165;
  const yMax = 240;
  const xScale = (value) => padding.left + ((value - xMin) / (xMax - xMin)) * plotWidth;
  const yScale = (value) => padding.top + plotHeight - ((value - yMin) / (yMax - yMin)) * plotHeight;

  context.font = "12px ui-sans-serif, system-ui, sans-serif";
  context.strokeStyle = "rgba(9, 27, 43, 0.10)";
  context.fillStyle = "#617282";
  context.lineWidth = 1;

  [170, 185, 200, 215, 230].forEach((tick) => {
    const y = yScale(tick);
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    context.fillText(String(tick), 20, y + 4);
  });

  [35, 40, 45, 50, 55, 60].forEach((tick) => {
    const x = xScale(tick);
    context.fillText(String(tick), x - 7, height - 24);
  });

  context.save();
  context.translate(14, padding.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("Flipper length (mm)", -54, 0);
  context.restore();
  context.fillText("Bill length (mm)", padding.left + plotWidth / 2 - 44, height - 5);

  state.rows.forEach((row) => {
    context.beginPath();
    context.arc(xScale(row.billLength), yScale(row.flipperLength), 3.25, 0, Math.PI * 2);
    context.fillStyle = `${SPECIES_META[row.species].color}a8`;
    context.fill();
  });

  if (state.sample) {
    const x = xScale(state.sample.billLength);
    const y = yScale(state.sample.flipperLength);
    context.beginPath();
    context.arc(x, y, 9, 0, Math.PI * 2);
    context.fillStyle = "#ffffff";
    context.fill();
    context.strokeStyle = "#091b2b";
    context.lineWidth = 3;
    context.stroke();
    context.beginPath();
    context.arc(x, y, 3, 0, Math.PI * 2);
    context.fillStyle = SPECIES_META[state.prediction.label].color;
    context.fill();
  }
}

function runPrediction() {
  state.sample = currentSample();
  state.prediction = state.model.predict(state.sample, state.k);
  renderPrediction();
}

async function initialize() {
  try {
    const response = await fetch("./data/penguins.csv");
    if (!response.ok) throw new Error(`Dataset request failed (${response.status})`);
    state.rows = parsePenguinsCsv(await response.text());
    const split = stratifiedSplit(state.rows, 0.8, 42);
    state.trainRows = split.train;
    state.testRows = split.test;
    state.model = trainKnn(state.trainRows);
    state.evaluation = evaluateModel(state.model, state.testRows, state.k);
    renderSummary();
    drawScatterplot();
    document.body.classList.add("is-ready");
  } catch (error) {
    console.error(error);
    document.querySelector("#header-status").textContent = "Model could not load";
    document.querySelector("#prediction-card").innerHTML = `
      <div class="load-error"><strong>Unable to load the model.</strong><p>${error.message}</p></div>
    `;
  }
}

Object.entries(inputs).forEach(([key, input]) => {
  updateRange(input, key);
  input.addEventListener("input", () => updateRange(input, key));
});

document.querySelector("#prediction-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.model) runPrediction();
});

document.querySelector("#k-value").addEventListener("change", (event) => {
  state.k = Number(event.target.value);
  document.querySelector("#hero-k").textContent = state.k;
  if (state.model) {
    state.evaluation = evaluateModel(state.model, state.testRows, state.k);
    renderSummary();
    if (state.sample) runPrediction();
  }
});

let resizeTimer;
window.addEventListener("resize", () => {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    if (state.rows.length) drawScatterplot();
  }, 120);
});

initialize();
