export const FEATURES = [
  { key: "billLength", csv: "bill_length_mm", label: "Bill length", unit: "mm" },
  { key: "billDepth", csv: "bill_depth_mm", label: "Bill depth", unit: "mm" },
  { key: "flipperLength", csv: "flipper_length_mm", label: "Flipper length", unit: "mm" },
  { key: "bodyMass", csv: "body_mass_g", label: "Body mass", unit: "g" },
];

const SPECIES = ["Adelie", "Chinstrap", "Gentoo"];

function parseCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      values.push(value);
      value = "";
    } else {
      value += character;
    }
  }
  values.push(value);
  return values;
}

export function parsePenguinsCsv(csvText) {
  const lines = csvText.trim().split(/\r?\n/).filter(Boolean);
  const headers = parseCsvLine(lines[0]);

  return lines.slice(1).map((line, rowIndex) => {
    const values = parseCsvLine(line);
    const raw = Object.fromEntries(headers.map((header, index) => [header, values[index]]));
    return {
      id: rowIndex + 1,
      species: raw.species,
      island: raw.island,
      billLength: Number(raw.bill_length_mm),
      billDepth: Number(raw.bill_depth_mm),
      flipperLength: Number(raw.flipper_length_mm),
      bodyMass: Number(raw.body_mass_g),
    };
  }).filter((row) => (
    SPECIES.includes(row.species)
    && FEATURES.every((feature) => Number.isFinite(row[feature.key]))
  ));
}

export function inferTaskType(values) {
  const cleanValues = values.filter((value) => value !== null && value !== undefined && value !== "");
  const numericValues = cleanValues.map(Number);
  const allNumeric = numericValues.every(Number.isFinite);
  const uniqueCount = new Set(cleanValues.map(String)).size;

  if (!allNumeric) {
    return {
      type: "classification",
      reason: `The target contains text labels (${uniqueCount} distinct classes).`,
    };
  }

  const categoryThreshold = Math.max(10, Math.floor(cleanValues.length * 0.05));
  if (uniqueCount <= categoryThreshold) {
    return {
      type: "classification",
      reason: `The numeric target has only ${uniqueCount} distinct values, so it behaves like a category.`,
    };
  }

  return {
    type: "regression",
    reason: `The target is numeric and has ${uniqueCount} distinct values.`,
  };
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let result = state;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled(items, random) {
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [copy[index], copy[swapIndex]] = [copy[swapIndex], copy[index]];
  }
  return copy;
}

export function stratifiedSplit(rows, trainRatio = 0.8, seed = 42) {
  const random = seededRandom(seed);
  const groups = Object.groupBy
    ? Object.groupBy(rows, (row) => row.species)
    : rows.reduce((result, row) => {
      (result[row.species] ??= []).push(row);
      return result;
    }, {});
  const train = [];
  const test = [];

  Object.keys(groups).sort().forEach((species) => {
    const group = shuffled(groups[species], random);
    const splitIndex = Math.max(1, Math.floor(group.length * trainRatio));
    train.push(...group.slice(0, splitIndex));
    test.push(...group.slice(splitIndex));
  });

  return { train: shuffled(train, random), test: shuffled(test, random) };
}

export function trainKnn(trainingRows) {
  if (!trainingRows.length) {
    throw new Error("KNN needs at least one training row.");
  }

  const means = {};
  const standardDeviations = {};

  FEATURES.forEach(({ key }) => {
    means[key] = trainingRows.reduce((sum, row) => sum + row[key], 0) / trainingRows.length;
    const variance = trainingRows.reduce((sum, row) => sum + ((row[key] - means[key]) ** 2), 0)
      / trainingRows.length;
    standardDeviations[key] = Math.sqrt(variance) || 1;
  });

  const normalizedTrainingRows = trainingRows.map((row) => ({
    row,
    values: FEATURES.map(({ key }) => (row[key] - means[key]) / standardDeviations[key]),
  }));

  function predict(sample, requestedK = 7) {
    const k = Math.max(1, Math.min(Math.round(requestedK), normalizedTrainingRows.length));
    const normalizedSample = FEATURES.map(({ key }) => (
      (Number(sample[key]) - means[key]) / standardDeviations[key]
    ));

    const neighbours = normalizedTrainingRows.map(({ row, values }) => ({
      row,
      distance: Math.sqrt(values.reduce((sum, value, index) => (
        sum + ((value - normalizedSample[index]) ** 2)
      ), 0)),
    })).sort((first, second) => first.distance - second.distance).slice(0, k);

    const weights = Object.fromEntries(SPECIES.map((species) => [species, 0]));
    neighbours.forEach(({ row, distance }) => {
      weights[row.species] += 1 / Math.max(distance, 0.000001);
    });
    const weightTotal = Object.values(weights).reduce((sum, weight) => sum + weight, 0);
    const probabilities = Object.fromEntries(
      SPECIES.map((species) => [species, weights[species] / weightTotal]),
    );
    const label = SPECIES.reduce((best, species) => (
      probabilities[species] > probabilities[best] ? species : best
    ), SPECIES[0]);

    return {
      label,
      confidence: probabilities[label],
      probabilities,
      neighbours: neighbours.map(({ row, distance }) => ({ ...row, distance })),
    };
  }

  return { predict, means, standardDeviations, trainingRows };
}

export function evaluateModel(model, testRows, k = 7) {
  const labels = [...SPECIES];
  const confusion = Object.fromEntries(
    labels.map((actual) => [actual, Object.fromEntries(labels.map((predicted) => [predicted, 0]))]),
  );

  testRows.forEach((row) => {
    const prediction = model.predict(row, k).label;
    confusion[row.species][prediction] += 1;
  });

  const correct = labels.reduce((sum, label) => sum + confusion[label][label], 0);
  const accuracy = correct / testRows.length;
  const f1ByClass = Object.fromEntries(labels.map((label) => {
    const truePositive = confusion[label][label];
    const falsePositive = labels.reduce((sum, actual) => (
      actual === label ? sum : sum + confusion[actual][label]
    ), 0);
    const falseNegative = labels.reduce((sum, predicted) => (
      predicted === label ? sum : sum + confusion[label][predicted]
    ), 0);
    const precision = truePositive / (truePositive + falsePositive || 1);
    const recall = truePositive / (truePositive + falseNegative || 1);
    const f1 = 2 * precision * recall / (precision + recall || 1);
    return [label, f1];
  }));
  const macroF1 = Object.values(f1ByClass).reduce((sum, value) => sum + value, 0) / labels.length;

  return { labels, confusion, accuracy, macroF1, f1ByClass, testSize: testRows.length };
}
