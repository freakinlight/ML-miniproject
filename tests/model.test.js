import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  evaluateModel,
  inferTaskType,
  parsePenguinsCsv,
  stratifiedSplit,
  trainKnn,
} from "../model.js";

const datasetUrl = new URL("../data/penguins.csv", import.meta.url);

test("parses and cleans the Palmer Penguins dataset", async () => {
  const rows = parsePenguinsCsv(await readFile(datasetUrl, "utf8"));
  assert.equal(rows.length, 342);
  assert.deepEqual([...new Set(rows.map((row) => row.species))].sort(), ["Adelie", "Chinstrap", "Gentoo"]);
  assert.ok(rows.every((row) => Number.isFinite(row.bodyMass)));
});

test("infers classification and regression task types", () => {
  assert.equal(inferTaskType(["cat", "dog", "cat"]).type, "classification");
  assert.equal(inferTaskType(Array.from({ length: 100 }, (_, index) => index * 0.37)).type, "regression");
  assert.equal(inferTaskType([0, 1, 0, 1, 1, 0]).type, "classification");
});

test("creates a deterministic stratified split", async () => {
  const rows = parsePenguinsCsv(await readFile(datasetUrl, "utf8"));
  const first = stratifiedSplit(rows, 0.8, 42);
  const second = stratifiedSplit(rows, 0.8, 42);
  assert.deepEqual(first.train.map((row) => row.id), second.train.map((row) => row.id));
  assert.equal(first.train.length + first.test.length, rows.length);
  assert.deepEqual([...new Set(first.test.map((row) => row.species))].sort(), ["Adelie", "Chinstrap", "Gentoo"]);
});

test("KNN reaches a credible held-out score", async () => {
  const rows = parsePenguinsCsv(await readFile(datasetUrl, "utf8"));
  const { train, test: testRows } = stratifiedSplit(rows, 0.8, 42);
  const model = trainKnn(train);
  const evaluation = evaluateModel(model, testRows, 7);
  assert.ok(evaluation.accuracy >= 0.9, `accuracy was ${evaluation.accuracy}`);
  assert.ok(evaluation.macroF1 >= 0.88, `macro F1 was ${evaluation.macroF1}`);
});

test("returns probabilities and the requested number of neighbours", async () => {
  const rows = parsePenguinsCsv(await readFile(datasetUrl, "utf8"));
  const { train } = stratifiedSplit(rows, 0.8, 42);
  const prediction = trainKnn(train).predict({
    billLength: 39.1,
    billDepth: 18.7,
    flipperLength: 181,
    bodyMass: 3750,
  }, 5);
  assert.equal(prediction.label, "Adelie");
  assert.equal(prediction.neighbours.length, 5);
  const probabilityTotal = Object.values(prediction.probabilities).reduce((sum, value) => sum + value, 0);
  assert.ok(Math.abs(probabilityTotal - 1) < 1e-9);
});
