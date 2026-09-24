# Penguin Neighbourhood

An interactive, browser-based machine-learning mini project that predicts a Palmer penguin's species from four physical measurements.

## What it demonstrates

- Multiclass classification versus regression
- A deterministic, stratified 80/20 train/test split
- Feature standardization
- Distance-weighted K-Nearest Neighbours
- Accuracy, macro F1, and a confusion matrix
- A local explanation using the closest training specimens

The site has no backend and no build step. The dataset and model logic run locally in the visitor's browser, making it compatible with GitHub Pages.

## Run locally

```bash
npm run serve
```

Then open <http://localhost:4173>.

## Test

```bash
npm test
```

## Deploy on GitHub Pages

1. Push this folder to a GitHub repository.
2. Open the repository's **Settings → Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Select the `main` branch and `/ (root)`, then save.

## Data

The project uses the [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) dataset. The data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.
