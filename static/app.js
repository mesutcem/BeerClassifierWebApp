const form = document.querySelector("#prediction-form");
const result = document.querySelector("#result");
const errorMessage = document.querySelector("#error-message");
const submitButton = form.querySelector("button");

function showProbabilities(probabilities) {
  const list = document.querySelector("#probability-list");
  list.replaceChildren();

  Object.entries(probabilities).forEach(([name, probability]) => {
    const percentage = probability * 100;
    const row = document.createElement("div");
    row.className = "probability-row";

    const label = document.createElement("span");
    label.textContent = name;

    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("span");
    fill.style.width = `${percentage}%`;
    bar.append(fill);

    const value = document.createElement("output");
    value.textContent = `${percentage.toFixed(1)}%`;

    row.append(label, bar, value);
    list.append(row);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.hidden = true;
  result.hidden = true;
  submitButton.disabled = true;
  submitButton.firstElementChild.textContent = "Hesaplanıyor...";

  const data = new FormData(form);
  const payload = Object.fromEntries(
    ["og", "abv", "ph", "ibu"].map((name) => [name, Number(data.get(name))]),
  );

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Tahmin alınamadı.");

    document.querySelector("#class-name").textContent = body.class_name;
    document.querySelector("#confidence-value").textContent =
      `${(body.confidence * 100).toFixed(1)}%`;
    showProbabilities(body.probabilities);
    result.hidden = false;
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.hidden = false;
  } finally {
    submitButton.disabled = false;
    submitButton.firstElementChild.textContent = "Tahmin et";
  }
});
