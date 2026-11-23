// Light / dark / auto switch of the studio page. "auto" follows the system setting.
const KEY = "sas-studio-theme";
const root = document.documentElement;

function current() {
  return root.dataset.theme || "auto";
}

function apply(choice) {
  if (choice === "auto") delete root.dataset.theme;
  else root.dataset.theme = choice;
  try {
    if (choice === "auto") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, choice);
  } catch {
    // Storage blocked (private window): the choice still applies until reload.
  }
  for (const button of document.querySelectorAll("[data-theme-choice]")) {
    button.setAttribute("aria-pressed", String(button.dataset.themeChoice === choice));
  }
}

for (const button of document.querySelectorAll("[data-theme-choice]")) {
  button.addEventListener("click", () => apply(button.dataset.themeChoice));
}
apply(current());
