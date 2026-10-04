/* ==========================================================================
   Fake News Detection System - front-end behaviour
   Vanilla JavaScript, no framework and no build step.

   Two jobs only:
     1. live character counters on the form
     2. animate the confidence bar on the result page

   Everything else is server-rendered. The page works with JavaScript disabled;
   only the counters and the bar animation are skipped.
   ========================================================================== */

(function () {
  "use strict";

  /* ------------------------------------------------------------------
     Character counters
     ------------------------------------------------------------------ */
  function initCounters() {
    var counters = document.querySelectorAll("[data-counter-for]");

    Array.prototype.forEach.call(counters, function (counter) {
      var field = document.getElementById(counter.getAttribute("data-counter-for"));
      if (!field) return;

      // Use a limit if the field declares one, otherwise just count characters.
      var limit = parseInt(field.getAttribute("maxlength"), 10);
      var isArticle = field.tagName === "TEXTAREA";

      function update() {
        var length = field.value.length;
        if (isValidLimit(limit)) {
          counter.textContent = length + " / " + limit;
          counter.style.color = length > limit * 0.9 ? "var(--fake)" : "";
        } else {
          counter.textContent = length + (length === 1 ? " character" : " characters");
          counter.style.color = "";
        }
      }

      field.addEventListener("input", update);
      update();
    });
  }

  function isValidLimit(value) {
    return !isNaN(value) && value > 0;
  }

  /* ------------------------------------------------------------------
     Submit state - stop double submission and show progress
     ------------------------------------------------------------------ */
  function initForm() {
    var form = document.getElementById("detect-form");
    if (!form) return;

    var button = document.getElementById("submit-btn");
    var label = button ? button.querySelector(".btn-label") : null;

    form.addEventListener("submit", function () {
      if (!button) return;
      // Browsers with slow round-trips would otherwise allow repeated clicks,
      // which means running the same prediction several times.
      button.disabled = true;
      button.classList.add("is-loading");
      if (label) label.textContent = "Analysing";
    });
  }

  /* ------------------------------------------------------------------
     Confidence bar animation
     The width is set from the value the server rendered, so the number and
     the bar can never disagree.
     ------------------------------------------------------------------ */
  function initConfidenceBar() {
    var fill = document.querySelector(".confidence-fill");
    if (!fill) return;

    var value = parseFloat(fill.getAttribute("data-confidence"));
    if (isNaN(value)) return;

    var clamped = Math.max(0, Math.min(100, value));

    // Start from zero, then let the CSS transition animate to the real value.
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        fill.style.width = clamped + "%";
      });
    });
  }

  /* ------------------------------------------------------------------
     Boot
     ------------------------------------------------------------------ */
  function init() {
    initCounters();
    initForm();
    initConfidenceBar();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();