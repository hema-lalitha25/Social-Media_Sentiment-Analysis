/* ============================================================
   script.js — SentimentAI
   All client-side interactivity for the project.
   No external dependencies beyond Bootstrap 5 + Chart.js
   (both loaded in base.html).
   ============================================================ */

"use strict";

/* ══════════════════════════════════════════════════════════ */
/*  DOM READY                                                  */
/* ══════════════════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", function () {

  initCharCounter();
  initSampleButtons();
  initAnalyzeFormSpinner();
  initDropZone();
  initCsvFormSpinner();
  initConfidenceBars();
  initScrollAnimations();
  initNavbarScroll();

});


/* ══════════════════════════════════════════════════════════ */
/*  1. CHARACTER COUNTER                                       */
/* ══════════════════════════════════════════════════════════ */
function initCharCounter() {
  const textarea  = document.getElementById("textInput");
  const charCount = document.getElementById("charCount");
  if (!textarea || !charCount) return;

  function update() {
    const len = textarea.value.length;
    charCount.textContent = len;

    // Colour feedback
    if (len > 1800) {
      charCount.style.color = "#ef4444";
    } else if (len > 1500) {
      charCount.style.color = "#f59e0b";
    } else {
      charCount.style.color = "#94a3b8";
    }
  }

  textarea.addEventListener("input", update);
  update(); // run on page load (pre-filled text after prediction)
}


/* ══════════════════════════════════════════════════════════ */
/*  2. QUICK-SAMPLE BUTTONS                                    */
/* ══════════════════════════════════════════════════════════ */
function initSampleButtons() {
  const buttons   = document.querySelectorAll(".btn-sample");
  const textarea  = document.getElementById("textInput");
  const charCount = document.getElementById("charCount");
  if (!textarea) return;

  buttons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      const text = btn.getAttribute("data-text");
      if (!text) return;

      textarea.value = text;

      // Update char counter
      if (charCount) charCount.textContent = text.length;

      // Animate the textarea to draw attention
      textarea.classList.add("border-primary");
      textarea.focus();
      setTimeout(function () {
        textarea.classList.remove("border-primary");
      }, 600);
    });
  });
}


/* ══════════════════════════════════════════════════════════ */
/*  3. ANALYZE FORM SPINNER                                    */
/* ══════════════════════════════════════════════════════════ */
function initAnalyzeFormSpinner() {
  const form    = document.getElementById("analyzeForm");
  const btn     = document.getElementById("analyzeBtn");
  if (!form || !btn) return;

  form.addEventListener("submit", function () {
    const textarea = document.getElementById("textInput");
    if (textarea && textarea.value.trim().length === 0) return; // let HTML validation handle

    const textEl    = btn.querySelector(".btn-text");
    const spinnerEl = btn.querySelector(".btn-spinner");

    if (textEl)    textEl.classList.add("d-none");
    if (spinnerEl) spinnerEl.classList.remove("d-none");
    btn.disabled = true;
  });
}


/* ══════════════════════════════════════════════════════════ */
/*  4. DRAG-AND-DROP CSV UPLOAD ZONE                          */
/* ══════════════════════════════════════════════════════════ */
function initDropZone() {
  const zone      = document.getElementById("dropZone");
  const input     = document.getElementById("csvFileInput");
  const content   = document.getElementById("dropZoneContent");
  const preview   = document.getElementById("dropZonePreview");
  const nameEl    = document.getElementById("fileName");
  const sizeEl    = document.getElementById("fileSize");
  const removeBtn = document.getElementById("removeFile");

  if (!zone || !input) return;

  // ── Drag events ──────────────────────────────────────────
  ["dragenter", "dragover"].forEach(function (evt) {
    zone.addEventListener(evt, function (e) {
      e.preventDefault();
      zone.classList.add("drag-over");
    });
  });

  ["dragleave", "dragend", "drop"].forEach(function (evt) {
    zone.addEventListener(evt, function (e) {
      e.preventDefault();
      zone.classList.remove("drag-over");
    });
  });

  zone.addEventListener("drop", function (e) {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      setFile(files[0]);
    }
  });

  // ── File input change ────────────────────────────────────
  input.addEventListener("change", function () {
    if (input.files.length > 0) {
      setFile(input.files[0]);
    }
  });

  // ── Remove file ──────────────────────────────────────────
  if (removeBtn) {
    removeBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      e.preventDefault();
      clearFile();
    });
  }

  // ── Helpers ──────────────────────────────────────────────
  function setFile(file) {
    // Validate extension
    if (!file.name.toLowerCase().endsWith(".csv")) {
      showZoneError("Only .csv files are accepted.");
      return;
    }

    if (nameEl)  nameEl.textContent  = file.name;
    if (sizeEl)  sizeEl.textContent  = formatBytes(file.size);

    if (content) content.classList.add("d-none");
    if (preview) preview.classList.remove("d-none");

    // Set file on input using DataTransfer (Chrome/Firefox/Edge)
    // Safari fallback: store file reference for FormData injection
    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    } catch (err) {
      // Safari: attach file directly to the form via submit interception
      input._safariFile = file;
    }
  }

  function clearFile() {
    input.value = "";
    if (content) content.classList.remove("d-none");
    if (preview) preview.classList.add("d-none");
    if (nameEl)  nameEl.textContent = "No file";
    if (sizeEl)  sizeEl.textContent = "";
    zone.classList.remove("drag-over");
  }

  function showZoneError(msg) {
    zone.style.borderColor = "#ef4444";
    zone.style.background  = "rgba(239,68,68,0.07)";
    const title = zone.querySelector(".drop-title");
    if (title) {
      const orig = title.textContent;
      title.textContent = msg;
      title.style.color = "#fca5a5";
      setTimeout(function () {
        title.textContent = orig;
        title.style.color = "";
        zone.style.borderColor = "";
        zone.style.background  = "";
      }, 2500);
    }
  }

  function formatBytes(bytes) {
    if (bytes < 1024)       return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }
}


/* ══════════════════════════════════════════════════════════ */
/*  5. CSV UPLOAD FORM SPINNER                                 */
/* ══════════════════════════════════════════════════════════ */
function initCsvFormSpinner() {
  const form      = document.getElementById("csvForm");
  const uploadBtn = document.getElementById("uploadBtn");
  if (!form || !uploadBtn) return;

  form.addEventListener("submit", function (e) {
    const input = document.getElementById("csvFileInput");

    // Safari fallback: if DataTransfer failed, use FormData with stored file
    if (input && input._safariFile) {
      e.preventDefault();
      const fd = new FormData(form);
      fd.set("csv_file", input._safariFile, input._safariFile.name);

      const textEl    = uploadBtn.querySelector(".btn-text");
      const spinnerEl = uploadBtn.querySelector(".btn-spinner");
      if (textEl)    textEl.classList.add("d-none");
      if (spinnerEl) spinnerEl.classList.remove("d-none");
      uploadBtn.disabled = true;

      fetch(form.action, { method: "POST", body: fd })
        .then(function (res) { window.location.href = res.url; })
        .catch(function ()   { window.location.reload(); });
      return;
    }

    if (!input || !input.files || input.files.length === 0) return;

    const textEl    = uploadBtn.querySelector(".btn-text");
    const spinnerEl = uploadBtn.querySelector(".btn-spinner");
    if (textEl)    textEl.classList.add("d-none");
    if (spinnerEl) spinnerEl.classList.remove("d-none");
    uploadBtn.disabled = true;
  });
}


/* ══════════════════════════════════════════════════════════ */
/*  6. CONFIDENCE BAR ANIMATION (result card)                 */
/* ══════════════════════════════════════════════════════════ */
function initConfidenceBars() {
  const fills = document.querySelectorAll(".confidence-fill[data-width]");

  fills.forEach(function (bar) {
    const target = parseFloat(bar.getAttribute("data-width")) || 0;
    // Start at 0, animate to target after brief delay
    bar.style.width = "0%";
    requestAnimationFrame(function () {
      setTimeout(function () {
        bar.style.width = target + "%";
      }, 120);
    });
  });

  // Also animate hero preview bars
  const heroBars = document.querySelectorAll(".conf-bar-animated");
  heroBars.forEach(function (bar) {
    const target = bar.style.width;
    bar.style.width = "0%";
    requestAnimationFrame(function () {
      setTimeout(function () {
        bar.style.width = target;
      }, 300);
    });
  });
}


/* ══════════════════════════════════════════════════════════ */
/*  7. SCROLL-TRIGGERED FADE-IN ANIMATIONS                    */
/* ══════════════════════════════════════════════════════════ */
function initScrollAnimations() {
  // Apply fade-in-up to cards when they enter the viewport
  const targets = document.querySelectorAll(
    ".glass-card, .how-card, .stat-card, .section-header"
  );

  if (!("IntersectionObserver" in window)) {
    // Fallback: just show everything
    targets.forEach(function (el) { el.style.opacity = 1; });
    return;
  }

  targets.forEach(function (el) {
    el.style.opacity = "0";
    el.style.transform = "translateY(18px)";
  });

  const observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.style.transition =
            "opacity 0.55s ease, transform 0.55s ease";
          entry.target.style.opacity   = "1";
          entry.target.style.transform = "translateY(0)";
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  targets.forEach(function (el) { observer.observe(el); });
}


/* ══════════════════════════════════════════════════════════ */
/*  8. NAVBAR SCROLL EFFECT                                    */
/* ══════════════════════════════════════════════════════════ */
function initNavbarScroll() {
  const navbar = document.querySelector(".glass-nav");
  if (!navbar) return;

  window.addEventListener("scroll", function () {
    if (window.scrollY > 40) {
      navbar.style.background    = "rgba(15,23,42,0.92)";
      navbar.style.boxShadow     = "0 4px 24px rgba(0,0,0,0.4)";
      navbar.style.backdropFilter = "blur(24px)";
    } else {
      navbar.style.background    = "rgba(15,23,42,0.72)";
      navbar.style.boxShadow     = "none";
      navbar.style.backdropFilter = "blur(20px)";
    }
  }, { passive: true });
}


/* ══════════════════════════════════════════════════════════ */
/*  AUTO-DISMISS FLASH ALERTS (after 5 s)                     */
/* ══════════════════════════════════════════════════════════ */
(function autoDismissAlerts() {
  setTimeout(function () {
    document.querySelectorAll(".flash-alert").forEach(function (alert) {
      // Use Bootstrap's dismiss API if available
      const bsAlert = window.bootstrap && bootstrap.Alert.getInstance(alert);
      if (bsAlert) {
        bsAlert.close();
      } else {
        alert.style.transition = "opacity 0.5s ease";
        alert.style.opacity    = "0";
        setTimeout(function () { alert.remove(); }, 500);
      }
    });
  }, 5000);
})();


/* ══════════════════════════════════════════════════════════ */
/*  TABLE ROW ANIMATION DELAYS (applied via CSS nth-child     */
/*  but JS ensures they run correctly after DOM paint)        */
/* ══════════════════════════════════════════════════════════ */
(function staggerTableRows() {
  const rows = document.querySelectorAll(".table-row-animate");
  rows.forEach(function (row, i) {
    row.style.animationDelay = (i * 0.06) + "s";
  });
})();
