// ============================================================
// АВТОМАТИЧЕСКАЯ КАТЕГОРИЗАЦИЯ ОБРАЩЕНИЙ — клиентская логика
// ============================================================

const EXTERNAL_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
  <polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>
</svg>`;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initClassifier();
  initPresentation();
  initGlossary();
});

/* ============================================================
   ВКЛАДКИ
   ============================================================ */
function initTabs(){
  const buttons = document.querySelectorAll(".tab-btn");
  const views    = document.querySelectorAll(".view");

  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      buttons.forEach(b => b.classList.toggle("is-active", b === btn));
      views.forEach(v => v.classList.toggle("is-active", v.id === `view-${target}`));

      if (target === "presentation") window.dispatchEvent(new Event("deck:shown"));
    });
  });
}

/* ============================================================
   КЛАССИФИКАТОР
   ============================================================ */
function initClassifier(){
  const textarea   = document.getElementById("ticket-text");
  const charCount   = document.getElementById("char-count");
  const classifyBtn = document.getElementById("classify-btn");
  const examplesBox = document.getElementById("examples-box");
  const resultBox    = document.getElementById("result");
  const placeholder  = document.getElementById("result-placeholder");

  textarea.addEventListener("input", () => {
    charCount.textContent = `${textarea.value.length}/1000`;
  });

  // подгружаем быстрые примеры
  fetch("/api/examples")
    .then(r => r.json())
    .then(examples => {
      examplesBox.innerHTML = "";
      Object.entries(examples).forEach(([key, text]) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "example-chip";
        chip.textContent = key === "ambiguous" ? "Неясное обращение" : text;
        chip.addEventListener("click", () => {
          textarea.value = text;
          charCount.textContent = `${text.length}/1000`;
          textarea.focus();
        });
        examplesBox.appendChild(chip);
      });
    })
    .catch(() => { examplesBox.innerHTML = "<span class='loading-row'>Не удалось загрузить примеры</span>"; });

  classifyBtn.addEventListener("click", () => classify(textarea, resultBox, placeholder, classifyBtn));
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) classify(textarea, resultBox, placeholder, classifyBtn);
  });

  refreshStats();
}

function classify(textarea, resultBox, placeholder, classifyBtn){
  const text = textarea.value.trim();
  if (text.length < 3) {
    textarea.focus();
    return;
  }

  classifyBtn.disabled = true;
  classifyBtn.textContent = "Классифицируем…";

  fetch("/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
    .then(r => r.json())
    .then(data => {
      if (data.error) { renderError(resultBox, placeholder, data.error); return; }
      renderResult(resultBox, placeholder, data);
      refreshStats();
    })
    .catch(() => renderError(resultBox, placeholder, "Сервис недоступен. Попробуйте ещё раз."))
    .finally(() => {
      classifyBtn.disabled = false;
      classifyBtn.textContent = "Классифицировать обращение";
    });
}

function renderError(resultBox, placeholder, message){
  placeholder.style.display = "flex";
  resultBox.classList.remove("is-visible");
  placeholder.innerHTML = `<div class="result-placeholder__icon">⚠️</div><div>${escapeHtml(message)}</div>`;
}

function renderResult(resultBox, placeholder, data){
  placeholder.style.display = "none";
  resultBox.classList.add("is-visible");

  resultBox.innerHTML = `
    <div class="result-head">
      <div class="result-head__icon" style="background:${data.color}22;color:${data.color};">${data.icon}</div>
      <div>
        <div class="result-head__name">${escapeHtml(data.category_name)}</div>
        <div class="result-head__conf">уверенность ${data.confidence_pct}% · ${data.timestamp}</div>
      </div>
    </div>
    ${data.needs_human
      ? `<div class="human-flag">⚠ Низкая уверенность — обращение передано оператору</div>`
      : ""}
    <div class="prob-bars">
      ${data.all_probabilities.map(p => `
        <div class="prob-row">
          <div class="prob-row__icon">${p.icon}</div>
          <div class="prob-row__track"><div class="prob-row__fill" data-width="${p.percentage}" style="background:${p.color};"></div></div>
          <div class="prob-row__pct">${p.percentage}%</div>
          <div class="prob-row__label">${escapeHtml(p.category_name)}</div>
        </div>
      `).join("")}
    </div>
  `;

  // анимация заполнения шкал
  requestAnimationFrame(() => {
    resultBox.querySelectorAll(".prob-row__fill").forEach(el => {
      el.style.width = `${el.dataset.width}%`;
    });
  });
}

function refreshStats(){
  fetch("/api/stats")
    .then(r => r.json())
    .then(s => {
      setStat("stat-total", s.total_requests);
      setStat("stat-auto", s.total_requests - s.redirected_to_human);
      setStat("stat-human", s.redirected_to_human);
      setStat("stat-pct", `${s.human_redirect_pct}%`);
    })
    .catch(() => {});
}

function setStat(id, value){
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

/* ============================================================
   ПРЕЗЕНТАЦИЯ
   ============================================================ */
function initPresentation(){
  const state = { slides: [], index: 0, loaded: false };

  const frame     = document.getElementById("deck-image");
  const counter   = document.getElementById("deck-counter");
  const section   = document.getElementById("deck-section");
  const title     = document.getElementById("deck-title");
  const subtitle  = document.getElementById("deck-subtitle");
  const termsBox  = document.getElementById("deck-terms");
  const sideList  = document.getElementById("deck-thumbs");
  const prevBtn   = document.getElementById("deck-prev");
  const nextBtn   = document.getElementById("deck-next");

  function render(){
    const s = state.slides[state.index];
    if (!s) return;

    frame.src = s.image;
    frame.alt = s.title;
    counter.innerHTML = `<b>${String(s.number).padStart(2, "0")}</b> / ${state.slides.length}`;
    section.textContent = s.section;
    title.textContent = s.title;
    subtitle.textContent = s.subtitle;

    termsBox.innerHTML = "";
    if (s.terms && s.terms.length){
      const label = document.createElement("div");
      label.className = "deck-terms__label";
      label.textContent = "Термины на этом слайде — почитать подробнее";
      termsBox.appendChild(label);
      s.terms.forEach(t => termsBox.appendChild(makeTermChip(t)));
    }

    prevBtn.disabled = state.index === 0;
    nextBtn.disabled = state.index === state.slides.length - 1;

    document.querySelectorAll(".thumb").forEach(el => {
      el.classList.toggle("is-active", Number(el.dataset.index) === state.index);
    });
    const activeThumb = sideList.querySelector(".thumb.is-active");
    if (activeThumb) activeThumb.scrollIntoView({ block: "nearest" });
  }

  function goTo(i){
    state.index = Math.max(0, Math.min(state.slides.length - 1, i));
    render();
  }

  prevBtn.addEventListener("click", () => goTo(state.index - 1));
  nextBtn.addEventListener("click", () => goTo(state.index + 1));

  document.addEventListener("keydown", (e) => {
    if (!document.getElementById("view-presentation").classList.contains("is-active")) return;
    if (e.key === "ArrowRight") goTo(state.index + 1);
    if (e.key === "ArrowLeft") goTo(state.index - 1);
  });

  window.addEventListener("deck:shown", () => { if (!state.loaded) load(); });

  function load(){
    state.loaded = true;
    fetch("/api/presentation")
      .then(r => r.json())
      .then(data => {
        state.slides = data.slides;
        sideList.innerHTML = "";
        state.slides.forEach((s, i) => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "thumb";
          btn.dataset.index = i;
          btn.innerHTML = `
            <img src="${s.image}" alt="" loading="lazy">
            <div class="thumb__meta">
              <div class="thumb__num">${String(s.number).padStart(2, "0")}</div>
              <div class="thumb__title">${escapeHtml(s.title)}</div>
            </div>`;
          btn.addEventListener("click", () => goTo(i));
          sideList.appendChild(btn);
        });
        goTo(0);
      })
      .catch(() => {
        sideList.innerHTML = "<div class='loading-row'>Не удалось загрузить презентацию</div>";
      });
  }
}

function makeTermChip(t){
  const a = document.createElement("a");
  a.className = "term-chip";
  a.href = t.url;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.title = t.definition;
  a.innerHTML = `${escapeHtml(t.name)} ${EXTERNAL_ICON}`;
  return a;
}

/* ============================================================
   ГЛОССАРИЙ
   ============================================================ */
function initGlossary(){
  const grid   = document.getElementById("glossary-grid");
  const search = document.getElementById("glossary-search");
  let terms = [];

  fetch("/api/glossary")
    .then(r => r.json())
    .then(data => { terms = data.terms; renderGlossary(terms); })
    .catch(() => { grid.innerHTML = "<div class='loading-row'>Не удалось загрузить глоссарий</div>"; });

  search.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    const filtered = !q ? terms : terms.filter(t =>
      t.name.toLowerCase().includes(q) || t.definition.toLowerCase().includes(q));
    renderGlossary(filtered);
  });

  function renderGlossary(list){
    if (!list.length){
      grid.innerHTML = "<div class='loading-row'>Ничего не найдено</div>";
      return;
    }
    grid.innerHTML = list.map(t => `
      <div class="term-card">
        <h3>${escapeHtml(t.name)}</h3>
        <p>${escapeHtml(t.definition)}</p>
        <a class="term-card__link" href="${t.url}" target="_blank" rel="noopener noreferrer">
          Читать подробнее ${EXTERNAL_ICON}
        </a>
        <span class="term-card__source">${escapeHtml(t.source)}</span>
      </div>
    `).join("");
  }
}

/* ============================================================
   УТИЛИТЫ
   ============================================================ */
function escapeHtml(str){
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
