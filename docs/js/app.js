/* ============================================================
   粤策哨 · 静态演示版 · 交互逻辑
   纯前端：无后端、无请求、无密钥；内容不出本机。
   ============================================================ */

(function () {
  "use strict";

  /* ---------- 元素引用 ---------- */
  const form = document.getElementById("check-form");
  const input = document.getElementById("project-input");
  const checkBtn = document.getElementById("check-btn");
  const clearBtn = document.getElementById("clear-btn");
  const charCount = document.getElementById("char-count");
  const caseList = document.getElementById("case-list");
  const notice = document.getElementById("notice");
  const noticeBack = document.getElementById("notice-back");
  const reportWrap = document.getElementById("report-wrap");
  const reportBody = document.getElementById("report-body");

  let activeCaseId = null;

  /* ---------- 工具函数 ---------- */
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null && text !== "") node.textContent = text;
    return node;
  }

  function normalize(text) {
    return text.replace(/[\s\u3000，。；：、！？""''（）()《》【】\-—,.;:!?'"‘’]/g, "");
  }

  function findMatchedCase(text) {
    const n = normalize(text);
    if (!n) return null;
    for (const c of PRESET_CASES) {
      const nc = normalize(c.inputText);
      if (n === nc) return c;
      if (n.length >= 60 && nc.includes(n)) return c;
    }
    return null;
  }

  function updateCharCount() {
    charCount.textContent = input.value.length + " 字";
  }

  function setStatus(message, isError) {
    let status = document.getElementById("form-status");
    if (!status) {
      status = el("p", "form-status");
      status.id = "form-status";
      status.setAttribute("role", "alert");
      form.querySelector(".form-actions").after(status);
    }
    status.textContent = message || "";
    status.classList.toggle("is-error", !!isError);
    status.hidden = !message;
  }

  function setActiveCard(id) {
    activeCaseId = id;
    caseList.querySelectorAll(".case-card").forEach(function (card) {
      const active = card.dataset.caseId === id;
      card.classList.toggle("is-active", active);
      card.setAttribute("aria-pressed", active ? "true" : "false");
    });
  }

  /* ---------- 预置案例卡片 ---------- */
  function renderCases() {
    PRESET_CASES.forEach(function (c) {
      const li = el("li");
      const card = el("button", "case-card");
      card.type = "button";
      card.dataset.caseId = c.id;
      card.setAttribute("aria-pressed", "false");

      card.appendChild(el("span", "case-tag", c.tag));
      card.appendChild(el("h3", "case-title", c.title));
      card.appendChild(el("p", "case-desc", c.brief));
      card.appendChild(el("p", "case-src", "案例来源：" + c.source));
      card.appendChild(el("p", "case-fill-hint", "点击填入输入框 →"));

      card.addEventListener("click", function () {
        input.value = c.inputText;
        updateCharCount();
        setActiveCard(c.id);
        setStatus("已填入「" + c.tag + "」案例描述，点击「开始体检」查看报告。");
        input.focus();
        input.scrollIntoView({ behavior: "smooth", block: "center" });
      });

      li.appendChild(card);
      caseList.appendChild(li);
    });
  }

  /* ---------- 报告渲染 ---------- */
  function renderReport(caseDef) {
    const report = REPORTS[caseDef.id];
    const meta = report.meta;
    const doc = el("article", "report-doc");

    /* 报告头部 */
    const head = el("header", "report-head");
    head.appendChild(el("p", "report-title-line", "粤策哨 · 四段式申报前自查报告"));

    /* 体检发现项摘要：只报计数、不下结论；点击平滑跳到对应段 */
    const suspectCount = report.s2.filter(function (h) { return h.level !== "rule"; }).length;
    const summary = el("nav", "report-summary");
    summary.setAttribute("aria-label", "体检发现项导航");
    summary.appendChild(el("span", "report-summary-label", "体检发现"));
    [
      { label: "适用政策", count: report.s1.length, target: "report-sec-1" },
      { label: "规则命中", count: report.s2.length - suspectCount, target: "report-sec-2", mod: "is-rule" },
      { label: "疑似命中", count: suspectCount, target: "report-sec-2", mod: "is-suspect" },
      { label: "材料缺项", count: report.s3.length, target: "report-sec-3" },
      { label: "修改建议", count: report.s4.length, target: "report-sec-4" }
    ].forEach(function (d) {
      if (!d.count) return;
      const chip = el("button", "sum-chip" + (d.mod ? " " + d.mod : ""));
      chip.type = "button";
      chip.appendChild(document.createTextNode(d.label + " "));
      chip.appendChild(el("b", null, String(d.count)));
      chip.addEventListener("click", function () {
        document.getElementById(d.target).scrollIntoView({ behavior: "smooth", block: "start" });
      });
      summary.appendChild(chip);
    });
    head.appendChild(summary);

    const metaList = el("dl", "report-meta");
    const mkMeta = function (dtText, ddText) {
      const row = el("div");
      row.appendChild(el("dt", null, dtText));
      row.appendChild(el("dd", null, ddText));
      metaList.appendChild(row);
    };
    mkMeta("体检输入", "「" + meta.caseTag + "」" + meta.caseTitle);
    mkMeta("案例来源", meta.caseSource);
    if (meta.sourceUrl) {
      const row = el("div");
      row.appendChild(el("dt", null, "来源链接"));
      const dd = el("dd");
      const a = el("a", null, meta.sourceUrl);
      a.href = meta.sourceUrl;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      dd.appendChild(a);
      row.appendChild(dd);
      metaList.appendChild(row);
    }
    mkMeta("生成方式", "静态预置输出（本页内置，无模型推理）");
    head.appendChild(metaList);
    doc.appendChild(head);

    if (meta.sourceNote) {
      const noteP = el("p", "sec-note");
      noteP.textContent = meta.sourceNote;
      const wrap = el("div", "report-head-note");
      wrap.appendChild(noteP);
      doc.appendChild(wrap);
    }

    /* 第①段 适用政策清单 */
    const s1 = el("section", "report-sec");
    s1.id = "report-sec-1";
    s1.appendChild(el("h3", null, ""));
    s1.querySelector("h3").appendChild(el("span", "sec-no", "①"));
    s1.querySelector("h3").appendChild(document.createTextNode("适用政策清单"));
    s1.appendChild(el("p", "sec-note",
      "按项目主题列出可能相关的政策文件与条款。时效状态取自知识库注册表（registry.json）与政策文件清单；「申报已截止」指文件载明的受理窗口已过，仅作参考。"));
    const policyList = el("ul", "policy-list");
    report.s1.forEach(function (p) {
      const item = el("li", "policy-item");
      const headRow = el("div", "policy-head");
      headRow.appendChild(el("span", "policy-wenhao", p.wenh));
      headRow.appendChild(el("span", "policy-title", p.title));
      const pillClass = p.status === "valid" ? "pill-valid" : p.status === "deadline" ? "pill-deadline" : "pill-void";
      headRow.appendChild(el("span", "pill " + pillClass, p.statusLabel));
      item.appendChild(headRow);
      item.appendChild(el("p", "policy-meta", p.dateLine));
      if (p.clauses && p.clauses.length) {
        const chips = el("div", "clause-chips");
        p.clauses.forEach(function (cl) {
          const chip = el("span", "clause-chip");
          const b = el("b", null, cl.no);
          chip.appendChild(b);
          chip.appendChild(document.createTextNode("　" + cl.text));
          chips.appendChild(chip);
        });
        item.appendChild(chips);
      }
      if (p.note) item.appendChild(el("p", "policy-note", "说明：" + p.note));
      policyList.appendChild(item);
    });
    s1.appendChild(policyList);
    doc.appendChild(s1);

    /* 第②段 负面清单命中提示 */
    const s2 = el("section", "report-sec");
    s2.id = "report-sec-2";
    const s2h = el("h3", null, "");
    s2h.appendChild(el("span", "sec-no", "②"));
    s2h.appendChild(document.createTextNode("负面清单命中提示"));
    s2.appendChild(s2h);
    s2.appendChild(el("p", "sec-note",
      "命中分两级：规则命中、疑似命中（建议对照原文）。每条附依据文号、条款号与日期；条款原文摘录自知识库清洗文本，逐字核对。"));
    const hitList = el("ol", "hit-list");
    report.s2.forEach(function (hit) {
      const item = el("li", "hit-item" + (hit.level === "rule" ? " is-rule" : " is-suspect"));
      const headRow = el("div", "hit-head");
      headRow.appendChild(el("span", "badge " + (hit.level === "rule" ? "badge-rule" : "badge-suspect"),
        hit.level === "rule" ? "规则命中" : "疑似命中，建议对照原文"));
      headRow.appendChild(el("span", "hit-title", hit.title));
      item.appendChild(headRow);
      item.appendChild(el("p", "hit-desc", hit.desc));

      const ev = el("div", "hit-evidence");
      hit.evidence.forEach(function (line) {
        if (!line.v && !line.k) return;
        const row = el("div", "ev-line");
        row.appendChild(el("span", "ev-k", line.k));
        if (line.k === "条款原文摘录" && line.v) {
          const quote = el("p", "ev-quote", line.v);
          ev.appendChild(quote);
          return;
        }
        row.appendChild(el("span", "ev-v", line.v || "——"));
        ev.appendChild(row);
      });
      item.appendChild(ev);
      if (hit.note) item.appendChild(el("p", "hit-note", "说明：" + hit.note));
      hitList.appendChild(item);
    });
    s2.appendChild(hitList);
    doc.appendChild(s2);

    /* 第③段 材料缺项清单 */
    const s3 = el("section", "report-sec");
    s3.id = "report-sec-3";
    const s3h = el("h3", null, "");
    s3h.appendChild(el("span", "sec-no", "③"));
    s3h.appendChild(document.createTextNode("材料缺项清单"));
    s3.appendChild(s3h);
    s3.appendChild(el("p", "sec-note", "对照第①段命中的现行政策文件，列出建议核对、补齐的申报材料（清单为提示性质）。"));
    const missList = el("ul", "missing-list");
    report.s3.forEach(function (m) {
      const item = el("li", "missing-item");
      const body = el("div", "missing-body");
      body.appendChild(el("p", "missing-name", m.name));
      body.appendChild(el("p", "missing-basis", "依据：" + m.basis));
      item.appendChild(body);
      missList.appendChild(item);
    });
    s3.appendChild(missList);
    doc.appendChild(s3);

    /* 第④段 修改建议 */
    const s4 = el("section", "report-sec");
    s4.id = "report-sec-4";
    const s4h = el("h3", null, "");
    s4h.appendChild(el("span", "sec-no", "④"));
    s4h.appendChild(document.createTextNode("修改建议"));
    s4.appendChild(s4h);
    s4.appendChild(el("p", "sec-note", "以下为申报表述层面的改进建议（建议口吻，不做任何判定）。"));
    const adviceList = el("ul", "advice-list");
    report.s4.forEach(function (a) {
      const item = el("li", "advice-item");
      item.appendChild(el("p", null, a));
      adviceList.appendChild(item);
    });
    s4.appendChild(adviceList);
    doc.appendChild(s4);

    /* 报告尾部：免责声明 + 操作 */
    const foot = el("footer", "report-foot");
    foot.appendChild(el("p", "report-disclaimer", "免责声明：" + DISCLAIMER_TEXT));
    const actions = el("div", "report-foot-actions");
    const backBtn = el("button", "btn btn-ghost", "返回输入框");
    backBtn.type = "button";
    backBtn.addEventListener("click", function () {
      input.scrollIntoView({ behavior: "smooth", block: "center" });
      input.focus();
    });
    actions.appendChild(backBtn);
    foot.appendChild(actions);
    foot.appendChild(el("p", "report-foot-note", "本报告为静态演示输出。全部引用文号、条款号与日期来源见页面底部「演示数据来源说明」。"));
    doc.appendChild(foot);

    reportBody.replaceChildren(doc);
    reportWrap.hidden = false;

    /* 焦点移交：标题可聚焦，键盘与读屏用户随滚动一起进入报告 */
    const titleEl = doc.querySelector(".report-title-line");
    titleEl.setAttribute("tabindex", "-1");
    titleEl.focus({ preventScroll: true });
  }

  /* ---------- 体检流程 ---------- */
  function runCheck() {
    const text = input.value.trim();

    if (!text) {
      setStatus("请先输入项目描述，或点击下方预置案例一键填入。", true);
      input.classList.add("is-error");
      input.focus();
      return;
    }
    input.classList.remove("is-error");

    const matched = findMatchedCase(text);
    if (matched) {
      setActiveCard(matched.id);
      setStatus("");
      notice.hidden = true;
      renderReport(matched);
      reportWrap.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      setStatus("");
      reportWrap.hidden = true;
      notice.hidden = false;
      notice.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  /* ---------- 事件绑定与初始化 ---------- */
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    runCheck();
  });

  input.addEventListener("input", function () {
    updateCharCount();
    if (input.value.trim()) {
      input.classList.remove("is-error");
      setStatus("");
    }
  });

  clearBtn.addEventListener("click", function () {
    input.value = "";
    updateCharCount();
    input.classList.remove("is-error");
    setStatus("");
    setActiveCard(null);
    reportWrap.hidden = true;
    notice.hidden = true;
    input.focus();
  });

  noticeBack.addEventListener("click", function () {
    notice.hidden = true;
    document.getElementById("presets").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  renderCases();
  updateCharCount();
})();
