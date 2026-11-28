/* ============================================================
   粤策哨 · 静态演示版 · 交互逻辑（R6 前端双模式 ＋ 阶段二工具型单页）
   - 离线演示（演示数据）：「试试示例」小卡片 → data.js 内置报告，无请求；
   - 在线体检（真实分析）：输入框自由描述 → apiBase 在线服务，
     提交 → 轮询 → 服务端 rendered_text 原样渲染。
   服务地址优先级：?api= 参数 ＞ localStorage(yueceshao_api_base) ＞ 默认常量。
   红线：前端不生成、不补写、不美化任何报告内容——在线模式的一切
   内容来自 API 响应，离线模式来自 data.js 既有数据；
   AI 生成标识行与免责声明在展示与复制中均完整保留。
   ============================================================ */

(function () {
  "use strict";

  /* ---------- 在线体检服务基址 ----------
     优先级：URL 参数 ?api= ＞ localStorage（键 yueceshao_api_base）＞ 默认常量。
     只接受 http(s) URL（new URL 校验，非法值静默降级到下一级，防注入）；
     当前生效地址在页脚一行只读展示（textContent，不解析链接）。
     默认本地 Mock：uvicorn app.main:app --port 8000；
     生产值由组长在部署时提供（备案域名），改本行即可。 */
  const API_BASE_DEFAULT = "http://127.0.0.1:8000";
  const API_BASE_STORAGE_KEY = "yueceshao_api_base";

  function sanitizeApiBase(raw) {
    if (typeof raw !== "string") return null;
    const value = raw.trim();
    if (!value) return null;
    let parsed;
    try {
      parsed = new URL(value);
    } catch (err) {
      return null; /* 非 URL（javascript:/data:/相对串等）一律拒绝 */
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
    return value.replace(/\/+$/, ""); /* 去掉结尾斜杠，拼接路径不双斜杠 */
  }

  function resolveApiBase() {
    let fromUrl = null;
    try {
      fromUrl = sanitizeApiBase(new URLSearchParams(window.location.search).get("api"));
    } catch (err) { fromUrl = null; }
    if (fromUrl) return fromUrl;
    let fromStore = null;
    try {
      fromStore = sanitizeApiBase(window.localStorage.getItem(API_BASE_STORAGE_KEY));
    } catch (err) { fromStore = null; } /* 隐私模式等读取异常按未设置处理 */
    return fromStore || API_BASE_DEFAULT;
  }

  const apiBase = resolveApiBase();

  /* 在线体检参数（契约：轮询每 5 秒一次；客户端轮询上限 15 分钟） */
  const POLL_INTERVAL_MS = 5000;
  const POLL_MAX_MS = 15 * 60 * 1000;
  /* 输入长度阈值与后端 app/main.py 校验口径一致（strip 后计字） */
  const MIN_DESC_CHARS = 20;
  const MAX_DESC_CHARS = 2000;

  /* ---------- 元素引用 ---------- */
  const form = document.getElementById("check-form");
  const input = document.getElementById("project-input");
  const checkBtn = document.getElementById("check-btn");
  const randomBtn = document.getElementById("random-btn");
  const clearBtn = document.getElementById("clear-btn");
  const charCount = document.getElementById("char-count");
  const sampleRow = document.getElementById("sample-row");
  const reportWrap = document.getElementById("report-wrap");
  const reportBody = document.getElementById("report-body");
  const analyzePanel = document.getElementById("analyze");
  const analyzeSteps = document.getElementById("analyze-steps");
  const onlineStatus = document.getElementById("online-status");
  const onlineStatusCard = onlineStatus.querySelector(".online-status-card");
  const onlineStatusTitle = document.getElementById("online-status-title");
  const onlineStatusText = document.getElementById("online-status-text");
  const onlineStatusMeta = document.getElementById("online-status-meta");
  const onlineStatusActions = document.getElementById("online-status-actions");
  const onlineRetryBtn = document.getElementById("online-retry-btn");
  const onlineReport = document.getElementById("online-report");
  const onlineReportPre = document.getElementById("online-report-pre");
  const copyOnlineBtn = document.getElementById("copy-online-btn");
  const apiNote = document.getElementById("api-base-note");

  const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let activeCaseId = null;
  let analyzeTimer = null;
  let analyzeFallbackTimer = null;

  /* 在线体检进行中状态 */
  let onlineInFlight = false;
  let onlineTaskId = null;
  let onlinePollUrl = null;
  let onlinePollTimer = null;
  let onlineElapsedTimer = null;
  let onlineDeadline = 0;
  let onlineStartedAt = 0;
  let onlineEtaText = ""; /* estimated_seconds 折算的「预计约 N 分钟」提示 */

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

  /* 取 evidence 数组里指定 k 的 v（如「条款原文摘录」） */
  function evidenceValue(hit, key) {
    const line = (hit.evidence || []).find(function (l) { return l.k === key; });
    return line ? line.v : "";
  }

  /* 关键词高亮：把 text 中出现的每个词包进 <mark>（纯 DOM 构建，不用 innerHTML）。
     只高亮在 text 中真实出现的词，避免空挂；长词优先，防止短词截断长词。 */
  function highlightText(text, words) {
    const frag = document.createDocumentFragment();
    const found = Array.from(new Set((words || []).filter(function (w) {
      return w && text.indexOf(w) !== -1;
    }))).sort(function (a, b) { return b.length - a.length; });
    if (!found.length) {
      frag.appendChild(document.createTextNode(text));
      return frag;
    }
    const re = new RegExp(found.map(function (w) {
      return w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }).join("|"), "g");
    let last = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      const mark = el("mark", "hit-mark");
      mark.textContent = m[0];
      frag.appendChild(mark);
      last = m.index + m[0].length;
      if (m[0].length === 0) re.lastIndex += 1; // 防空匹配死循环
    }
    if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
    return frag;
  }

  /* 数字滚动计数（尊重减少动效偏好，直接落在终值） */
  function animateCount(node, target) {
    if (REDUCED_MOTION || target <= 0) {
      node.textContent = String(target);
      return;
    }
    const start = performance.now();
    const duration = 600;
    let settled = false;
    function finish() {
      if (settled) return;
      settled = true;
      node.textContent = String(target);
    }
    function tick(now) {
      const p = Math.min((now - start) / duration, 1);
      node.textContent = String(Math.round(target * (1 - Math.pow(1 - p, 3))));
      if (p < 1) {
        requestAnimationFrame(tick);
      } else {
        finish();
      }
    }
    requestAnimationFrame(tick);
    setTimeout(finish, duration + 400); // 兜底：极端节流环境（后台标签页等）也保证终值正确
  }

  /* 预计耗时文案（estimated_seconds 为 UX 参考值非承诺）：≥60 秒折算分钟。 */
  function formatEta(seconds) {
    const s = Number(seconds);
    if (!isFinite(s) || s <= 0) return "";
    if (s >= 60) return "预计约 " + Math.max(1, Math.round(s / 60)) + " 分钟";
    return "预计约 " + Math.round(s) + " 秒";
  }

  /* 状态条 meta：预计提示与已等待秒数并存（不做倒计时进度条）。 */
  function elapsedText() {
    const waited = Math.max(0, Math.floor((Date.now() - onlineStartedAt) / 1000));
    return (onlineEtaText ? onlineEtaText + " · " : "") + "已等待 " + waited + " 秒";
  }

  /* ---------- 复制报告文本（在线/离线通用） ----------
     复制内容与页面所见逐字一致（含 AI 生成标识行＋免责声明），不截断；
     navigator.clipboard 优先，非安全上下文/失败降级 execCommand；
     成功反馈「已复制」。 */
  function copyReportText(text, btn) {
    const LABEL = "复制报告文本";
    function flash(message, ms) {
      btn.textContent = message;
      clearTimeout(btn._copyTimer);
      btn._copyTimer = setTimeout(function () { btn.textContent = LABEL; }, ms);
    }
    function legacyCopy() {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
      document.body.removeChild(ta);
      if (ok) { flash("已复制", 1800); } else { flash("复制失败，请手动选择文本", 2400); }
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () { flash("已复制", 1800); },
        legacyCopy);
    } else {
      legacyCopy();
    }
  }

  /* 离线报告文本：从当前已渲染 DOM 取所见文本（四个段落全部计入——
     临时展开切换视图，复制按钮/切换条等交互控件文字不计入），逐字一致。 */
  function offlineReportText() {
    const doc = reportBody.querySelector(".report-doc");
    if (!doc) return "";
    const sections = Array.prototype.slice.call(doc.querySelectorAll(".report-sec"));
    const hiddenBackup = sections.map(function (s) { return s.hidden; });
    const chrome = Array.prototype.slice.call(
      doc.querySelectorAll(".report-tabs, .report-foot-actions"));
    const displayBackup = chrome.map(function (n) { return n.style.display; });
    sections.forEach(function (s) { s.hidden = false; });
    chrome.forEach(function (n) { n.style.display = "none"; });
    var text;
    try {
      text = doc.innerText;
    } finally {
      sections.forEach(function (s, i) { s.hidden = hiddenBackup[i]; });
      chrome.forEach(function (n, i) { n.style.display = displayBackup[i]; });
    }
    return text.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
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
    sampleRow.querySelectorAll(".sample-card").forEach(function (card) {
      const active = card.dataset.caseId === id;
      card.classList.toggle("is-active", active);
      card.setAttribute("aria-pressed", active ? "true" : "false");
    });
  }

  /* ---------- 试试示例（弱化保留：输入框下方一行小卡片） ----------
     点击即填入并直接走离线渲染（演示数据，内容不出本机），不抢主视觉。 */
  function renderSamples() {
    PRESET_CASES.forEach(function (c) {
      const card = el("button", "sample-card");
      card.type = "button";
      card.dataset.caseId = c.id;
      card.setAttribute("aria-pressed", "false");

      card.appendChild(el("span", "sample-tag", c.tag));
      card.appendChild(el("span", "sample-title", c.title));

      card.addEventListener("click", function () {
        if (onlineInFlight || !analyzePanel.hidden) return;
        input.value = c.inputText;
        updateCharCount();
        input.classList.remove("is-error");
        setStatus("");
        runCheck(); /* 预置案例命中 → 离线渲染对应报告（含 AI 行＋免责声明） */
      });

      sampleRow.appendChild(card);
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

    const metaList = el("dl", "report-meta");
    const mkMeta = function (dtText, ddText) {
      const row = el("div");
      row.appendChild(el("dt", null, dtText));
      row.appendChild(el("dd", null, ddText));
      metaList.appendChild(row);
    };
    mkMeta("体检输入", "「" + meta.caseTag + "」" + meta.caseTitle);

    /* 行政信息折叠：案例来源 / 来源链接 / 生成方式默认收起 */
    const metaFold = el("div", "meta-fold");
    const foldDetails = el("details", "meta-details");
    const foldSummary = el("summary", "meta-summary", "案例来源 · 链接 · 生成方式");
    const foldRows = el("div", "meta-rows");
    const mkFoldRow = function (k, valueNode) {
      const row = el("div", "meta-row");
      row.appendChild(el("span", "meta-k", k));
      const v = el("span", "meta-v");
      if (valueNode) v.appendChild(valueNode);
      row.appendChild(v);
      foldRows.appendChild(row);
    };
    mkFoldRow("案例来源", document.createTextNode(meta.caseSource));
    if (meta.sourceUrl) {
      const a = el("a", null, meta.sourceUrl);
      a.href = meta.sourceUrl;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      mkFoldRow("来源链接", a);
    }
    mkFoldRow("生成方式", document.createTextNode("静态预置输出（本页内置，无模型推理）"));
    foldDetails.appendChild(foldSummary);
    foldDetails.appendChild(foldRows);
    metaFold.appendChild(foldDetails);
    metaList.appendChild(metaFold);
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
    s1.querySelector("h3").appendChild(el("span", "sec-no", "1"));
    s1.querySelector("h3").appendChild(document.createTextNode("适用政策清单"));
    s1.appendChild(el("p", "sec-note",
      "按项目主题列出可能相关的政策文件与条款。时效状态取自知识库注册表（registry.json）与政策文件清单；「申报已截止」指文件载明的受理窗口已过，仅作参考。"));
    const policyList = el("ul", "policy-list");
    report.s1.forEach(function (p) {
      const item = el("li", "policy-item");
      const headRow = el("div", "policy-head");
      /* 空字段守卫：wenh 与 title 为空时不渲染空 span（C02 无文号条目
         文号以《标题》形式放在 wenh、title 为空，见 data.js） */
      if (p.wenh) headRow.appendChild(el("span", "policy-wenhao", p.wenh));
      if (p.title) headRow.appendChild(el("span", "policy-title", p.title));
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

    /* 第②段 负面清单命中提示：三步走并排对照
       ①命中级别与说明 → ②左栏材料原文命中句 → ③右栏条款原文摘录；
       规则命中的触发词两栏同高亮（<mark>），疑似命中只并排、不高亮。 */
    const s2 = el("section", "report-sec");
    s2.id = "report-sec-2";
    const s2h = el("h3", null, "");
    s2h.appendChild(el("span", "sec-no", "2"));
    s2h.appendChild(document.createTextNode("负面清单命中提示"));
    s2.appendChild(s2h);
    s2.appendChild(el("p", "sec-note",
      "命中分两级：规则命中、疑似命中（建议对照原文）。每条命中左侧为材料原文里命中的那一句、右侧为条款原文摘录，两栏并排便于对照；高亮词＝触发命中的关键词，仅规则命中标注——疑似命中为语义提示、无字眼级定位，不高亮。依据文号、条款号与日期逐条附于对照区下方，条款原文摘录自知识库清洗文本、逐字核对。"));
    const hitList = el("ol", "hit-list");
    report.s2.forEach(function (hit) {
      const isRule = hit.level === "rule";
      const item = el("li", "hit-item" + (isRule ? " is-rule" : " is-suspect"));
      const headRow = el("div", "hit-head");
      headRow.appendChild(el("span", "badge " + (isRule ? "badge-rule" : "badge-suspect"),
        isRule ? "规则命中" : "疑似命中，建议对照原文"));
      headRow.appendChild(el("span", "hit-title", hit.title));
      item.appendChild(headRow);
      item.appendChild(el("p", "hit-desc", hit.desc));

      /* 并排对照区：左＝材料原文命中句（触发词高亮），右＝条款原文摘录（对应词高亮） */
      const pairs = isRule && Array.isArray(hit.matchedWords) ? hit.matchedWords : [];
      const compare = el("div", "hit-compare");
      compare.setAttribute("aria-label", "材料原文与条款原文并排对照");

      const srcPane = el("div", "hit-pane hit-pane-src");
      srcPane.appendChild(el("p", "hit-pane-head", "材料原文 · 命中句"));
      const srcQuote = el("blockquote", "hit-quote hit-quote-src");
      srcQuote.appendChild(highlightText(hit.sourceQuote || "",
        pairs.map(function (p) { return p.desc; })));
      srcPane.appendChild(srcQuote);

      const link = el("span", "hit-compare-link", "对照");

      const clausePane = el("div", "hit-pane hit-pane-clause");
      clausePane.appendChild(el("p", "hit-pane-head", "条款原文摘录"));
      const clauseQuote = el("blockquote", "hit-quote hit-quote-clause");
      clauseQuote.appendChild(highlightText(evidenceValue(hit, "条款原文摘录"),
        pairs.map(function (p) { return p.clause; })));
      clausePane.appendChild(clauseQuote);

      compare.appendChild(srcPane);
      compare.appendChild(link);
      compare.appendChild(clausePane);
      item.appendChild(compare);

      /* 依据信息平铺展示（文号/条款号/日期/来源逐条强制标注）；条款原文已上移至右栏 */
      const ev = el("div", "hit-evidence");
      hit.evidence.forEach(function (line) {
        if (!line.v && !line.k) return;
        if (line.k === "条款原文摘录") return; /* 已在右栏整句展示 */
        const row = el("div", "ev-line");
        row.appendChild(el("span", "ev-k", line.k));
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
    s3h.appendChild(el("span", "sec-no", "3"));
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
    s4h.appendChild(el("span", "sec-no", "4"));
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

    /* 段落切换条：四段改为切换视图，避免长滚动信息过载 */
    const tabSpecs = [
      { sec: s1, label: "适用政策", count: report.s1.length },
      { sec: s2, label: "负面命中", count: report.s2.length },
      { sec: s3, label: "材料缺项", count: report.s3.length },
      { sec: s4, label: "修改建议", count: report.s4.length }
    ];
    const tabBar = el("div", "report-tabs");
    tabBar.setAttribute("role", "tablist");
    tabBar.setAttribute("aria-label", "报告段落切换");
    const tabs = [];
    function activateTab(idx, moveFocus) {
      tabSpecs.forEach(function (spec, j) {
        spec.sec.hidden = j !== idx;
        const t = tabs[j];
        t.classList.toggle("is-active", j === idx);
        t.setAttribute("aria-selected", j === idx ? "true" : "false");
        t.setAttribute("tabindex", j === idx ? "0" : "-1");
      });
      if (moveFocus) tabs[idx].focus();
    }
    tabSpecs.forEach(function (spec, idx) {
      spec.sec.hidden = idx !== 0;
      spec.sec.setAttribute("role", "tabpanel");
      const t = el("button", "report-tab" + (idx === 0 ? " is-active" : ""));
      t.type = "button";
      t.id = "report-tab-" + (idx + 1);
      t.setAttribute("role", "tab");
      t.setAttribute("aria-selected", idx === 0 ? "true" : "false");
      t.setAttribute("tabindex", idx === 0 ? "0" : "-1");
      t.setAttribute("aria-controls", spec.sec.id);
      spec.sec.setAttribute("aria-labelledby", t.id);
      t.appendChild(document.createTextNode(spec.label + " "));
      const num = el("b", null, "0");
      t.appendChild(num);
      animateCount(num, spec.count);
      t.addEventListener("click", function () { activateTab(idx, false); });
      tabs.push(t);
      tabBar.appendChild(t);
    });
    tabBar.addEventListener("keydown", function (e) {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault();
      let cur = 0;
      for (let j = 0; j < tabs.length; j++) {
        if (tabs[j].classList.contains("is-active")) { cur = j; break; }
      }
      const delta = e.key === "ArrowRight" ? 1 : -1;
      activateTab((cur + delta + tabs.length) % tabs.length, true);
    });
    head.appendChild(tabBar);

    /* 报告尾部：AI 生成标识行（R6 待办①，与在线口径同位同文：插在免责声明之前）
       ＋免责声明 ＋ 操作 */
    const foot = el("footer", "report-foot");
    if (meta.aiNote) {
      foot.appendChild(el("p", "report-ai-note", meta.aiNote));
    }
    foot.appendChild(el("p", "report-disclaimer", "免责声明：" + DISCLAIMER_TEXT));
    const actions = el("div", "report-foot-actions");
    const copyBtn = el("button", "btn btn-ghost btn-copy", "复制报告文本");
    copyBtn.type = "button";
    copyBtn.addEventListener("click", function () {
      copyReportText(offlineReportText(), copyBtn);
    });
    actions.appendChild(copyBtn);
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

  /* ---------- 模拟体检过程（演示动画，非真实推理） ---------- */
  function playAnalyze(onDone) {
    clearTimeout(analyzeFallbackTimer);
    checkBtn.disabled = true;
    reportWrap.hidden = true;
    analyzePanel.hidden = false;
    const steps = Array.prototype.slice.call(analyzeSteps.children);
    steps.forEach(function (li) { li.classList.remove("is-active", "is-done"); });

    /* 进度反馈：进度条 + 百分比 + 分步文字 */
    const fill = document.getElementById("analyze-fill");
    const pct = document.getElementById("analyze-pct");
    const statusText = document.getElementById("analyze-status-text");
    const stepTexts = ["正在解析项目描述", "正在扫描政策语料库", "正在进行负面清单比对", "正在汇总结果"];
    fill.style.width = "0%";
    pct.textContent = "0%";
    statusText.textContent = stepTexts[0];

    analyzePanel.scrollIntoView({ behavior: "smooth", block: "center" });
    steps[0].classList.add("is-active");
    let i = 0;
    clearInterval(analyzeTimer);
    analyzeTimer = setInterval(function () {
      steps[i].classList.remove("is-active");
      steps[i].classList.add("is-done");
      i += 1;
      fill.style.width = i * 25 + "%";
      pct.textContent = i * 25 + "%";
      statusText.textContent = stepTexts[Math.min(i, 3)];
      if (i < steps.length) {
        steps[i].classList.add("is-active");
      } else {
        finishAnalyze(onDone);
      }
    }, 520);
    /* 兜底：极端节流环境（后台标签页等）下计时器停摆，也保证结果最终出现 */
    analyzeFallbackTimer = setTimeout(function () {
      if (!analyzePanel.hidden) finishAnalyze(onDone);
    }, 520 * 4 + 900);
  }

  function finishAnalyze(onDone) {
    clearInterval(analyzeTimer);
    clearTimeout(analyzeFallbackTimer);
    checkBtn.disabled = false;
    analyzePanel.hidden = true;
    onDone();
  }

  function showReport(caseDef) {
    renderReport(caseDef);
    reportWrap.scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "start" });
  }

  /* ---------- 在线体检（真实分析）：提交 → 轮询 → 原样渲染 ----------
     契约（§三）：POST /api/checkup → 202{task_id,poll_url,estimated_seconds}；
     GET poll_url 每 5 秒 → queued/running 继续、done 渲染 result.rendered_text、
     failed 展示 error 安全文案；404＝任务不存在/已过期；422/429 照实展示。
     内容红线：rendered_text 服务端已注入四段＋AI 生成标识行＋免责声明，
     前端等宽块原样渲染，不隐藏、不删改、不美化。 */

  function setOnlineStatus(title, text, meta, showRetry) {
    onlineStatus.hidden = false;
    onlineStatusTitle.textContent = title;
    onlineStatusText.textContent = text || "";
    if (meta !== null && meta !== undefined) onlineStatusMeta.textContent = meta;
    onlineStatusActions.hidden = !showRetry;
  }

  function stopOnlineTimers() {
    clearInterval(onlinePollTimer);
    clearInterval(onlineElapsedTimer);
    onlinePollTimer = null;
    onlineElapsedTimer = null;
  }

  function startOnlineElapsed() {
    onlineStartedAt = Date.now();
    onlineStatusMeta.textContent = elapsedText();
    clearInterval(onlineElapsedTimer);
    onlineElapsedTimer = setInterval(function () {
      onlineStatusMeta.textContent = elapsedText();
    }, 1000);
  }

  /* 终态（失败/未完成）：如实展示原因＋提供「重新提交」 */
  function showOnlineError(message) {
    stopOnlineTimers();
    onlineInFlight = false;
    checkBtn.disabled = false;
    onlineStatusCard.classList.add("is-error");
    setOnlineStatus("在线体检 · 未完成", message,
      onlineTaskId ? "任务编号 " + onlineTaskId : "", true);
  }

  /* 提交后进入轮询循环（含 15 分钟客户端上限） */
  function startPolling() {
    onlineDeadline = Date.now() + POLL_MAX_MS;
    clearInterval(onlinePollTimer);
    onlinePollTimer = setInterval(pollTaskStatus, POLL_INTERVAL_MS);
    pollTaskStatus();
  }

  async function pollTaskStatus() {
    if (Date.now() >= onlineDeadline) {
      /* 客户端轮询上限 15 分钟：停止轮询，展示 task_id＋重试指引 */
      stopOnlineTimers();
      onlineInFlight = false;
      checkBtn.disabled = false;
      onlineStatusCard.classList.remove("is-error");
      setOnlineStatus("在线体检 · 真实分析", "分析时间较长，可稍后手动重试。",
        onlineTaskId ? "任务编号 " + onlineTaskId : "", true);
      return;
    }
    let res;
    try {
      const url = /^https?:\/\//i.test(onlinePollUrl || "")
        ? onlinePollUrl : apiBase + (onlinePollUrl || "");
      res = await fetch(url);
    } catch (err) {
      setOnlineStatus("在线体检 · 真实分析", "轮询连接波动，正在重试…", null, false);
      return; /* 瞬时断连继续轮询，至截止时间为止 */
    }
    if (res.status === 404) {
      showOnlineError(await errorDetail(res) ||
        "任务不存在或已过期（保留 1 小时），请重新提交。");
      return;
    }
    if (!res.ok) {
      setOnlineStatus("在线体检 · 真实分析",
        "查询任务状态失败（HTTP " + res.status + "），正在重试…", null, false);
      return; /* 瞬时错误继续轮询；截止后走超时文案 */
    }
    let data;
    try {
      data = await res.json();
    } catch (err) {
      setOnlineStatus("在线体检 · 真实分析", "任务状态响应无法解析，正在重试…", null, false);
      return;
    }
    if (data.status === "queued" || data.status === "running") {
      /* 状态如实显示：queued 就写排队中 */
      onlineStatusCard.classList.remove("is-error");
      setOnlineStatus("在线体检 · 真实分析",
        data.status === "queued" ? "排队中" : "分析中", null, false);
      return;
    }
    if (data.status === "done") {
      stopOnlineTimers();
      onlineInFlight = false;
      checkBtn.disabled = false;
      renderOnlineReport(data.result);
      return;
    }
    if (data.status === "failed") {
      showOnlineError(data.error || "体检执行失败，请稍后重试");
      return;
    }
    /* 未知状态：如实显示并继续轮询 */
    setOnlineStatus("在线体检 · 真实分析", "当前状态：" + data.status, null, false);
  }

  /* 服务端错误详情（422 detail 为校验数组、429/404 为字符串），照实展示 */
  async function errorDetail(res) {
    try {
      const body = await res.json();
      const detail = body && body.detail;
      if (!detail) return "";
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail) && detail.length && detail[0] && detail[0].msg) {
        return String(detail[0].msg).replace(/^Value error,\s*/, "");
      }
      return "";
    } catch (err) {
      return "";
    }
  }

  function renderOnlineReport(result) {
    const text = result && typeof result.rendered_text === "string"
      ? result.rendered_text : "";
    if (!text.trim()) {
      showOnlineError("体检服务返回的报告内容为空，请稍后重试。");
      return;
    }
    onlineStatusCard.classList.remove("is-error");
    onlineStatus.hidden = true;
    onlineReportPre.textContent = text; /* 原样渲染，不增删改 */
    onlineReport.hidden = false;
    onlineReport.scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "start" });
  }

  async function submitOnlineCheck(description) {
    onlineInFlight = true;
    checkBtn.disabled = true;
    onlineTaskId = null;
    onlinePollUrl = null;
    onlineEtaText = "";
    reportWrap.hidden = true;
    analyzePanel.hidden = true;
    onlineReport.hidden = true;
    onlineStatusCard.classList.remove("is-error");
    setOnlineStatus("在线体检 · 真实分析", "正在提交项目描述…", "已等待 0 秒", false);
    onlineStatus.scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "center" });
    startOnlineElapsed();
    let res;
    try {
      res = await fetch(apiBase + "/api/checkup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description: description })
      });
    } catch (err) {
      showOnlineError("无法连接体检服务（" + apiBase + "）。请确认服务已启动；演示数据可点击下方「试试示例」离线查看。");
      return;
    }
    if (res.status === 202) {
      let data;
      try {
        data = await res.json();
      } catch (err) {
        showOnlineError("体检服务响应无法解析，请稍后重试。");
        return;
      }
      onlineTaskId = data.task_id || null;
      onlinePollUrl = data.poll_url || "/api/tasks/" + onlineTaskId;
      onlineEtaText = formatEta(data.estimated_seconds);
      onlineStatusCard.classList.remove("is-error");
      setOnlineStatus("在线体检 · 真实分析", "排队中", elapsedText(), false);
      startPolling();
      return;
    }
    if (res.status === 422) {
      showOnlineError(await errorDetail(res) || "提交被拒收，请检查项目描述后重试。");
      return;
    }
    if (res.status === 429) {
      const retryAfter = res.headers.get("Retry-After");
      showOnlineError((await errorDetail(res) || "提交过于频繁，请稍后再试。") +
        (retryAfter ? " 约 " + retryAfter + " 秒后可重试。" : ""));
      return;
    }
    showOnlineError("体检服务返回异常状态（HTTP " + res.status + "），请稍后重试。");
  }

  /* ---------- 体检流程（R6 双模式） ---------- */
  function runCheck() {
    if (onlineInFlight) return;          // 在线体检进行中，忽略重复提交
    if (!analyzePanel.hidden) return;    // 演示动画进行中，忽略重复提交
    const text = input.value.trim();

    if (!text) {
      setStatus("请先输入项目描述，或点击下方「试试示例」直接查看离线报告。", true);
      input.classList.add("is-error");
      input.focus();
      return;
    }
    input.classList.remove("is-error");

    const matched = findMatchedCase(text);
    if (matched) {
      /* 演示数据路径：预置案例离线渲染（R6 前行为保持不变） */
      setActiveCard(matched.id);
      setStatus("");
      onlineStatus.hidden = true;
      onlineReport.hidden = true;
      if (REDUCED_MOTION) {
        showReport(matched);
      } else {
        playAnalyze(function () { showReport(matched); });
      }
      return;
    }

    /* 真实分析路径：任意描述提交在线服务（长度阈值与后端校验口径一致） */
    if (text.length < MIN_DESC_CHARS) {
      setStatus("项目描述过短（当前 " + text.length + " 字），至少 20 字后再提交在线体检。", true);
      input.focus();
      return;
    }
    if (text.length > MAX_DESC_CHARS) {
      setStatus("项目描述过长（当前 " + text.length + " 字），上限 2000 字。", true);
      input.focus();
      return;
    }
    setActiveCard(null);
    setStatus("");
    submitOnlineCheck(text);
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

  /* ---------- 随机生成项目描述 ---------- */
  randomBtn.addEventListener("click", function () {
    let pool = PRESET_CASES;
    if (activeCaseId && PRESET_CASES.length > 1) {
      const others = PRESET_CASES.filter(function (c) { return c.id !== activeCaseId; });
      /* 排除当前项后仅剩 1 条或为空时降级回全池，保证随机性/可用性（当前 3 条案例不会触发，纯防御） */
      if (others.length > 1) pool = others;
    }
    if (!pool.length) {
      setStatus("暂无可抽取的示例，请直接点击下方「试试示例」。", true);
      return;
    }
    const c = pool[Math.floor(Math.random() * pool.length)];
    input.value = c.inputText;
    updateCharCount();
    setActiveCard(c.id);
    input.classList.remove("is-error");
    setStatus("已随机生成一份「" + c.tag + "」项目描述，点击「开始体检」查看报告。");
    input.focus();
  });

  clearBtn.addEventListener("click", function () {
    input.value = "";
    updateCharCount();
    input.classList.remove("is-error");
    setStatus("");
    setActiveCard(null);
    reportWrap.hidden = true;
    stopOnlineTimers();
    onlineInFlight = false;
    onlineTaskId = null;
    onlinePollUrl = null;
    onlineStatus.hidden = true;
    onlineReport.hidden = true;
    clearInterval(analyzeTimer);
    clearTimeout(analyzeFallbackTimer);
    analyzePanel.hidden = true;
    checkBtn.disabled = false;
    input.focus();
  });

  /* 在线体检失败/超时后的「重新提交」：按当前输入重新走一遍双模式判定 */
  onlineRetryBtn.addEventListener("click", function () {
    runCheck();
  });

  /* 复制报告文本（在线）：pre 块 textContent 原样复制，与页面所见逐字一致 */
  copyOnlineBtn.addEventListener("click", function () {
    copyReportText(onlineReportPre.textContent, copyOnlineBtn);
  });

  /* 页脚只读提示：当前生效的体检服务地址（textContent 写入，不解析链接） */
  apiNote.textContent = "当前体检服务地址：" + apiBase;

  renderSamples();
  updateCharCount();
})();
