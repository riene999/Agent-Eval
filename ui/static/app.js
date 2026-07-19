const state = {
  overview: null,
  datasets: [],
  models: [],
  reports: [],
  traces: [],
  proxyOnline: false,
};

const pageMeta = {
  overview: ["评测总览", "集中查看评测结果、运行状态与失败分布"],
  datasets: ["数据集", "管理当前项目已经接入的评测任务资源"],
  models: ["模型配置", "查看 models.json 中可供评测选择的模型路由"],
  launch: ["发起评测", "选择 Agent、模型与任务范围，启动一次批量评测"],
  jobs: ["任务记录", "查看由当前控制台启动的后台评测任务"],
  reports: ["评测报告", "浏览历史报告并下钻到题目与试验结果"],
  traces: ["轨迹详情", "按时序检查模型调用、工具调用和失败归因"],
};

const categoryNames = {
  constraint_forgetting: "遗忘约束",
  missing_step: "遗漏必要步骤",
  misread_tool_output: "误读工具结果",
  policy_violation: "违反业务规则",
  wrong_tool: "工具选择错误",
  param_hallucination: "参数编造",
  planning_error: "规划错误",
  premature_stop: "过早结束",
  loop_call: "循环调用",
  unknown: "其他错误",
};

const eventNames = {
  llm_call: "模型调用",
  tool_call: "工具调用",
  tool_return: "工具返回",
  final_output: "最终结果",
  llm_judge: "模型辅助评分",
  attribution: "错误归因",
};

function $(selector) { return document.querySelector(selector); }
function $$(selector) { return [...document.querySelectorAll(selector)]; }

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch (_) { /* 保留状态文本 */ }
    throw new Error(message);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

function formatPct(value, digits = 1) {
  return value === null || value === undefined ? "-" : `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatNumber(value) {
  if (value === null || value === undefined) return "-";
  return Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 1 });
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", { hour12: false });
}

function statusHtml(value) {
  if (value === true || value === "可用" || value === "已完成") return `<span class="status success">${escapeHtml(value === true ? "成功" : value)}</span>`;
  if (value === false || value === "失败" || value === "未安装") return `<span class="status failed">${escapeHtml(value === false ? "失败" : value)}</span>`;
  if (value === "运行中") return `<span class="status running">运行中</span>`;
  return `<span class="status warning">${escapeHtml(value ?? "未知")}</span>`;
}

let toastTimer = null;
function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.className = "toast"; }, 3600);
}

function navigate(page) {
  $$(".page").forEach(node => node.classList.toggle("active", node.id === `page-${page}`));
  $$(".nav-item").forEach(node => node.classList.toggle("active", node.dataset.page === page));
  $("#pageTitle").textContent = pageMeta[page][0];
  $("#pageDescription").textContent = pageMeta[page][1];
  document.title = `${pageMeta[page][0]} · Agent-Eval`;
  $(".sidebar").classList.remove("open");
  if (page === "jobs") loadJobs();
  if (page === "traces" && !state.traces.length) loadTraces();
}

function openModal(eyebrow, title, content) {
  $("#modalEyebrow").textContent = eyebrow;
  $("#modalTitle").textContent = title;
  $("#modalBody").innerHTML = content;
  $("#modalBackdrop").classList.add("open");
}

function closeModal() { $("#modalBackdrop").classList.remove("open"); }

async function checkProxy() {
  try {
    const health = await api("/api/proxy-health");
    state.proxyOnline = health.online;
  } catch (_) {
    state.proxyOnline = false;
  }
  const status = $("#proxyStatus");
  status.className = `status-pill ${state.proxyOnline ? "online" : "offline"}`;
  status.innerHTML = `<i></i>${state.proxyOnline ? "模型代理在线" : "模型代理离线"}`;
  $("#launchProxyDot").className = `dot ${state.proxyOnline ? "online" : "offline"}`;
  $("#launchProxyText").textContent = state.proxyOnline ? "模型代理已连接，可以开始评测" : "模型代理未启动，评测会立即失败";
}

async function loadOverview() {
  state.overview = await api("/api/overview");
  const { counts, latest_report: latest, recent_reports: recent } = state.overview;
  const cards = [
    ["✓", "最近准确率", latest ? formatPct(latest.accuracy) : "-", latest ? `${latest.agent_id} · ${latest.model}` : "暂无报告"],
    ["▥", "历史报告", formatNumber(counts.reports), "reports 目录中的 JSON 报告"],
    ["⑂", "已记录轨迹", formatNumber(counts.traces), "完整时序 JSONL 文件"],
    ["◫", "可选模型", formatNumber(counts.models), "models.json 已配置路由"],
  ];
  $("#overviewMetrics").innerHTML = cards.map(card => `
    <article class="metric-card"><span class="metric-icon">${card[0]}</span><div><small>${card[1]}</small><strong>${card[2]}</strong><p>${escapeHtml(card[3])}</p></div></article>
  `).join("");

  renderLatestReport(latest);
  renderFailureChart(latest?.failure_distribution || {}, "#failureChart");
  $("#recentReportsBody").innerHTML = recent.length ? recent.map(reportRow).join("") : emptyRow(7, "暂无评测报告");
}

function renderLatestReport(report) {
  if (!report) {
    $("#latestReport").className = "empty-state";
    $("#latestReport").textContent = "暂无报告，先发起一次评测";
    return;
  }
  const pct = Math.max(0, Math.min(100, Number(report.accuracy || 0) * 100));
  const circumference = 339.292;
  const offset = circumference * (1 - pct / 100);
  $("#latestReport").className = "latest-content";
  $("#latestReport").innerHTML = `
    <div class="score-ring">
      <svg viewBox="0 0 120 120" aria-label="准确率 ${pct.toFixed(1)}%"><circle class="track" cx="60" cy="60" r="54"></circle><circle class="value" cx="60" cy="60" r="54" stroke-dasharray="${circumference}" stroke-dashoffset="${offset}"></circle></svg>
      <div><strong>${pct.toFixed(1)}%</strong><small>准确率</small></div>
    </div>
    <div class="latest-meta">
      <h3>${escapeHtml(report.run_id)}<span>${escapeHtml(report.agent_id)} · ${escapeHtml(report.model)}</span></h3>
      <div class="mini-metrics">
        <div><small>题目数</small><strong>${formatNumber(report.task_count)}</strong></div>
        <div><small>同题次数</small><strong>${formatNumber(report.trials)}</strong></div>
        <div><small>pass@N</small><strong>${formatPct(report.pass_at_n)}</strong></div>
        <div><small>平均 Token</small><strong>${formatNumber(report.avg_total_tokens)}</strong></div>
        <div><small>平均工具调用</small><strong>${formatNumber(report.avg_tool_calls)}</strong></div>
        <div><small>报告类型</small><strong>${escapeHtml(report.kind)}</strong></div>
      </div>
    </div>`;
}

function renderFailureChart(distribution, selector) {
  const node = $(selector);
  const entries = Object.entries(distribution || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    node.className = "bar-chart empty-state";
    node.textContent = "暂无归因数据";
    return;
  }
  const max = Math.max(...entries.map(([, count]) => count));
  node.className = "bar-chart";
  node.innerHTML = entries.slice(0, 6).map(([name, count]) => `
    <div class="bar-row"><label title="${escapeHtml(categoryNames[name] || name)}">${escapeHtml(categoryNames[name] || name)}</label><div class="bar-track"><div class="bar-value" style="width:${Math.max(5, count / max * 100)}%"></div></div><b>${count}</b></div>
  `).join("");
}

function reportRow(report) {
  const failures = Object.values(report.failure_distribution || {}).reduce((sum, n) => sum + Number(n), 0);
  return `<tr>
    <td><span class="primary-text">${escapeHtml(report.run_id)}</span></td>
    <td>${escapeHtml(report.agent_id)}<span class="subtext">${escapeHtml(report.model)}</span></td>
    <td>${formatNumber(report.task_count)} 题${report.trials > 1 ? ` × ${report.trials}` : ""}</td>
    <td><strong>${formatPct(report.accuracy)}</strong></td>
    <td>${escapeHtml(report.kind)}</td>
    <td>${formatTime(report.timestamp)}</td>
    <td><button class="text-button report-open" data-run="${escapeHtml(report.run_id)}">查看</button></td>
  </tr>`;
}

function emptyRow(columns, message) { return `<tr><td colspan="${columns}"><div class="empty-state">${escapeHtml(message)}</div></td></tr>`; }

async function loadDatasets() {
  state.datasets = await api("/api/datasets");
  renderDatasets();
}

function renderDatasets() {
  const query = $("#datasetSearch").value.trim().toLowerCase();
  const status = $("#datasetStatus").value;
  const rows = state.datasets.filter(item => (!query || `${item.name} ${item.id}`.toLowerCase().includes(query)) && (!status || item.status === status));
  $("#datasetCount").textContent = `共 ${rows.length} 个`;
  $("#datasetsBody").innerHTML = rows.length ? rows.map(item => `
    <tr><td><span class="primary-text">${escapeHtml(item.name)}</span><span class="subtext">${escapeHtml(item.description)}</span></td>
    <td>${Object.entries(item.splits).map(([name, count]) => `<span class="tag">${escapeHtml(name)} ${count}</span>`).join("") || "-"}</td>
    <td>${formatNumber(item.task_count)}</td><td>${formatNumber(item.document_count)}</td><td>${formatNumber(item.tool_count)}</td><td>${statusHtml(item.status)}</td>
    <td><button class="text-button dataset-open" data-id="${item.id}">查看详情</button></td></tr>
  `).join("") : emptyRow(7, "没有符合条件的数据集");
}

function showDataset(id) {
  const item = state.datasets.find(dataset => dataset.id === id);
  if (!item) return;
  openModal("数据集详情", item.name, `<div class="dataset-detail">
    <div class="detail-row"><label>标识</label><strong>${escapeHtml(item.id)}</strong></div>
    <div class="detail-row"><label>说明</label><span>${escapeHtml(item.description)}</span></div>
    <div class="detail-row"><label>任务分组</label><span>${Object.entries(item.splits).map(([name, count]) => `<span class="tag">${escapeHtml(name)} ${count} 题</span>`).join("") || "-"}</span></div>
    <div class="detail-row"><label>覆盖领域</label><span>${item.domains.map(name => `<span class="tag purple">${escapeHtml(name)}</span>`).join("")}</span></div>
    <div class="detail-row"><label>资源规模</label><span>${item.task_count} 道题 · ${item.document_count} 份文档 · ${item.tool_count} 个工具</span></div>
    <div class="detail-row"><label>状态</label><span>${statusHtml(item.status)}</span></div>
  </div>`);
}

async function loadModels() {
  state.models = await api("/api/models");
  renderModels();
  const options = state.models.map(item => `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)} · ${escapeHtml(item.provider)}</option>`).join("");
  $("#runModel").innerHTML = options || `<option value="">models.json 未配置模型</option>`;
  $("#judgeModel").innerHTML = `<option value="">与被测模型相同</option>${options}`;
}

function renderModels() {
  const query = $("#modelSearch").value.trim().toLowerCase();
  const rows = state.models.filter(item => !query || `${item.name} ${item.provider} ${item.base_url}`.toLowerCase().includes(query));
  $("#modelCount").textContent = `共 ${rows.length} 个`;
  $("#modelsBody").innerHTML = rows.length ? rows.map(item => `
    <tr><td><span class="primary-text">${escapeHtml(item.name)}</span></td><td>${escapeHtml(item.provider)}</td><td><span class="tag">OpenAI 兼容 HTTP</span></td><td><span class="subtext" title="${escapeHtml(item.base_url)}">${escapeHtml(item.base_url)}</span></td><td>${statusHtml(item.configured ? "可用" : "配置不完整")}</td></tr>
  `).join("") : emptyRow(5, "没有符合条件的模型");
}

async function loadReports() {
  state.reports = await api("/api/reports");
  renderReports();
}

function renderReports() {
  const query = $("#reportSearch").value.trim().toLowerCase();
  const rows = state.reports.filter(item => !query || `${item.run_id} ${item.agent_id} ${item.model}`.toLowerCase().includes(query));
  $("#reportCount").textContent = `共 ${rows.length} 份`;
  $("#reportsBody").innerHTML = rows.length ? rows.map(item => {
    const failures = Object.values(item.failure_distribution || {}).reduce((sum, n) => sum + Number(n), 0);
    return `<tr><td><span class="primary-text">${escapeHtml(item.run_id)}</span><span class="subtext">${escapeHtml(item.kind)}</span></td><td>${escapeHtml(item.agent_id)}<span class="subtext">${escapeHtml(item.model)}</span></td><td>${item.task_count} 题${item.trials > 1 ? ` × ${item.trials}` : ""}</td><td><strong>${formatPct(item.accuracy)}</strong></td><td>${formatPct(item.pass_at_n)}</td><td>${failures ? `<span class="tag purple">${failures} 次归因</span>` : "-"}</td><td>${formatTime(item.timestamp)}</td><td><button class="text-button report-open" data-run="${escapeHtml(item.run_id)}">查看报告</button></td></tr>`;
  }).join("") : emptyRow(8, "没有符合条件的报告");
}

function flattenMetrics(report) {
  const tasks = report.tasks || [];
  if (tasks.some(item => Array.isArray(item.trials))) return tasks.flatMap(item => item.trials || []);
  return tasks;
}

function mean(values) {
  const usable = values.filter(value => value !== null && value !== undefined && !Number.isNaN(Number(value))).map(Number);
  return usable.length ? usable.reduce((sum, value) => sum + value, 0) / usable.length : null;
}

async function showReport(runId) {
  try {
    const payload = await api(`/api/reports/${encodeURIComponent(runId)}`);
    const report = payload.report;
    const summary = report.summary || {};
    const metrics = flattenMetrics(report);
    const accuracy = summary.avg_pass_at_1 ?? summary.accuracy ?? mean(metrics.map(item => item.accuracy));
    const passN = summary.avg_pass_at_n;
    const tokens = summary.avg_total_tokens ?? mean(metrics.map(item => item.total_tokens));
    const tools = summary.avg_tool_calls ?? mean(metrics.map(item => item.tool_call_count));
    const cost = summary.avg_cost_usd ?? mean(metrics.map(item => item.cost_usd));
    const latency = summary.latency_p95 ?? mean(metrics.map(item => item.latency_p95));
    const failures = summary.failure_distribution || {};
    const taskRows = (report.tasks || []).slice(0, 100).map(item => {
      const taskMetrics = item.trials || [item];
      const taskAccuracy = item.pass_at_1 ?? item.accuracy ?? mean(taskMetrics.map(row => row.accuracy));
      return `<tr><td>${escapeHtml(item.task_id || "-")}</td><td>${taskMetrics.length}</td><td><strong>${formatPct(taskAccuracy)}</strong></td><td>${formatNumber(mean(taskMetrics.map(row => row.total_tokens)))}</td><td>${formatNumber(mean(taskMetrics.map(row => row.tool_call_count)))}</td><td>${formatNumber(mean(taskMetrics.map(row => row.latency_p95)))} ms</td></tr>`;
    }).join("");
    openModal("评测报告", runId, `
      <div class="report-summary">
        <div><small>准确率 / pass@1</small><strong>${formatPct(accuracy)}</strong></div><div><small>pass@N</small><strong>${formatPct(passN)}</strong></div>
        <div><small>平均 Token</small><strong>${formatNumber(tokens)}</strong></div><div><small>平均工具调用</small><strong>${formatNumber(tools)}</strong></div>
        <div><small>平均成本</small><strong>${cost == null ? "-" : `$${Number(cost).toFixed(4)}`}</strong></div><div><small>延迟 p95</small><strong>${latency == null ? "-" : `${formatNumber(latency)} ms`}</strong></div>
        <div><small>Agent</small><strong>${escapeHtml(report.meta?.agent_id || "-")}</strong></div><div><small>模型</small><strong>${escapeHtml(report.meta?.model || "-")}</strong></div>
      </div>
      <h3 class="report-section-title">失败原因</h3><div id="modalFailureChart" class="bar-chart"></div>
      <h3 class="report-section-title">题目明细（最多展示 100 条）</h3>
      <div class="table-wrap"><table><thead><tr><th>任务</th><th>试验数</th><th>成功率</th><th>平均 Token</th><th>平均工具调用</th><th>平均 p95 延迟</th></tr></thead><tbody>${taskRows || emptyRow(6, "报告没有题目明细")}</tbody></table></div>
    `);
    renderFailureChart(failures, "#modalFailureChart");
  } catch (error) { toast(`读取报告失败：${error.message}`, true); }
}

async function loadJobs() {
  try {
    const jobs = await api("/api/jobs");
    $("#jobsBody").innerHTML = jobs.length ? jobs.map(job => `
      <tr><td><span class="primary-text">${escapeHtml(job.run_id)}</span></td><td>${escapeHtml(job.agent)}<span class="subtext">${escapeHtml(job.model)}</span></td><td>${escapeHtml(job.dataset)}</td><td>${job.count} 题 × ${job.trials}</td><td>${statusHtml(job.status)}</td><td>${formatTime(job.started_at)}</td><td><button class="text-button log-open" data-job="${escapeHtml(job.job_id)}">查看日志</button>${job.status === "已完成" ? `<button class="text-button report-open" data-run="${escapeHtml(job.run_id)}">报告</button>` : ""}</td></tr>
    `).join("") : emptyRow(7, "当前控制台尚未启动评测任务");
  } catch (error) { toast(`读取任务状态失败：${error.message}`, true); }
}

async function showLog(jobId) {
  try {
    const log = await api(`/api/jobs/${encodeURIComponent(jobId)}/log`);
    openModal("运行日志", jobId, `<pre class="log-view">${escapeHtml(log || "日志暂时为空")}</pre>`);
  } catch (error) { toast(`读取日志失败：${error.message}`, true); }
}

async function loadTraces() {
  const params = new URLSearchParams({
    agent: $("#traceAgent").value.trim(),
    task: $("#traceTask").value.trim(),
    run: $("#traceRun").value.trim(),
    result: $("#traceResult").value,
    limit: "200",
  });
  try {
    state.traces = await api(`/api/traces?${params}`);
    $("#traceCount").textContent = `共 ${state.traces.length} 条（最多显示 200 条）`;
    $("#tracesBody").innerHTML = state.traces.length ? state.traces.map(item => `
      <tr><td><span class="primary-text">${escapeHtml(item.task_id)}</span></td><td>${escapeHtml(item.agent_id)}</td><td>${escapeHtml(item.run_id)}</td><td>${item.event_count}</td><td>${statusHtml(item.success)}</td><td>${item.has_judge ? '<span class="tag purple">辅助评分</span>' : ""}${item.has_attribution ? '<span class="tag">错误归因</span>' : ""}${!item.has_judge && !item.has_attribution ? "-" : ""}</td><td>${formatTime(item.timestamp)}</td><td><button class="text-button trace-open" data-agent="${escapeHtml(item.agent_id)}" data-task="${escapeHtml(item.task_id)}" data-run="${escapeHtml(item.run_id)}">查看轨迹</button></td></tr>
    `).join("") : emptyRow(8, "没有符合条件的轨迹");
  } catch (error) { toast(`读取轨迹失败：${error.message}`, true); }
}

async function openTrace(agent, task, run) {
  try {
    const detail = await api(`/api/traces/${encodeURIComponent(agent)}/${encodeURIComponent(task)}/${encodeURIComponent(run)}`);
    $("#drawerTitle").textContent = task;
    $("#drawerSubtitle").textContent = `${agent} · ${run} · ${detail.events.length} 个事件`;
    $("#timeline").innerHTML = detail.events.length ? detail.events.map(event => `
      <article class="timeline-event ${escapeHtml(event.event_type)}" data-agent="${escapeHtml(agent)}" data-task="${escapeHtml(task)}" data-run="${escapeHtml(run)}" data-seq="${event.seq}">
        <span class="event-index">${event.seq}</span><div class="event-card"><header><strong>${escapeHtml(eventNames[event.event_type] || event.event_type)}</strong><time>${formatTime(event.timestamp)}</time></header><p>${escapeHtml(event.summary)}</p></div>
      </article>
    `).join("") : `<div class="empty-state">轨迹没有事件</div>`;
    $("#traceDrawer").classList.add("open");
    $("#traceDrawer").setAttribute("aria-hidden", "false");
    $("#drawerBackdrop").classList.add("open");
  } catch (error) { toast(`读取轨迹失败：${error.message}`, true); }
}

function closeDrawer() {
  $("#traceDrawer").classList.remove("open");
  $("#traceDrawer").setAttribute("aria-hidden", "true");
  $("#drawerBackdrop").classList.remove("open");
}

async function showEvent(agent, task, run, seq) {
  try {
    const event = await api(`/api/traces/${encodeURIComponent(agent)}/${encodeURIComponent(task)}/${encodeURIComponent(run)}/events/${seq}`);
    openModal("完整事件 · 大字段已从内容存储还原", `#${seq} ${eventNames[event.event_type] || event.event_type}`, `<pre class="json-view">${escapeHtml(JSON.stringify(event, null, 2))}</pre>`);
  } catch (error) { toast(`还原事件失败：${error.message}`, true); }
}

async function submitRun(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const valueOrNull = name => form.get(name) === "" ? null : Number(form.get(name));
  const payload = {
    agent: form.get("agent"), model: form.get("model"), dataset: form.get("dataset"), split: form.get("split"),
    start: Number(form.get("start")), count: Number(form.get("count")), trials: Number(form.get("trials")), concurrency: Number(form.get("concurrency")), temperature: Number(form.get("temperature")),
    seed: valueOrNull("seed"), run_id: form.get("run_id") || null, llm_judge: form.has("llm_judge"), attribution: form.has("attribution"), attribution_mode: form.get("attribution_mode"), judge_model: form.get("judge_model") || null,
  };
  const button = $("#runSubmit");
  button.disabled = true;
  button.textContent = "正在提交…";
  try {
    const job = await api("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    toast(`评测任务 ${job.run_id} 已启动`);
    navigate("jobs");
  } catch (error) {
    toast(`启动失败：${error.message}`, true);
  } finally {
    button.disabled = false;
    button.textContent = "开始评测";
  }
}

function bindEvents() {
  $$(".nav-item").forEach(node => node.addEventListener("click", () => navigate(node.dataset.page)));
  $$('[data-go]').forEach(node => node.addEventListener("click", () => navigate(node.dataset.go)));
  $("#menuButton").addEventListener("click", () => $(".sidebar").classList.toggle("open"));
  $("#refreshButton").addEventListener("click", refreshAll);
  $("#modalClose").addEventListener("click", closeModal);
  $("#modalBackdrop").addEventListener("click", event => { if (event.target === $("#modalBackdrop")) closeModal(); });
  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#drawerBackdrop").addEventListener("click", closeDrawer);

  $("#datasetQuery").addEventListener("click", renderDatasets);
  $("#datasetReset").addEventListener("click", () => { $("#datasetSearch").value = ""; $("#datasetStatus").value = ""; renderDatasets(); });
  $("#modelQuery").addEventListener("click", renderModels);
  $("#modelReset").addEventListener("click", () => { $("#modelSearch").value = ""; renderModels(); });
  $("#reportQuery").addEventListener("click", renderReports);
  $("#reportReset").addEventListener("click", () => { $("#reportSearch").value = ""; renderReports(); });
  $("#traceQuery").addEventListener("click", loadTraces);
  $("#traceReset").addEventListener("click", () => { $("#traceAgent").value = ""; $("#traceTask").value = ""; $("#traceRun").value = ""; $("#traceResult").value = "all"; loadTraces(); });
  $("#jobsRefresh").addEventListener("click", loadJobs);
  $("#runDataset").addEventListener("change", event => { $("#splitField").style.visibility = event.target.value === "tau_retail" ? "visible" : "hidden"; });
  $("#runForm").addEventListener("submit", submitRun);

  document.addEventListener("click", event => {
    const report = event.target.closest(".report-open");
    const dataset = event.target.closest(".dataset-open");
    const log = event.target.closest(".log-open");
    const trace = event.target.closest(".trace-open");
    const timeline = event.target.closest(".timeline-event");
    if (report) showReport(report.dataset.run);
    if (dataset) showDataset(dataset.dataset.id);
    if (log) showLog(log.dataset.job);
    if (trace) openTrace(trace.dataset.agent, trace.dataset.task, trace.dataset.run);
    if (timeline) showEvent(timeline.dataset.agent, timeline.dataset.task, timeline.dataset.run, timeline.dataset.seq);
  });
}

async function refreshAll() {
  const button = $("#refreshButton");
  button.disabled = true;
  try {
    await Promise.all([checkProxy(), loadOverview(), loadDatasets(), loadModels(), loadReports()]);
    toast("数据已刷新");
  } catch (error) {
    toast(`刷新失败：${error.message}`, true);
  } finally {
    button.disabled = false;
  }
}

async function init() {
  bindEvents();
  try {
    await Promise.all([checkProxy(), loadOverview(), loadDatasets(), loadModels(), loadReports()]);
  } catch (error) {
    toast(`页面初始化失败：${error.message}`, true);
  }
  setInterval(() => {
    if ($("#page-jobs").classList.contains("active")) loadJobs();
  }, 5000);
}

init();
