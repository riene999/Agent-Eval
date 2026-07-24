const state = {
  overview: null,
  datasets: [],
  models: [],
  skills: [],
  skillTools: [],
  reports: [],
  traces: [],
  traceChoices: [],
  exports: [],
  dpoRuns: [],
  currentTrace: null,
  replay: null,
  proxyOnline: false,
};

const pageMeta = {
  overview: ["评测总览", "集中查看评测结果、运行状态与失败分布"],
  datasets: ["数据集", "管理当前项目已经接入的评测任务资源"],
  models: ["模型配置", "查看 models.json 中可供评测选择的模型路由"],
  skills: ["Skill 管理", "导入和管理能力说明、业务规则与工具集合"],
  launch: ["发起评测", "选择 Agent、模型与任务范围，启动一次批量评测"],
  skill_eval: ["Skill 效果评测", "运行单 Skill 边界测试或 N+1 能力增益与回归测试"],
  jobs: ["任务记录", "查看由当前控制台启动的后台评测任务"],
  reports: ["评测报告", "浏览历史报告并下钻到题目与试验结果"],
  compare: ["评测对比", "比较两个版本的总体指标，并下钻查看两条轨迹从哪一步开始不同"],
  traces: ["轨迹详情", "按时序检查模型调用、工具调用和失败归因"],
  exports: ["训练数据导出", "将标准轨迹和失败轨迹整理为 SFT 或 DPO 数据"],
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
  skill_route: "Skill 路由",
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
    const newApiPrefixes = ["/api/exports", "/api/comparisons", "/api/trace-diff", "/api/replay", "/api/skills", "/api/skill-runs"];
    const deletingReport = String(options.method || "GET").toUpperCase() === "DELETE" && String(path).startsWith("/api/reports/");
    if ([404, 405].includes(response.status) && (deletingReport || newApiPrefixes.some(prefix => String(path).startsWith(prefix)))) {
      message = "当前可视化后端尚未加载新接口，请重启 ui.server 后再刷新浏览器";
    }
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
  if (page === "skills") loadSkills();
  if (page === "skill_eval") loadSkills();
  if (page === "traces" && !state.traces.length) loadTraces();
  if (page === "compare") prepareComparisonPage();
  if (page === "exports") {
    loadExports();
    loadDpoRunIds();
  }
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
  $("#skillEvalModel").innerHTML = options || `<option value="">models.json 未配置模型</option>`;
}

function renderModels() {
  const query = $("#modelSearch").value.trim().toLowerCase();
  const rows = state.models.filter(item => !query || `${item.name} ${item.provider} ${item.base_url}`.toLowerCase().includes(query));
  $("#modelCount").textContent = `共 ${rows.length} 个`;
  $("#modelsBody").innerHTML = rows.length ? rows.map(item => `
    <tr><td><span class="primary-text">${escapeHtml(item.name)}</span></td><td>${escapeHtml(item.provider)}</td><td><span class="tag">OpenAI 兼容 HTTP</span></td><td><span class="subtext" title="${escapeHtml(item.base_url)}">${escapeHtml(item.base_url)}</span></td><td>${statusHtml(item.configured ? "可用" : "配置不完整")}</td></tr>
  `).join("") : emptyRow(5, "没有符合条件的模型");
}

async function loadSkills() {
  try {
    const payload = await api("/api/skills");
    state.skills = payload.skills || [];
    state.skillTools = payload.available_tools || [];
    renderSkills();
    renderSkillEvalOptions();
  } catch (error) {
    toast(`读取 Skill 失败：${error.message}`, true);
  }
}

function renderSkills() {
  const query = $("#skillSearch").value.trim().toLowerCase();
  const rows = state.skills.filter(item => !query || `${item.skill_id} ${item.name} ${item.domains.join(" ")}`.toLowerCase().includes(query));
  $("#skillCount").textContent = `共 ${rows.length} 个`;
  $("#skillsBody").innerHTML = rows.length ? rows.map(item => `
    <tr>
      <td><span class="primary-text">${escapeHtml(item.name)}</span><span class="subtext">${escapeHtml(item.skill_id)} · ${escapeHtml(item.description)}</span></td>
      <td>${item.domains.map(domain => `<span class="tag purple">${escapeHtml(domain)}</span>`).join("") || "-"}</td>
      <td>${escapeHtml(item.version)}</td>
      <td>${item.tool_count}</td>
      <td>${statusHtml(item.enabled ? "可用" : "已停用")}</td>
      <td><button class="text-button skill-open" data-skill="${escapeHtml(item.skill_id)}">查看详情</button></td>
    </tr>
  `).join("") : emptyRow(6, "没有符合条件的 Skill");
}

function renderSkillEvalOptions() {
  const active = state.skills.filter(item => item.enabled);
  const current = $("#skillEvalTarget").value;
  $("#skillEvalTarget").innerHTML = active.map(item => `<option value="${escapeHtml(item.skill_id)}">${escapeHtml(item.name)} · ${escapeHtml(item.skill_id)}</option>`).join("") || `<option value="">请先导入 Skill</option>`;
  if (active.some(item => item.skill_id === current)) $("#skillEvalTarget").value = current;
  const target = $("#skillEvalTarget").value;
  $("#skillBaselinePicker").innerHTML = active.filter(item => item.skill_id !== target).map(item => `
    <label class="skill-check"><input type="checkbox" name="baseline_skill" value="${escapeHtml(item.skill_id)}"><span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.skill_id)}</small></span></label>
  `).join("") || `<div class="picker-empty">没有其他可用 Skill</div>`;
}

async function showSkill(skillId) {
  try {
    const item = await api(`/api/skills/${encodeURIComponent(skillId)}`);
    openModal("Skill 详情", item.name, `<div class="dataset-detail">
      <div class="detail-row"><label>标识 / 版本</label><strong>${escapeHtml(item.skill_id)} · ${escapeHtml(item.version)}</strong></div>
      <div class="detail-row"><label>能力说明</label><span>${escapeHtml(item.description)}</span></div>
      <div class="detail-row"><label>适用业务域</label><span>${item.domains.map(domain => `<span class="tag purple">${escapeHtml(domain)}</span>`).join("") || "-"}</span></div>
      <div class="detail-row"><label>使用规则</label><span>${escapeHtml(item.instructions)}</span></div>
      <div class="detail-row"><label>工具集合</label><span>${item.tools.map(tool => `<span class="tag">${escapeHtml(tool)}</span>`).join("")}</span></div>
    </div>`);
  } catch (error) {
    toast(`读取 Skill 详情失败：${error.message}`, true);
  }
}

function showSkillTemplate() {
  const sample = {
    skill_id: "example_skill",
    name: "示例能力",
    description: "说明这个 Skill 能处理哪些问题。",
    instructions: "写清楚工具调用顺序、业务限制和不能做的事情。",
    tools: state.skillTools.slice(0, 3),
    domains: ["hr"],
    version: "1.0.0",
    enabled: true,
  };
  openModal("Skill JSON 导入格式", "声明式能力包", `
    <p class="modal-note">tools 只能填写项目已注册工具。当前可用：${state.skillTools.map(tool => `<code>${escapeHtml(tool)}</code>`).join(" ")}</p>
    <pre class="json-view">${escapeHtml(JSON.stringify(sample, null, 2))}</pre>`);
}

async function previewSkillFile(file) {
  if (!file) return;
  try {
    const skill = JSON.parse(await file.text());
    openModal("导入 Skill", skill.name || file.name, `
      <div class="import-preview"><p>确认将这个能力包保存到项目的 <code>data/skills</code>。</p><pre class="json-view">${escapeHtml(JSON.stringify(skill, null, 2))}</pre></div>
      <label class="plain-check"><input type="checkbox" id="skillOverwrite">同名 Skill 已存在时覆盖</label>
      <div class="confirm-actions"><button type="button" class="button secondary" id="skillImportCancel">取消</button><button type="button" class="button primary" id="skillImportConfirm">确认导入</button></div>`);
    $("#skillImportCancel").addEventListener("click", closeModal);
    $("#skillImportConfirm").addEventListener("click", () => submitSkillImport(skill));
  } catch (error) {
    toast(`读取 Skill 文件失败：${error.message}`, true);
  } finally {
    $("#skillFile").value = "";
  }
}

async function submitSkillImport(skill) {
  const button = $("#skillImportConfirm");
  button.disabled = true;
  button.textContent = "正在导入…";
  try {
    const result = await api("/api/skills", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skill, overwrite: $("#skillOverwrite").checked }),
    });
    closeModal();
    await loadSkills();
    toast(`Skill ${result.name} 已导入`);
  } catch (error) {
    button.disabled = false;
    button.textContent = "确认导入";
    toast(`导入失败：${error.message}`, true);
  }
}

function updateSkillEvalMode() {
  const mode = document.querySelector('input[name="skill_eval_mode"]:checked').value;
  $("#skillBaselineField").classList.toggle("hidden", mode !== "n_plus_one");
}

async function submitSkillEvaluation(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const mode = document.querySelector('input[name="skill_eval_mode"]:checked').value;
  const baseline = $$("input[name='baseline_skill']:checked").map(node => node.value);
  if (mode === "n_plus_one" && !baseline.length) {
    toast("N+1 评测请至少勾选一个原有 Skill", true);
    return;
  }
  const numberOrNull = name => form.get(name) === "" ? null : Number(form.get(name));
  const payload = {
    mode,
    agent: $("#skillEvalAgent").value,
    skill: $("#skillEvalTarget").value,
    baseline_skills: baseline,
    model: $("#skillEvalModel").value,
    run_id: form.get("run_id") || null,
    start: Number(form.get("start")),
    count: Number(form.get("count")),
    trials: Number(form.get("trials")),
    concurrency: Number(form.get("concurrency")),
    temperature: Number(form.get("temperature")),
    seed: numberOrNull("seed"),
  };
  const button = $("#skillEvalSubmit");
  button.disabled = true;
  button.textContent = "正在提交…";
  try {
    const job = await api("/api/skill-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    toast(`Skill 评测 ${job.run_id} 已启动`);
    navigate("jobs");
  } catch (error) {
    toast(`启动 Skill 评测失败：${error.message}`, true);
  } finally {
    button.disabled = false;
    button.textContent = "开始 Skill 评测";
  }
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
    return `<tr><td><span class="primary-text">${escapeHtml(item.run_id)}</span><span class="subtext">${escapeHtml(item.kind)}</span></td><td>${escapeHtml(item.agent_id)}<span class="subtext">${escapeHtml(item.model)}</span></td><td>${item.task_count} 题${item.trials > 1 ? ` × ${item.trials}` : ""}</td><td><strong>${formatPct(item.accuracy)}</strong></td><td>${formatPct(item.pass_at_n)}</td><td>${failures ? `<span class="tag purple">${failures} 次归因</span>` : "-"}</td><td>${formatTime(item.timestamp)}</td><td><div class="table-actions"><button type="button" class="text-button report-open" data-run="${escapeHtml(item.run_id)}">查看报告</button><button type="button" class="text-button danger-text report-delete" data-run="${escapeHtml(item.run_id)}">删除</button></div></td></tr>`;
  }).join("") : emptyRow(8, "没有符合条件的报告");
}

function confirmDeleteReport(runId) {
  const report = state.reports.find(item => item.run_id === runId);
  openModal("危险操作 · 二次确认", "删除评测报告", `
    <div class="delete-confirmation">
      <div class="delete-warning-icon">!</div>
      <div><strong>确定删除报告 ${escapeHtml(runId)} 吗？</strong><p>将真实删除该报告的 JSON 和 Markdown 文件。原始轨迹不会删除，模型调用记录和工具调用记录仍会保留。</p></div>
    </div>
    <div class="delete-summary">
      <div><label>运行标识</label><strong>${escapeHtml(runId)}</strong></div>
      <div><label>Agent / 模型</label><span>${escapeHtml(report?.agent_id || "-")} · ${escapeHtml(report?.model || "-")}</span></div>
      <div><label>删除后果</label><span>报告列表、总览和评测对比中不再出现</span></div>
    </div>
    <div class="confirm-actions"><button type="button" class="button secondary" id="deleteCancel">取消</button><button type="button" class="button danger" id="deleteConfirm">确认删除</button></div>
  `);
  $("#deleteCancel").addEventListener("click", closeModal);
  $("#deleteConfirm").addEventListener("click", () => deleteReport(runId));
}

async function deleteReport(runId) {
  const button = $("#deleteConfirm");
  button.disabled = true;
  button.textContent = "正在删除…";
  try {
    await api(`/api/reports/${encodeURIComponent(runId)}`, { method: "DELETE" });
    closeModal();
    await Promise.all([loadReports(), loadOverview(), loadJobs()]);
    toast(`报告 ${runId} 已删除，原始轨迹仍保留`);
  } catch (error) {
    button.disabled = false;
    button.textContent = "确认删除";
    toast(`删除报告失败：${error.message}`, true);
  }
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
    let skillCards = "";
    if (summary.routing_accuracy != null) {
      skillCards = `
        <h3 class="report-section-title">Skill 效果（旧版路由数据）</h3>
        <div class="report-summary">
          <div><small>路由准确率</small><strong>${formatPct(summary.routing_accuracy)}</strong></div>
          <div><small>范围外识别率</small><strong>${formatPct(summary.boundary_accuracy)}</strong></div>
          <div><small>Skill 混淆率</small><strong>${formatPct(summary.skill_confusion_rate)}</strong></div>
          <div><small>跨 Skill 工具调用</small><strong>${formatPct(summary.cross_skill_tool_rate)}</strong></div>
        </div>`;
    } else if (summary.skill_task_count != null) {
      skillCards = `
        <h3 class="report-section-title">Skill 提示词效果</h3>
        <div class="report-summary">
          <div><small>提示词命中任务</small><strong>${summary.in_scope_count ?? 0}</strong></div>
          <div><small>命中任务成功率</small><strong>${formatPct(summary.in_scope_success_rate)}</strong></div>
          <div><small>其他任务</small><strong>${summary.out_of_scope_count ?? 0}</strong></div>
          <div><small>其他任务成功率</small><strong>${formatPct(summary.out_of_scope_success_rate)}</strong></div>
        </div>`;
    }
    const skillComparison = report.skill_evaluation;
    const nPlusCards = !skillComparison ? "" : `
      <h3 class="report-section-title">N+1 能力变化</h3>
      <div class="report-summary">
        <div><small>新增后做对</small><strong class="positive-number">+${skillComparison.gained_tasks.length}</strong></div>
        <div><small>旧能力退化</small><strong class="negative-number">-${skillComparison.regressed_tasks.length}</strong></div>
        <div><small>净收益</small><strong>${skillComparison.net_gain >= 0 ? "+" : ""}${skillComparison.net_gain}</strong></div>
        <div><small>配对题目</small><strong>${skillComparison.paired_task_count}</strong></div>
      </div>`;
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
      ${skillCards}${nPlusCards}
      <h3 class="report-section-title">失败原因</h3><div id="modalFailureChart" class="bar-chart"></div>
      <h3 class="report-section-title">题目明细（最多展示 100 条）</h3>
      <div class="table-wrap"><table><thead><tr><th>任务</th><th>试验数</th><th>成功率</th><th>平均 Token</th><th>平均工具调用</th><th>平均 p95 延迟</th></tr></thead><tbody>${taskRows || emptyRow(6, "报告没有题目明细")}</tbody></table></div>
    `);
    renderFailureChart(failures, "#modalFailureChart");
  } catch (error) { toast(`读取报告失败：${error.message}`, true); }
}

const comparisonMetricNames = {
  accuracy: "准确率", total_tokens: "平均 Token", tool_call_count: "平均工具调用",
  redundant_call_rate: "重复调用比例", latency_p95: "p95 延迟", cost_usd: "平均成本",
  tool_selection: "工具选择正确率", arg_correctness: "参数正确率",
};

function reportOption(report) {
  return `<option value="${escapeHtml(report.run_id)}">${escapeHtml(report.run_id)} · ${escapeHtml(report.agent_id)} · ${escapeHtml(report.model)}</option>`;
}

function traceOption(trace) {
  const value = `${trace.agent_id}|${trace.task_id}|${trace.run_id}`;
  return `<option value="${escapeHtml(value)}">${escapeHtml(trace.task_id)} · ${escapeHtml(trace.agent_id)} · ${escapeHtml(trace.run_id)} · ${trace.success ? "成功" : "失败"}</option>`;
}

async function prepareComparisonPage() {
  if (!state.reports.length) await loadReports();
  const options = state.reports.map(reportOption).join("");
  $("#baselineReport").innerHTML = options || `<option value="">没有可比较的报告</option>`;
  $("#candidateReport").innerHTML = options || `<option value="">没有可比较的报告</option>`;
  if (state.reports.length > 1) {
    $("#baselineReport").value = state.reports[1].run_id;
    $("#candidateReport").value = state.reports[0].run_id;
  }
  try {
    state.traceChoices = await api("/api/traces?limit=2000");
    const traceOptions = state.traceChoices.map(traceOption).join("");
    $("#baselineTrace").innerHTML = traceOptions || `<option value="">暂无轨迹</option>`;
    $("#candidateTrace").innerHTML = traceOptions || `<option value="">暂无轨迹</option>`;
    if (state.traceChoices.length > 1) $("#candidateTrace").selectedIndex = 1;
  } catch (error) { toast(`读取轨迹列表失败：${error.message}`, true); }
}

function comparisonValue(name, value) {
  if (value === null || value === undefined) return "-";
  if (["accuracy", "redundant_call_rate", "tool_selection", "arg_correctness"].includes(name)) return formatPct(value);
  if (name === "cost_usd") return `$${Number(value).toFixed(4)}`;
  if (name === "latency_p95") return `${formatNumber(value)} ms`;
  return formatNumber(value);
}

function changeValue(name, value, relative = false) {
  if (value === null || value === undefined) return "-";
  if (relative || ["accuracy", "redundant_call_rate", "tool_selection", "arg_correctness"].includes(name)) return `${value >= 0 ? "+" : ""}${formatPct(value)}`;
  if (name === "cost_usd") return `${value >= 0 ? "+" : ""}$${Number(value).toFixed(4)}`;
  if (name === "latency_p95") return `${value >= 0 ? "+" : ""}${formatNumber(value)} ms`;
  return `${value >= 0 ? "+" : ""}${formatNumber(value)}`;
}

async function submitComparison(event) {
  event.preventDefault();
  const payload = {
    baseline_run_id: $("#baselineReport").value,
    candidate_run_id: $("#candidateReport").value,
    accuracy_drop_max: Number($("#gateAccuracy").value) / 100,
    token_increase_max: Number($("#gateTokens").value) / 100,
    latency_p95_increase_max: Number($("#gateLatency").value) / 100,
    cost_increase_max: Number($("#gateCost").value) / 100,
    redundant_call_rate_increase_max: Number($("#gateRedundant").value) / 100,
  };
  const button = $("#compareSubmit");
  button.disabled = true;
  button.textContent = "正在对比…";
  try {
    const result = await api("/api/comparisons", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    renderComparison(result);
    toast(`对比完成：回归门禁${result.status === "passed" ? "通过" : "未通过"}`);
  } catch (error) { toast(`对比失败：${error.message}`, true); }
  finally { button.disabled = false; button.textContent = "开始对比"; }
}

function renderComparison(result) {
  const passed = result.status === "passed";
  $("#pairedCount").textContent = `共同题目 ${result.paired_task_count} 道`;
  $("#compareResult").className = "gate-result-content";
  $("#compareResult").innerHTML = `
    <div class="gate-banner ${passed ? "passed" : "failed"}"><span>${passed ? "✓" : "!"}</span><div><strong>${passed ? "回归门禁通过" : "发现指标退化"}</strong><p>${escapeHtml(result.baseline.run_id)} → ${escapeHtml(result.candidate.run_id)} · 配对 ${result.paired_task_count} 道题</p></div></div>
    <div class="gate-checks">${result.checks.map(check => `<div><span class="gate-dot ${check.status}"></span><label>${escapeHtml(check.label)}</label><strong>${check.value == null ? "无数据，跳过" : changeValue(check.metric, check.value, check.value_kind === "相对变化")}</strong></div>`).join("")}</div>`;
  const rows = Object.entries(result.metrics).filter(([, metric]) => metric.baseline !== null).map(([name, metric]) => {
    const interval = metric.delta_ci95 ? `${changeValue(name, metric.delta_ci95[0])} ～ ${changeValue(name, metric.delta_ci95[1])}` : "-";
    return `<tr><td><strong>${escapeHtml(comparisonMetricNames[name] || name)}</strong></td><td>${comparisonValue(name, metric.baseline)}</td><td>${comparisonValue(name, metric.candidate)}</td><td>${changeValue(name, metric.delta)}</td><td>${changeValue(name, metric.relative_change, true)}</td><td>${interval}</td></tr>`;
  }).join("");
  $("#comparisonMetricsBody").innerHTML = rows || emptyRow(6, "共同题目中没有可比较指标");
}

function parseTraceRef(value) {
  const [agent_id, task_id, run_id] = value.split("|");
  return { agent_id, task_id, run_id };
}

function actionText(action) {
  if (!action) return "—";
  return `${escapeHtml(action.tool_name)}<span class="subtext code-subtext">${escapeHtml(JSON.stringify(action.args || {}))}</span>`;
}

async function submitTraceDiff(event) {
  event.preventDefault();
  try {
    const result = await api("/api/trace-diff", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ baseline: parseTraceRef($("#baselineTrace").value), candidate: parseTraceRef($("#candidateTrace").value) }) });
    const alignment = result.alignment;
    const labels = { match: "一致", argument_mismatch: "参数不同", wrong_tool: "工具不同", missing_call: "候选漏调", extra_call: "候选多调" };
    $("#traceDiffResult").className = "trace-diff-result";
    $("#traceDiffResult").innerHTML = `
      <div class="diff-summary"><div><small>动作相似度</small><strong>${formatPct(alignment.similarity)}</strong></div><div><small>第一次偏离</small><strong>${alignment.first_deviation_seq == null ? "无" : `第 ${alignment.first_deviation_seq} 步`}</strong></div><div><small>基线 / 候选调用数</small><strong>${alignment.baseline_action_count} / ${alignment.candidate_action_count}</strong></div></div>
      <div class="table-wrap"><table><thead><tr><th>#</th><th>判断</th><th>基线轨迹</th><th>候选轨迹</th></tr></thead><tbody>${alignment.operations.map((operation, index) => `<tr class="diff-${operation.type}"><td>${index + 1}</td><td><span class="diff-label ${operation.type}">${labels[operation.type]}</span></td><td>${actionText(operation.baseline)}</td><td>${actionText(operation.candidate)}</td></tr>`).join("") || emptyRow(4, "两条轨迹都没有工具调用")}</tbody></table></div>`;
  } catch (error) { toast(`轨迹对比失败：${error.message}`, true); }
}

async function loadJobs() {
  try {
    const jobs = await api("/api/jobs");
    $("#jobsBody").innerHTML = jobs.length ? jobs.map(job => `
      <tr><td><span class="primary-text">${escapeHtml(job.run_id)}</span></td><td>${escapeHtml(job.agent)}<span class="subtext">${escapeHtml(job.model)}</span></td><td>${escapeHtml(job.dataset)}</td><td>${job.count} 题 × ${job.trials}</td><td>${statusHtml(job.status)}</td><td>${formatTime(job.started_at)}</td><td><button class="text-button log-open" data-job="${escapeHtml(job.job_id)}">查看日志</button>${job.status === "已完成" && job.report_exists ? `<button class="text-button report-open" data-run="${escapeHtml(job.run_id)}">报告</button>` : ""}</td></tr>
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
    state.currentTrace = { agent_id: agent, task_id: task, run_id: run };
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

async function startReplay() {
  if (!state.currentTrace) return;
  try {
    const payload = await api("/api/replay", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.currentTrace) });
    state.replay = { payload, index: 0 };
    renderReplay();
  } catch (error) { toast(`轨迹回放失败：${error.message}`, true); }
}

function replayDataPreview(event) {
  const data = event.data || {};
  if (event.event_type === "llm_call") return { model: data.model, request: data.request, response: data.response, prompt_tokens: data.prompt_tokens, completion_tokens: data.completion_tokens, latency_ms: data.latency_ms };
  return data;
}

function renderReplay() {
  if (!state.replay) return;
  const { payload, index } = state.replay;
  const event = payload.events[index];
  const total = payload.events.length;
  openModal("离线轨迹回放 · 不会重新调用模型或工具", payload.trace.task_id, `
    <div class="replay-head"><div><small>进度</small><strong>${index + 1} / ${total}</strong></div><div class="replay-progress"><span style="width:${total ? (index + 1) / total * 100 : 0}%"></span></div><div class="replay-nav"><button class="button secondary small" id="replayPrev" ${index === 0 ? "disabled" : ""}>上一步</button><button class="button primary small" id="replayNext" ${index >= total - 1 ? "disabled" : ""}>下一步</button></div></div>
    <article class="replay-event"><header><span class="event-index">${event.seq}</span><div><small>${formatTime(event.timestamp)}</small><h3>${escapeHtml(eventNames[event.event_type] || event.event_type)}</h3></div></header><pre class="json-view">${escapeHtml(JSON.stringify(replayDataPreview(event), null, 2))}</pre></article>
  `);
  $("#replayPrev").addEventListener("click", () => { state.replay.index -= 1; renderReplay(); });
  $("#replayNext").addEventListener("click", () => { state.replay.index += 1; renderReplay(); });
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function loadExports() {
  try {
    state.exports = await api("/api/exports");
    $("#exportsBody").innerHTML = state.exports.length ? state.exports.map(item => `<tr><td><span class="primary-text">${escapeHtml(item.filename)}</span></td><td>${formatNumber(item.rows)}</td><td>${formatBytes(item.size_bytes)}</td><td>${formatTime(item.updated_at)}</td><td><a class="text-button link-button" href="${escapeHtml(item.download_url)}">下载</a></td></tr>`).join("") : emptyRow(5, "还没有导出训练数据");
  } catch (error) { toast(`读取导出文件失败：${error.message}`, true); }
}

function selectedDpoRunIds() {
  return $$("input[name='dpo_run_id']:checked").map(node => node.value);
}

async function loadDpoRunIds() {
  const picker = $("#dpoRunPicker");
  const selected = new Set(selectedDpoRunIds());
  const agent = $("#exportAgent").value.trim();
  if (!agent) {
    picker.innerHTML = `<div class="picker-empty">请先填写 Agent 标识，再刷新轨迹。</div>`;
    return;
  }
  picker.innerHTML = `<div class="picker-empty">正在读取已有轨迹…</div>`;
  try {
    state.dpoRuns = await api(`/api/dpo-run-ids?agent=${encodeURIComponent(agent)}`);
    picker.innerHTML = state.dpoRuns.length ? state.dpoRuns.map(item => `
      <label class="run-choice">
        <input type="checkbox" name="dpo_run_id" value="${escapeHtml(item.run_id)}" ${selected.has(item.run_id) ? "checked" : ""}>
        <span><strong>${escapeHtml(item.run_id)}</strong><small>${item.task_count} 题 · ${item.trajectory_count} 条轨迹 · 成功 ${item.success_count} · 更新于 ${formatTime(item.updated_at)}</small></span>
        <span class="run-meta">失败 ${item.failed_count}</span>
      </label>
    `).join("") : `<div class="picker-empty">当前 Agent 下没有企业知识问答轨迹。DPO 导出只展示带标准轨迹的 enterprise_kb 运行。</div>`;
  } catch (error) {
    picker.innerHTML = `<div class="picker-empty">读取轨迹失败：${escapeHtml(error.message)}</div>`;
  }
}

function updateExportKind() {
  const kind = document.querySelector('input[name="export_kind"]:checked').value;
  $$(".dpo-only").forEach(node => node.classList.toggle("hidden", kind !== "dpo"));
  const filename = $("#exportFilename");
  if (kind === "sft" && (filename.value.includes("dpo") || !filename.value)) filename.value = "enterprise_sft.jsonl";
  if (kind === "dpo" && (filename.value.includes("sft") || !filename.value)) filename.value = "enterprise_dpo.jsonl";
  if (kind === "dpo") loadDpoRunIds();
}

async function submitExport(event) {
  event.preventDefault();
  const kind = document.querySelector('input[name="export_kind"]:checked').value;
  const runIds = selectedDpoRunIds();
  if (kind === "dpo" && !runIds.length) {
    toast("请至少勾选一个评测运行标识", true);
    return;
  }
  const payload = { kind, filename: $("#exportFilename").value.trim(), run_ids: runIds, agent: $("#exportAgent").value.trim(), mode: $("#exportMode").value, append: $("#exportAppend").checked };
  const button = $("#exportSubmit");
  button.disabled = true;
  button.textContent = "正在生成…";
  try {
    const result = await api("/api/exports", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    await loadExports();
    openModal("训练数据导出完成", result.file.filename, `
      <div class="report-summary"><div><small>本次写入</small><strong>${result.written} 条</strong></div><div><small>跳过</small><strong>${result.skipped} 条</strong></div><div><small>文件总条数</small><strong>${result.file.rows} 条</strong></div><div><small>文件大小</small><strong>${formatBytes(result.file.size_bytes)}</strong></div></div>
      <a class="button primary download-button" href="${escapeHtml(result.file.download_url)}">下载 ${escapeHtml(result.file.filename)}</a>
      <h3 class="report-section-title">前 3 条预览</h3><pre class="json-view">${escapeHtml(result.preview.join("\n\n"))}</pre>`);
  } catch (error) { toast(`导出失败：${error.message}`, true); }
  finally { button.disabled = false; button.textContent = "开始导出"; }
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
  $("#replayButton").addEventListener("click", startReplay);

  $("#datasetQuery").addEventListener("click", renderDatasets);
  $("#datasetReset").addEventListener("click", () => { $("#datasetSearch").value = ""; $("#datasetStatus").value = ""; renderDatasets(); });
  $("#modelQuery").addEventListener("click", renderModels);
  $("#modelReset").addEventListener("click", () => { $("#modelSearch").value = ""; renderModels(); });
  $("#skillQuery").addEventListener("click", renderSkills);
  $("#skillReset").addEventListener("click", () => { $("#skillSearch").value = ""; renderSkills(); });
  $("#skillTemplate").addEventListener("click", showSkillTemplate);
  $("#skillImport").addEventListener("click", () => $("#skillFile").click());
  $("#skillFile").addEventListener("change", event => previewSkillFile(event.target.files[0]));
  $("#skillEvalTarget").addEventListener("change", renderSkillEvalOptions);
  $$('input[name="skill_eval_mode"]').forEach(node => node.addEventListener("change", updateSkillEvalMode));
  $("#skillEvalForm").addEventListener("submit", submitSkillEvaluation);
  $("#reportQuery").addEventListener("click", renderReports);
  $("#reportReset").addEventListener("click", () => { $("#reportSearch").value = ""; renderReports(); });
  $("#traceQuery").addEventListener("click", loadTraces);
  $("#traceReset").addEventListener("click", () => { $("#traceAgent").value = ""; $("#traceTask").value = ""; $("#traceRun").value = ""; $("#traceResult").value = "all"; loadTraces(); });
  $("#jobsRefresh").addEventListener("click", loadJobs);
  $("#compareForm").addEventListener("submit", submitComparison);
  $("#traceDiffForm").addEventListener("submit", submitTraceDiff);
  $("#exportForm").addEventListener("submit", submitExport);
  $("#exportsRefresh").addEventListener("click", loadExports);
  $("#dpoRunsRefresh").addEventListener("click", loadDpoRunIds);
  $("#exportAgent").addEventListener("change", loadDpoRunIds);
  $$('input[name="export_kind"]').forEach(node => node.addEventListener("change", updateExportKind));
  $("#runDataset").addEventListener("change", event => { $("#splitField").style.visibility = event.target.value === "tau_retail" ? "visible" : "hidden"; });
  $("#runForm").addEventListener("submit", submitRun);

  document.addEventListener("click", event => {
    const report = event.target.closest(".report-open");
    const reportDelete = event.target.closest(".report-delete");
    const dataset = event.target.closest(".dataset-open");
    const log = event.target.closest(".log-open");
    const trace = event.target.closest(".trace-open");
    const timeline = event.target.closest(".timeline-event");
    const skill = event.target.closest(".skill-open");
    if (report) showReport(report.dataset.run);
    if (reportDelete) confirmDeleteReport(reportDelete.dataset.run);
    if (dataset) showDataset(dataset.dataset.id);
    if (log) showLog(log.dataset.job);
    if (trace) openTrace(trace.dataset.agent, trace.dataset.task, trace.dataset.run);
    if (timeline) showEvent(timeline.dataset.agent, timeline.dataset.task, timeline.dataset.run, timeline.dataset.seq);
    if (skill) showSkill(skill.dataset.skill);
  });
}

async function refreshAll() {
  const button = $("#refreshButton");
  button.disabled = true;
  try {
    await Promise.all([checkProxy(), loadOverview(), loadDatasets(), loadModels(), loadSkills(), loadReports()]);
    toast("数据已刷新");
  } catch (error) {
    toast(`刷新失败：${error.message}`, true);
  } finally {
    button.disabled = false;
  }
}

async function init() {
  bindEvents();
  updateExportKind();
  updateSkillEvalMode();
  try {
    await Promise.all([checkProxy(), loadOverview(), loadDatasets(), loadModels(), loadSkills(), loadReports()]);
  } catch (error) {
    toast(`页面初始化失败：${error.message}`, true);
  }
  setInterval(() => {
    if ($("#page-jobs").classList.contains("active")) loadJobs();
  }, 5000);
}

init();
