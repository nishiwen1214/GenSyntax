// ===== State =====
let tasks = {};
let selectedFile = null;
let selectedSingleFile = null;
let batchResults = null;
let pollTimer = null;

// Tasks that use file upload in single-prediction mode (client-side authority,
// does not rely on server returning input_mode to avoid stale-cache issues).
const FILE_MODE_TASKS = new Set(['minimal_genome']);

// ===== Task Templates =====
// Each task that needs a fill-in-the-blank form is defined here.
// parts: { f } = fixed text, { v, tag, optional } = editable field
// sendMode 'single_var' : send only the named var (backend adds prefix)
// sendMode 'assemble'   : concatenate all parts verbatim
// sendMode 'gene_function': custom assembly with dynamic options + [unknown product] check
const TASK_TEMPLATES = {
    plasmid_host: {
        sendMode: 'single_var',
        sendVar: 'products',
        parts: [
            { f: 'This is the list of protein products encoded by a plasmid. Which bacterial host is this plasmid most likely to come from? Answer strictly in the following format: [genus, species, strain]\n' },
            { v: 'products', tag: 'textarea', placeholder: '[CDS product 1][CDS product 2][CDS product 3]...', rows: 5 }
        ]
    },
    gene_function: {
        sendMode: 'gene_function',
        parts: [
            { f: 'The following list presents the products encoded by the ' },
            { v: 'organism', tag: 'input', placeholder: 'name', size: 18 },
            { f: ' ' },
            { v: 'plasmid_chr', tag: 'input', placeholder: 'plasmid / chromosome', size: 18 },
            { f: '. One of the genes in the list is designated as [unknown product]. Please predict which product from the options below [unknown product] is most likely to be. A: ' },
            { v: 'opt_a', tag: 'input', size: 14, placeholder: 'option A' },
            { f: '; B: ' },
            { v: 'opt_b', tag: 'input', size: 14, placeholder: 'option B' },
            { f: '; C: ' },
            { v: 'opt_c', tag: 'input', size: 14, placeholder: 'option C', optional: true },
            { f: '; D: ' },
            { v: 'opt_d', tag: 'input', size: 14, placeholder: 'option D', optional: true },
            { f: '; E: ' },
            { v: 'opt_e', tag: 'input', size: 14, placeholder: 'option E', optional: true },
            { f: '; F: ' },
            { v: 'opt_f', tag: 'input', size: 14, placeholder: 'option F', optional: true },
            { f: '; G: ' },
            { v: 'opt_g', tag: 'input', size: 14, placeholder: 'option G', optional: true },
            { f: '; H: ' },
            { v: 'opt_h', tag: 'input', size: 14, placeholder: 'option H', optional: true },
            { f: '\nAnswer strictly in the following format: [A] or [B] or ... or [last option]\n' },
            { v: 'gene_list', tag: 'textarea', placeholder: '[gene1][gene2]...[unknown product]...[geneN]\n(gene list must contain [unknown product])', rows: 5 }
        ]
    },
    genome_assembly: {
        sendMode: 'assemble',
        parts: [
            { f: 'Below are ' },
            { v: 'n_contigs', tag: 'input', placeholder: 'N', size: 3 },
            { f: ' contigs of an ' },
            { v: 'organism', tag: 'input', placeholder: 'organism name', size: 20 },
            { f: ' chromosome genome, assemble them in the correct order. Answer strictly in the format: Contig X-Contig Y-Contig Z...\n' },
            { v: 'contigs', tag: 'textarea', placeholder: 'Contig 1: [gene1][gene2]...\nContig 2: [gene1][gene2]...', rows: 7 }
        ]
    },
    gene_essentiality: {
        sendMode: 'assemble',
        parts: [
            { f: 'The following list presents the protein products encoded by the ' },
            { v: 'organism', tag: 'input', placeholder: 'organism name', size: 20 },
            { f: ' chromosome. Please predict whether the gene corresponding to the protein product $$' },
            { v: 'target_gene', tag: 'input', placeholder: 'target gene product', size: 22 },
            { f: '$$ is essential for this organism? Answer strictly in the following format: non-essential or essential\n' },
            { v: 'gene_list', tag: 'textarea', placeholder: '[gene1][gene2]...$$target gene product$$...[geneN]', rows: 5 }
        ]
    }
};

function renderTemplate(taskId) {
    const tmpl = TASK_TEMPLATES[taskId];
    if (!tmpl) return;
    const container = document.getElementById('template-form');
    container.innerHTML = '';
    for (const part of tmpl.parts) {
        if (part.f !== undefined) {
            const lines = part.f.split('\n');
            lines.forEach((line, i) => {
                if (line) {
                    const span = document.createElement('span');
                    span.className = 'tpl-fixed';
                    span.textContent = line;
                    container.appendChild(span);
                }
                if (i < lines.length - 1) container.appendChild(document.createElement('br'));
            });
        } else if (part.tag === 'textarea') {
            const ta = document.createElement('textarea');
            ta.className = 'tpl-var-textarea';
            ta.dataset.varId = part.v;
            ta.placeholder = part.placeholder || '';
            ta.rows = part.rows || 4;
            ta.addEventListener('input', updatePredictBtn);
            container.appendChild(ta);
        } else {
            const inp = document.createElement('input');
            inp.type = 'text';
            inp.className = part.optional ? 'tpl-var-input tpl-var-opt' : 'tpl-var-input';
            inp.dataset.varId = part.v;
            if (part.optional) inp.dataset.optional = 'true';
            inp.placeholder = part.placeholder || '';
            inp.size = part.size || 12;
            inp.addEventListener('input', updatePredictBtn);
            container.appendChild(inp);
        }
    }
}

// Custom assembly for gene_function: skips empty options, builds dynamic answer format line.
function assembleGeneFunctionPrompt() {
    const get = v => {
        const el = document.querySelector(`#template-form [data-var-id="${v}"]`);
        return el ? el.value.trim() : '';
    };
    const organism   = get('organism');
    const plasmidChr = get('plasmid_chr');
    const geneList   = get('gene_list');
    const letters    = ['a','b','c','d','e','f','g','h'];
    const options    = letters.map(l => get(`opt_${l}`)).filter(Boolean);
    const optionStr  = options.map((v, i) => `${String.fromCharCode(65 + i)}: ${v}`).join('; ');
    const formatStr  = options.map((_, i) => `[${String.fromCharCode(65 + i)}]`).join(' or ');
    return (
        `The following list presents the products encoded by the ${organism} ${plasmidChr}. ` +
        `One of the genes in the list is designated as [unknown product]. ` +
        `Please predict which product from the options below [unknown product] is most likely to be. ${optionStr}\n` +
        `Answer strictly in the following format: ${formatStr}\n` +
        geneList
    );
}

function assemblePromptFromTemplate(taskId) {
    const tmpl = TASK_TEMPLATES[taskId];
    if (!tmpl) return null;
    if (tmpl.sendMode === 'gene_function') return assembleGeneFunctionPrompt();
    if (tmpl.sendMode === 'single_var') {
        const el = document.querySelector(`#template-form [data-var-id="${tmpl.sendVar}"]`);
        return el ? el.value.trim() : '';
    }
    let prompt = '';
    for (const part of tmpl.parts) {
        if (part.f !== undefined) {
            prompt += part.f;
        } else {
            const el = document.querySelector(`#template-form [data-var-id="${part.v}"]`);
            prompt += el ? el.value : '';
        }
    }
    return prompt.trim();
}

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    loadTasks();
    setupDragDrop();
    setupSingleDragDrop();
});

async function loadTasks() {
    try {
        const resp = await fetch('/api/tasks');
        tasks = await resp.json();
        populateSelects();
    } catch (e) {
        console.error('Failed to load tasks:', e);
    }
}

function populateSelects() {
    ['task-select', 'batch-task-select'].forEach(id => {
        const sel = document.getElementById(id);
        Object.entries(tasks).forEach(([key, val]) => {
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = val.name;
            sel.appendChild(opt);
        });
    });
    updatePredictBtn();
}

function onTaskChange() {
    const sel = document.getElementById('task-select');
    const taskId = sel.value;
    const taskCfg = taskId ? tasks[taskId] : null;
    // FILE_MODE_TASKS is authoritative client-side; server input_mode is a fallback
    const isFileMode = FILE_MODE_TASKS.has(taskId) || !!(taskCfg && taskCfg.input_mode === 'file');
    const hasTemplate = !isFileMode && !!TASK_TEMPLATES[taskId];

    document.getElementById('single-text-group').style.display = (!isFileMode && !hasTemplate) ? '' : 'none';
    document.getElementById('single-file-group').style.display = isFileMode ? '' : 'none';
    document.getElementById('template-form-group').style.display = hasTemplate ? '' : 'none';

    // Show/hide the minimal-genome format hint
    const hint = document.getElementById('minimal-genome-hint');
    if (hint) hint.style.display = (isFileMode && taskId === 'minimal_genome') ? '' : 'none';

    if (hasTemplate) {
        renderTemplate(taskId);
    } else if (!isFileMode) {
        const textarea = document.getElementById('input-text');
        textarea.placeholder = taskCfg ? taskCfg.placeholder : 'Select a task first...';
    }
    updatePredictBtn();
}

function updatePredictBtn() {
    const task = document.getElementById('task-select').value;
    const taskCfg = task ? tasks[task] : null;
    const isFileMode = FILE_MODE_TASKS.has(task) || !!(taskCfg && taskCfg.input_mode === 'file');
    const hasTemplate = !isFileMode && !!TASK_TEMPLATES[task];
    let ready = false;
    const missingReasons = [];

    if (!task) {
        ready = false;
    } else if (isFileMode) {
        ready = !!selectedSingleFile;
        if (!ready) missingReasons.push('Upload a JSON file');
    } else if (hasTemplate) {
        const FIELD_LABELS = {
            organism: 'Organism name', plasmid_chr: 'Plasmid / chromosome',
            opt_a: 'Option A', opt_b: 'Option B', gene_list: 'Gene list',
            n_contigs: 'Number of contigs', contigs: 'Contig sequences',
            target_gene: 'Target gene product', products: 'Protein products',
        };
        const reqFields = Array.from(document.querySelectorAll('#template-form [data-var-id]'))
            .filter(f => !f.dataset.optional);
        reqFields.forEach(f => {
            if (!f.value.trim()) missingReasons.push(FIELD_LABELS[f.dataset.varId] || f.dataset.varId);
        });
        const allFilled = reqFields.length > 0 && reqFields.every(f => f.value.trim());
        let extraValid = true;
        if (task === 'gene_function') {
            const gl = document.querySelector('#template-form [data-var-id="gene_list"]');
            if (gl && gl.value.trim() && !gl.value.includes('unknown product')) {
                extraValid = false;
                missingReasons.push('Gene list must contain [unknown product]');
            }
        }
        ready = allFilled && extraValid;
    } else {
        ready = !!document.getElementById('input-text').value.trim();
        if (!ready) missingReasons.push('Input data');
    }

    document.getElementById('predict-btn').disabled = !ready;

    const hint = document.getElementById('predict-hint');
    if (hint) {
        if (!ready && missingReasons.length) {
            hint.innerHTML = '<span class="missing-label">Still needed:</span>' +
                missingReasons.map(r => `<span class="missing-field">${escapeHtml(r)}</span>`).join('');
            hint.style.display = 'flex';
        } else {
            hint.style.display = 'none';
        }
    }
}

function updateBatchBtn() {
    const task = document.getElementById('batch-task-select').value;
    document.getElementById('batch-predict-btn').disabled = !(task && selectedFile);
}

// ===== Mode Switch =====
function switchMode(mode) {
    document.querySelectorAll('.mode-tab').forEach(t => t.classList.toggle('active', t.dataset.mode === mode));
    document.querySelectorAll('.mode-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(mode + '-mode').classList.add('active');
}

// ===== Single Prediction =====
async function predictSingle() {
    const task = document.getElementById('task-select').value;
    const taskCfg = task ? tasks[task] : null;
    const isFileMode = FILE_MODE_TASKS.has(task) || !!(taskCfg && taskCfg.input_mode === 'file');
    const hasTemplate = !isFileMode && !!TASK_TEMPLATES[task];

    const btn = document.getElementById('predict-btn');
    setBtnLoading(btn, true);
    hidePanel('result-panel');
    hidePanel('error-panel');

    try {
        if (isFileMode) {
            if (!selectedSingleFile) return;
            const formData = new FormData();
            formData.append('file', selectedSingleFile);
            const resp = await fetch(`/api/predict/upload?task=${encodeURIComponent(task)}`, {
                method: 'POST',
                body: formData,
            });
            const jobData = await resp.json();
            if (!resp.ok) throw new Error(jobData.detail || 'Failed to submit job');
            // Reuse batch polling — show results in the batch result panel
            setBtnLoading(btn, false);
            switchMode('batch');
            document.getElementById('batch-task-select').value = task;
            showProgress(0, jobData.total);
            pollJob(jobData.job_id, jobData.total);
            return;
        } else {
            let data;
            const input = hasTemplate
                ? assemblePromptFromTemplate(task)
                : document.getElementById('input-text').value.trim();
            if (!input) return;
            const resp = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task, input }),
            });
            data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || 'Prediction failed');
            document.getElementById('result-text').textContent = data.result;
            showPanel('result-panel');
        }
    } catch (e) {
        showError('error-panel', 'error-text', e.message);
    } finally {
        setBtnLoading(btn, false);
    }
}

// ===== Single File Upload (file-mode tasks) =====
function setupSingleDragDrop() {
    const zone = document.getElementById('single-drop-zone');
    if (!zone) return;
    ['dragenter', 'dragover'].forEach(ev => {
        zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('drag-over'); });
    });
    ['dragleave', 'drop'].forEach(ev => {
        zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('drag-over'); });
    });
    zone.addEventListener('drop', e => {
        const file = e.dataTransfer.files[0];
        if (file) handleSingleFile(file);
    });
}

function onSingleFileSelect(event) {
    const file = event.target.files[0];
    if (file) handleSingleFile(file);
}

function handleSingleFile(file) {
    if (!file.name.toLowerCase().endsWith('.json')) {
        showError('error-panel', 'error-text', 'Please upload a JSON file.');
        return;
    }
    selectedSingleFile = file;
    const info = document.getElementById('single-file-info');
    info.innerHTML = `<span>${file.name} (${(file.size / 1024).toFixed(1)} KB)</span>
        <button class="remove-file" onclick="clearSingleFile()">Remove</button>`;
    info.style.display = 'flex';
    updatePredictBtn();
}

function clearSingleFile() {
    selectedSingleFile = null;
    document.getElementById('single-file-input').value = '';
    document.getElementById('single-file-info').style.display = 'none';
    updatePredictBtn();
}

// ===== Drag & Drop =====
function setupDragDrop() {
    const zone = document.getElementById('drop-zone');
    ['dragenter', 'dragover'].forEach(ev => {
        zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('drag-over'); });
    });
    ['dragleave', 'drop'].forEach(ev => {
        zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('drag-over'); });
    });
    zone.addEventListener('drop', e => {
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });
}

function onFileSelect(event) {
    const file = event.target.files[0];
    if (file) handleFile(file);
}

function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.json')) {
        showError('batch-error-panel', 'batch-error-text', 'Please upload a JSON file.');
        return;
    }
    selectedFile = file;
    const info = document.getElementById('file-info');
    info.innerHTML = `<span>${file.name} (${(file.size / 1024).toFixed(1)} KB)</span>
        <button class="remove-file" onclick="clearFile()">Remove</button>`;
    info.style.display = 'flex';
    updateBatchBtn();
}

function clearFile() {
    selectedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('file-info').style.display = 'none';
    updateBatchBtn();
}

// ===== Batch Prediction (async with polling) =====
async function predictBatch() {
    const task = document.getElementById('batch-task-select').value;
    if (!task || !selectedFile) return;

    const btn = document.getElementById('batch-predict-btn');
    setBtnLoading(btn, true);
    hidePanel('batch-result-panel');
    hidePanel('batch-error-panel');
    showProgress(0, 0);

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const resp = await fetch(`/api/predict/batch?task=${encodeURIComponent(task)}`, {
            method: 'POST',
            body: formData,
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Failed to submit batch job');

        // Start polling
        pollJob(data.job_id, data.total);
    } catch (e) {
        hideProgress();
        setBtnLoading(btn, false);
        showError('batch-error-panel', 'batch-error-text', e.message);
    }
}

function pollJob(jobId, total) {
    const btn = document.getElementById('batch-predict-btn');
    if (pollTimer) clearInterval(pollTimer);

    pollTimer = setInterval(async () => {
        try {
            const resp = await fetch(`/api/predict/batch/${jobId}`);
            const job = await resp.json();
            if (!resp.ok) throw new Error(job.detail || 'Polling failed');

            showProgress(job.done, job.total);

            if (job.status === 'done') {
                clearInterval(pollTimer);
                pollTimer = null;
                hideProgress();
                setBtnLoading(btn, false);

                batchResults = job.results;
                renderBatchResults(job);
                showPanel('batch-result-panel');
            }
        } catch (e) {
            clearInterval(pollTimer);
            pollTimer = null;
            hideProgress();
            setBtnLoading(btn, false);
            showError('batch-error-panel', 'batch-error-text', e.message);
        }
    }, 1000);
}

function showProgress(done, total) {
    const bar = document.getElementById('progress-bar');
    const fill = document.getElementById('progress-fill');
    const text = document.getElementById('progress-text');
    bar.style.display = 'block';
    const pct = total > 0 ? Math.round(done / total * 100) : 0;
    fill.style.width = pct + '%';
    text.textContent = `${done} / ${total} (${pct}%)`;
}

function hideProgress() {
    document.getElementById('progress-bar').style.display = 'none';
}

function renderBatchResults(data) {
    const successCount = data.results.filter(r => !r.startsWith('[ERROR]')).length;
    const errorCount = data.total - successCount;

    document.getElementById('batch-summary').innerHTML = `
        <div class="summary-chip total">Total: ${data.total}</div>
        <div class="summary-chip success">Success: ${successCount}</div>
        ${errorCount > 0 ? `<div class="summary-chip error">Errors: ${errorCount}</div>` : ''}
    `;

    const list = document.getElementById('batch-results-list');
    list.innerHTML = data.results.map((r, i) => `
        <div class="result-item">
            <div class="index ${r.startsWith('[ERROR]') ? 'error' : ''}">${i + 1}</div>
            <div class="content">
                <div class="${r.startsWith('[ERROR]') ? 'error-msg' : 'output'}">${escapeHtml(r)}</div>
            </div>
        </div>
    `).join('');
}

function downloadResults() {
    if (!batchResults) return;
    const text = batchResults.join('\n');
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'gensyntax_predictions.txt';
    a.click();
    URL.revokeObjectURL(url);
}

function copyResult() {
    const text = document.getElementById('result-text').textContent;
    const btn = document.querySelector('.btn-copy');
    const orig = btn.innerHTML;
    const copied = () => {
        btn.innerHTML = '<svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"/></svg> Copied!';
        setTimeout(() => { btn.innerHTML = orig; }, 1500);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(copied);
    } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        copied();
    }
}

// ===== Helpers =====
function setBtnLoading(btn, loading) {
    btn.disabled = loading;
    btn.querySelector('.btn-text').style.display = loading ? 'none' : '';
    btn.querySelector('.btn-loading').style.display = loading ? 'inline-flex' : 'none';
}

function showPanel(id) { document.getElementById(id).style.display = ''; }
function hidePanel(id) { document.getElementById(id).style.display = 'none'; }
function showError(panelId, textId, msg) {
    document.getElementById(textId).textContent = msg;
    showPanel(panelId);
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

document.getElementById('input-text')?.addEventListener('input', updatePredictBtn);
document.getElementById('batch-task-select')?.addEventListener('change', updateBatchBtn);
