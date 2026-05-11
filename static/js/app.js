/* app.js — Alpine.js application for Aakhi */

function aakhi() {
  return {
    // ── State ───────────────────────────────────────────────────────────── //
    lang: localStorage.getItem("aakhi_lang") || "en",
    i18n: {},
    languages: [
      { code: "en",  label: "EN" },
      { code: "hi",  label: "हि" },
      { code: "mr",  label: "म" },
      { code: "or",  label: "ଓ" },
      { code: "bn",  label: "বা" },
      { code: "te",  label: "తె" },
      { code: "ta",  label: "த" },
      { code: "gu",  label: "ગ" },
      { code: "sat", label: "ᱟ" },
    ],

    patient: { name: "", age: 30, gender: "Male", eye: "Right Eye (OD)" },
    patientConfirmed: false,

    selectedFile: null,
    previewUrl: null,

    jobId: null,
    analyzing: false,
    phase1Ready: false,
    phase2Ready: false,
    statusMsg: "Starting...",
    maProgressPct: 0,
    rfnldReady: false,

    results: null,
    reportLoading: false,
    activeTab: "drg",

    steps: [
      { key: "drg",      label: "DR Grading",            done: false, running: false },
      { key: "odoc",     label: "OD/OC Segmentation",    done: false, running: false },
      { key: "lesion",   label: "Lesion Detection",       done: false, running: false },
      { key: "glaucoma", label: "Glaucoma Grading",       done: false, running: false },
      { key: "ma",       label: "Microaneurysm Detection",done: false, running: false },
      { key: "rfnld",    label: "RFNLD Detection",        done: false, running: false },
    ],

    tabs: [
      { key: "drg",      label: "DR Grading" },
      { key: "odoc",     label: "OD / OC" },
      { key: "lesion",   label: "Lesions" },
      { key: "glaucoma", label: "Glaucoma" },
      { key: "ma",       label: "MA Detection" },
      { key: "rfnld",    label: "RFNLD" },
    ],

    // ── Init ────────────────────────────────────────────────────────────── //
    async init() {
      await this.loadI18n(this.lang);
      this.updateTabLabels();
    },

    async loadI18n(code) {
      try {
        const r = await fetch(`/api/i18n/${code}`);
        this.i18n = await r.json();
        this.lang = code;
        localStorage.setItem("aakhi_lang", code);
        this.updateTabLabels();
      } catch (e) {
        console.warn("i18n load failed", e);
      }
    },

    t(key, fallback = "") {
      return this.i18n[key] || fallback || key;
    },

    async switchLang(code) {
      await this.loadI18n(code);
    },

    updateTabLabels() {
      const map = {
        drg:      this.t("dr_grading",  "DR Grading"),
        odoc:     this.t("odoc",         "OD / OC"),
        lesion:   this.t("lesion",       "Lesions"),
        glaucoma: this.t("glaucoma",     "Glaucoma"),
        ma:       this.t("ma",           "MA Detection"),
        rfnld:    this.t("rfnld",        "RFNLD"),
      };
      this.tabs = this.tabs.map(t => ({ ...t, label: map[t.key] || t.label }));
      this.steps = this.steps.map(s => ({
        ...s,
        label: (map[s.key] || s.label),
      }));
    },

    // ── Patient ─────────────────────────────────────────────────────────── //
    confirmPatient() {
      if (!this.patient.name.trim()) return;
      this.patientConfirmed = true;
    },

    resetPatient() {
      this.patientConfirmed = false;
      this.selectedFile = null;
      this.previewUrl   = null;
      this.resetJob();
    },

    resetJob() {
      this.jobId         = null;
      this.analyzing     = false;
      this.phase1Ready   = false;
      this.phase2Ready   = false;
      this.rfnldReady    = false;
      this.maProgressPct = 0;
      this.results       = null;
      this.steps.forEach(s => { s.done = false; s.running = false; });
    },

    // ── File handling ────────────────────────────────────────────────────── //
    handleFile(event) {
      const file = event.target.files[0];
      if (!file) return;
      this.selectedFile = file;
      this.previewUrl   = URL.createObjectURL(file);
      this.resetJob();
    },

    handleDrop(event) {
      const file = event.dataTransfer.files[0];
      if (!file) return;
      this.selectedFile = file;
      this.previewUrl   = URL.createObjectURL(file);
      this.resetJob();
    },

    // ── Analysis ─────────────────────────────────────────────────────────── //
    async startAnalysis() {
      if (!this.selectedFile || this.analyzing) return;
      this.resetJob();
      this.analyzing = true;
      this.statusMsg = this.t("analyzing", "Analyzing...");

      const fd = new FormData();
      fd.append("image",          this.selectedFile);
      fd.append("patient_name",   this.patient.name);
      fd.append("patient_age",    this.patient.age);
      fd.append("patient_gender", this.patient.gender);
      fd.append("patient_eye",    this.patient.eye);

      try {
        const r    = await fetch("/api/analyze", { method: "POST", body: fd });
        const data = await r.json();
        if (!r.ok || data.error) throw new Error(data.error || "Analysis failed");
        this.jobId = data.job_id;
        this.listenSSE();
      } catch (e) {
        this.analyzing = false;
        this.statusMsg = "Error: " + e.message;
      }
    },

    listenSSE() {
      const es = new EventSource(`/api/stream/${this.jobId}`);
      es.onmessage = (e) => {
        const evt = JSON.parse(e.data);
        this.handleSSEEvent(evt);
        if (["rfnld_ready", "rfnld_error", "complete", "error"].includes(evt.type)) {
          es.close();
          this.fetchResults();
        }
        if (evt.type === "phase1_ready" || evt.type === "phase2_ready") {
          this.fetchResults();
        }
      };
      es.onerror = () => {
        es.close();
        // Fall back to polling
        const poll = setInterval(async () => {
          await this.fetchResults();
          if (this.rfnldReady) clearInterval(poll);
        }, 2000);
      };
    },

    handleSSEEvent(evt) {
      switch (evt.type) {
        case "status":
          this.statusMsg = evt.msg;
          break;
        case "progress":
          this.markStepDone(evt.step);
          break;
        case "phase1_ready":
          this.phase1Ready = true;
          this.analyzing   = false;
          this.markStepRunning("ma");
          break;
        case "ma_started":
          this.markStepRunning("ma");
          break;
        case "ma_progress":
          this.maProgressPct = evt.pct || 0;
          break;
        case "phase2_ready":
          this.phase2Ready = true;
          this.markStepDone("ma");
          this.maProgressPct = 100;
          this.markStepRunning("rfnld");
          break;
        case "rfnld_started":
          this.markStepRunning("rfnld");
          break;
        case "rfnld_ready":
          this.rfnldReady = true;
          this.markStepDone("rfnld");
          break;
        case "rfnld_error":
          this.rfnldReady = true;
          this.markStepDone("rfnld");
          break;
        case "ma_error":
          this.markStepDone("ma");
          break;
      }
    },

    markStepDone(key) {
      const s = this.steps.find(s => s.key === key);
      if (s) { s.done = true; s.running = false; }
    },

    markStepRunning(key) {
      const s = this.steps.find(s => s.key === key);
      if (s) { s.done = false; s.running = true; }
    },

    async fetchResults() {
      if (!this.jobId) return;
      try {
        const r    = await fetch(`/api/results/${this.jobId}`);
        const data = await r.json();
        this.results      = data;
        this.phase1Ready  = data.phase1_ready || this.phase1Ready;
        this.phase2Ready  = data.phase2_ready || this.phase2Ready;
        this.maProgressPct= data.results?.ma?.progress_pct || this.maProgressPct;
        // Sync step states
        if (data.results?.drg?.grade)     this.markStepDone("drg");
        if (data.results?.odoc?.overlay_b64) this.markStepDone("odoc");
        if (data.results?.lesion?.image_b64) this.markStepDone("lesion");
        if (data.results?.glaucoma?.grade) this.markStepDone("glaucoma");
        if (data.phase2_ready)             this.markStepDone("ma");
        else if (data.phase1_ready)        this.markStepRunning("ma");
        if (data.rfnld_ready)              this.markStepDone("rfnld");
        else if (data.phase2_ready)        this.markStepRunning("rfnld");
        this.rfnldReady = data.rfnld_ready || this.rfnldReady;
      } catch (e) {
        console.warn("fetchResults error", e);
      }
    },

    // ── Report download ──────────────────────────────────────────────────── //
    async downloadReport(phase) {
      if (!this.jobId) return;
      this.reportLoading = true;
      try {
        const r = await fetch(`/api/report/${this.jobId}/${phase}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ lang: this.lang }),
        });
        if (!r.ok) {
          const err = await r.json();
          alert("Report error: " + (err.error || r.statusText));
          return;
        }
        const blob = await r.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement("a");
        const name = this.patient.name.replace(/\s+/g, "_") || "patient";
        a.href     = url;
        a.download = `Aakhi_Report_${name}_Phase${phase}.pdf`;
        a.click();
        URL.revokeObjectURL(url);
      } finally {
        this.reportLoading = false;
      }
    },

    // ── Display helpers ──────────────────────────────────────────────────── //
    drGradeColor(level) {
      const colors = [
        "text-emerald-600", "text-yellow-600",
        "text-orange-600",  "text-red-600", "text-red-800",
      ];
      return colors[level ?? 0] || "text-slate-600";
    },

    cdrColor(vcdr) {
      if (!vcdr) return "text-slate-600";
      if (vcdr < 0.5)  return "text-emerald-600";
      if (vcdr <= 0.7) return "text-amber-600";
      return "text-red-600";
    },

    drDescription(grade) {
      const desc = {
        "No DR":            "No signs of diabetic retinopathy. The retina appears normal.",
        "Mild DR":          "At least one microaneurysm present. Regular monitoring recommended.",
        "Moderate DR":      "Multiple microaneurysms, haemorrhages, and possible exudates. Ophthalmology referral advised.",
        "Severe DR":        "Pre-proliferative stage. Urgent referral required.",
        "Proliferative DR": "Neovascularisation present. Immediate treatment required.",
      };
      return desc[grade] || "";
    },

    glaucomaDesc(grade) {
      const desc = {
        "No Glaucoma":       "No evidence of glaucomatous optic neuropathy.",
        "Glaucoma Suspect":  "Suspicious features. Full glaucoma workup recommended (IOP, visual fields, OCT-RNFL).",
        "Moderate Glaucoma": "Moderate optic neuropathy with rim thinning. IOP management required.",
        "Advanced Glaucoma": "Significant rim loss. Urgent specialist management required.",
      };
      return desc[grade] || "";
    },

    quadrantData() {
      const m = this.results?.results?.odoc?.measurements || {};
      const inf = m.rim_inferior_pct || 0;
      return [
        { label: "S", pct: m.rim_superior_pct || 0, ok: true },
        { label: "I", pct: inf,                      ok: inf >= (m.rim_superior_pct || 0) },
        { label: "N", pct: m.rim_nasal_pct    || 0, ok: true },
        { label: "T", pct: m.rim_temporal_pct || 0, ok: true },
      ];
    },

    lesionAreas() {
      const a = this.results?.results?.lesion?.areas || {};
      return [
        [this.t("hard_exudates",  "Hard Exudates"),  a.hard_exudates  || 0, "#ef4444"],
        [this.t("hemorrhages",    "Haemorrhages"),    a.hemorrhages    || 0, "#22c55e"],
        [this.t("microaneurysms", "Microaneurysms"),  a.microaneurysms || 0, "#3b82f6"],
        [this.t("soft_exudates",  "Soft Exudates"),   a.soft_exudates  || 0, "#eab308"],
      ];
    },

    glGradeColor(level) {
      const colors = [
        "text-emerald-600", "text-yellow-600",
        "text-orange-600",  "text-red-600", "text-red-800"
      ];
      return colors[level ?? 0] || "text-slate-600";
    },

    overallSeverityColor() {
      const drLevel = this.results?.results?.drg?.level || 0;
      const glLevel = this.results?.results?.glaucoma?.level || 0;
      let maxLevel = Math.max(drLevel, glLevel);
      
      if (this.phase2Ready) {
        if ((this.results?.results?.ma?.count || 0) > 0) maxLevel = Math.max(maxLevel, 1);
        if (this.results?.results?.rfnld?.defects_found) maxLevel = Math.max(maxLevel, 1);
      }
      
      if (maxLevel >= 3) return "bg-red-50 text-red-800 border border-red-200";
      if (maxLevel >= 2) return "bg-orange-50 text-orange-800 border border-orange-200";
      if (maxLevel >= 1) return "bg-yellow-50 text-yellow-800 border border-yellow-200";
      return "bg-emerald-50 text-emerald-800 border border-emerald-200";
    },

    overallNextSteps() {
      const drLevel = this.results?.results?.drg?.level || 0;
      const glLevel = this.results?.results?.glaucoma?.level || 0;
      
      let steps = [];
      
      if (drLevel >= 3) steps.push("🚨 <b>Diabetic Retinopathy:</b> Urgent Ophthalmology referral required for advanced DR.");
      else if (drLevel >= 1) steps.push("⚠️ <b>Diabetic Retinopathy:</b> Regular monitoring and specialist consultation recommended.");
      else steps.push("✅ <b>Diabetic Retinopathy:</b> No signs of DR.");

      if (glLevel >= 3) steps.push("🚨 <b>Glaucoma:</b> Urgent specialist management required for Advanced Glaucoma.");
      else if (glLevel >= 1) steps.push("⚠️ <b>Glaucoma:</b> Full glaucoma workup recommended (IOP, visual fields, OCT).");
      else steps.push("✅ <b>Glaucoma:</b> No evidence of glaucomatous neuropathy.");
      
      if (this.phase2Ready) {
        const maCount = this.results?.results?.ma?.count || 0;
        const rfnld = this.results?.results?.rfnld?.defects_found || false;
        
        if (maCount > 0 && drLevel === 0) steps.push("🔍 <b>Microaneurysms:</b> Found in Phase 2. Re-evaluate for early DR.");
        else if (maCount > 0) steps.push("🔍 <b>Microaneurysms:</b> Detected, consistent with DR findings.");
        else steps.push("✅ <b>Microaneurysms:</b> None detected.");

        if (rfnld && glLevel === 0) steps.push("🔍 <b>RFNLD:</b> Defects detected. Suggestive of early glaucomatous damage, clinical correlation needed.");
        else if (rfnld) steps.push("🔍 <b>RFNLD:</b> Defects detected, consistent with glaucoma findings.");
        else steps.push("✅ <b>RFNLD:</b> No significant defects.");
      } else {
        steps.push("⏳ <i>Awaiting Phase 2 analysis (MA & RFNLD) for further insights...</i>");
      }
      
      return steps.join("<br>");
    }
  };
}
