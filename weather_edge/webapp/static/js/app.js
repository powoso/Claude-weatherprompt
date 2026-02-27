/**
 * Weather Edge - Dashboard JavaScript
 * Uses Alpine.js for state management and Chart.js for visualizations.
 */

// Chart.js global defaults for dark theme
Chart.defaults.color = '#9ca3af';
Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;

// Shared chart color palette
const COLORS = {
  accent: '#6366f1',
  accentLight: '#818cf8',
  positive: '#10b981',
  negative: '#ef4444',
  warn: '#f59e0b',
  gray: '#6b7280',
  surface700: '#1a1f2e',
  elNino: '#ef4444',
  laNina: '#3b82f6',
  neutral: '#6b7280',
};

// Track chart instances for cleanup
const charts = {};

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    delete charts[id];
  }
}

// Alpine.js main component
function dashboard() {
  return {
    // Navigation
    tabs: [
      { id: 'opportunities', label: 'Opportunities' },
      { id: 'base_rates', label: 'Base Rates' },
      { id: 'enso', label: 'ENSO & Outlook' },
      { id: 'calibration', label: 'Calibration' },
      { id: 'bankroll', label: 'Bankroll' },
    ],
    activeTab: 'opportunities',
    loading: false,

    // Data stores
    oppData: { opportunities: [], count: 0, is_sample: true, last_scan: null },
    brData: null,
    ensoData: null,
    calData: { is_sample: true, calibration: {}, brier_score: null, roi: {}, total_predictions: 0, resolved_predictions: 0 },
    bankData: {},

    // Base rates controls
    brCity: 'New York City',
    brMetric: 'temperature_2m_max',
    brMonth: '7',

    // Computed
    get bestEdge() {
      const opps = this.oppData.opportunities || [];
      if (opps.length === 0) return null;
      return Math.max(...opps.map(o => o.edge || 0));
    },
    get totalKelly() {
      const opps = this.oppData.opportunities || [];
      return opps.reduce((sum, o) => sum + (o.kelly_bet || 0), 0);
    },

    // Lifecycle
    init() {
      this.fetchOpportunities();
      this.fetchEnso();
      this.fetchCalibration();
      this.fetchBankroll();

      // Watch for tab changes to render charts
      this.$watch('activeTab', (tab) => {
        if (tab === 'base_rates' && !this.brData) {
          this.fetchBaseRates();
        }
        // Re-render charts after Alpine updates DOM
        this.$nextTick(() => {
          if (tab === 'opportunities') this.renderConvergenceChart();
          if (tab === 'base_rates' && this.brData && !this.brData.error) this.renderBaseRateCharts();
          if (tab === 'enso' && this.ensoData && !this.ensoData.error) this.renderEnsoChart();
          if (tab === 'calibration') this.renderCalibrationChart();
        });
      });
    },

    // Formatters
    formatPct(v) {
      if (v == null) return '--';
      return (v * 100).toFixed(1) + '%';
    },
    formatEdge(v) {
      if (v == null) return '--';
      return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%';
    },
    formatTime(ts) {
      if (!ts) return '';
      try {
        const d = new Date(ts);
        return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      } catch { return ts; }
    },
    edgeClass(edge) {
      if (edge >= 0.08) return 'bg-positive/10 text-positive';
      if (edge >= 0.05) return 'bg-positive/5 text-positive/80';
      if (edge > 0) return 'bg-gray-500/10 text-gray-400';
      return 'bg-gray-500/5 text-gray-500';
    },
    ensoPhaseStyle(phase) {
      if (phase.includes('el_nino')) return 'bg-red-500/10 border-red-500/20 text-red-300';
      if (phase.includes('la_nina')) return 'bg-blue-500/10 border-blue-500/20 text-blue-300';
      return 'bg-surface-800 border-white/5';
    },

    // Data fetching
    async fetchOpportunities() {
      this.loading = true;
      try {
        const resp = await fetch('/api/opportunities');
        this.oppData = await resp.json();
        this.$nextTick(() => this.renderConvergenceChart());
      } catch (e) { console.error('Failed to fetch opportunities:', e); }
      this.loading = false;
    },

    async fetchBaseRates() {
      try {
        const params = new URLSearchParams({ city: this.brCity, metric: this.brMetric, month: this.brMonth });
        const resp = await fetch('/api/base-rates?' + params);
        this.brData = await resp.json();
        this.$nextTick(() => this.renderBaseRateCharts());
      } catch (e) { console.error('Failed to fetch base rates:', e); }
    },

    async fetchEnso() {
      try {
        const resp = await fetch('/api/enso');
        this.ensoData = await resp.json();
        if (this.activeTab === 'enso') {
          this.$nextTick(() => this.renderEnsoChart());
        }
      } catch (e) { console.error('Failed to fetch ENSO:', e); }
    },

    async fetchCalibration() {
      try {
        const resp = await fetch('/api/calibration');
        this.calData = await resp.json();
        if (this.activeTab === 'calibration') {
          this.$nextTick(() => this.renderCalibrationChart());
        }
      } catch (e) { console.error('Failed to fetch calibration:', e); }
    },

    async fetchBankroll() {
      try {
        const resp = await fetch('/api/bankroll');
        this.bankData = await resp.json();
      } catch (e) { console.error('Failed to fetch bankroll:', e); }
    },

    // Chart rendering
    renderConvergenceChart() {
      const el = document.getElementById('convergenceChart');
      if (!el) return;
      destroyChart('convergence');

      const opps = this.oppData.opportunities || [];
      if (opps.length === 0) {
        // Show placeholder
        charts['convergence'] = new Chart(el, {
          type: 'line',
          data: {
            labels: ['Day 1', 'Day 5', 'Day 10', 'Day 15', 'Day 20', 'Day 25', 'Day 30'],
            datasets: [
              { label: 'Model Probability', data: [0.12, 0.13, 0.12, 0.14, 0.13, 0.12, 0.123], borderColor: COLORS.accent, backgroundColor: COLORS.accent + '20', tension: 0.4, fill: true, pointRadius: 3 },
              { label: 'Market Price', data: [0.06, 0.07, 0.07, 0.08, 0.08, 0.09, 0.08], borderColor: COLORS.negative, borderDash: [5, 5], tension: 0.4, pointRadius: 3 },
            ],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { ticks: { callback: v => (v * 100).toFixed(0) + '%' }, grid: { color: 'rgba(255,255,255,0.03)' } }, x: { grid: { display: false } } },
            plugins: { legend: { position: 'top' }, tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + (ctx.parsed.y * 100).toFixed(1) + '%' } } },
          },
        });
        return;
      }
      // If we have real convergence data for first contract, use it
      const cid = opps[0].contract_id;
      fetch('/api/convergence/' + encodeURIComponent(cid))
        .then(r => r.json())
        .then(data => {
          if (data.timestamps.length === 0) return;
          charts['convergence'] = new Chart(el, {
            type: 'line',
            data: {
              labels: data.timestamps.map(t => new Date(t).toLocaleDateString()),
              datasets: [
                { label: 'Model', data: data.model_prob, borderColor: COLORS.accent, backgroundColor: COLORS.accent + '20', tension: 0.3, fill: true },
                { label: 'Market', data: data.market_prob, borderColor: COLORS.negative, borderDash: [5, 5], tension: 0.3 },
              ],
            },
            options: {
              responsive: true, maintainAspectRatio: false,
              scales: { y: { ticks: { callback: v => (v * 100).toFixed(0) + '%' }, grid: { color: 'rgba(255,255,255,0.03)' } }, x: { grid: { display: false } } },
            },
          });
        });
    },

    renderBaseRateCharts() {
      if (!this.brData || this.brData.error) return;

      // Histogram
      const histEl = document.getElementById('histogramChart');
      if (histEl) {
        destroyChart('histogram');
        charts['histogram'] = new Chart(histEl, {
          type: 'bar',
          data: {
            labels: this.brData.histogram.bins.map(b => b.toFixed(1)),
            datasets: [{
              label: this.brData.month_name + ' Distribution',
              data: this.brData.histogram.counts,
              backgroundColor: COLORS.accent + '80',
              borderColor: COLORS.accent,
              borderWidth: 1,
              borderRadius: 2,
            }],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
              x: { grid: { display: false }, ticks: { maxTicksLimit: 12 } },
              y: { grid: { color: 'rgba(255,255,255,0.03)' }, title: { display: true, text: 'Days' } },
            },
            plugins: { legend: { display: false } },
          },
        });
      }

      // Yearly trend
      const yearEl = document.getElementById('yearlyChart');
      if (yearEl) {
        destroyChart('yearly');
        const years = this.brData.yearly_avg.years;
        const values = this.brData.yearly_avg.values;

        // Compute trend line
        const n = years.length;
        const xMean = years.reduce((a, b) => a + b, 0) / n;
        const yMean = values.reduce((a, b) => a + b, 0) / n;
        let num = 0, den = 0;
        for (let i = 0; i < n; i++) {
          num += (years[i] - xMean) * (values[i] - yMean);
          den += (years[i] - xMean) ** 2;
        }
        const slope = den !== 0 ? num / den : 0;
        const intercept = yMean - slope * xMean;
        const trendData = years.map(y => slope * y + intercept);

        charts['yearly'] = new Chart(yearEl, {
          type: 'line',
          data: {
            labels: years,
            datasets: [
              { label: 'Average', data: values, borderColor: COLORS.accent, backgroundColor: COLORS.accent + '15', tension: 0.3, fill: true, pointRadius: 2, pointHoverRadius: 5 },
              { label: 'Trend', data: trendData, borderColor: COLORS.warn, borderDash: [6, 4], pointRadius: 0, borderWidth: 2 },
            ],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
              x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } },
              y: { grid: { color: 'rgba(255,255,255,0.03)' } },
            },
          },
        });
      }
    },

    renderEnsoChart() {
      if (!this.ensoData || this.ensoData.error) return;
      const el = document.getElementById('ensoChart');
      if (!el) return;
      destroyChart('enso');

      const labels = this.ensoData.history.labels;
      const values = this.ensoData.history.values;

      // Color each point by phase
      const bgColors = values.map(v => {
        if (v >= 0.5) return COLORS.elNino + '60';
        if (v <= -0.5) return COLORS.laNina + '60';
        return COLORS.neutral + '40';
      });

      charts['enso'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'ONI Anomaly (\u00b0C)',
            data: values,
            backgroundColor: bgColors,
            borderColor: values.map(v => v >= 0.5 ? COLORS.elNino : v <= -0.5 ? COLORS.laNina : COLORS.neutral),
            borderWidth: 1,
            borderRadius: 1,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { grid: { display: false }, ticks: { maxTicksLimit: 15, maxRotation: 0 } },
            y: {
              grid: { color: 'rgba(255,255,255,0.03)' },
              title: { display: true, text: 'ONI (\u00b0C)' },
            },
          },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: ctx => 'ONI: ' + ctx.parsed.y.toFixed(2) + '\u00b0C' } },
            annotation: {
              annotations: {
                elNinoLine: { type: 'line', yMin: 0.5, yMax: 0.5, borderColor: COLORS.elNino + '60', borderDash: [4, 4], borderWidth: 1 },
                laNinaLine: { type: 'line', yMin: -0.5, yMax: -0.5, borderColor: COLORS.laNina + '60', borderDash: [4, 4], borderWidth: 1 },
              },
            },
          },
        },
      });
    },

    renderCalibrationChart() {
      const el = document.getElementById('calibrationChart');
      if (!el) return;
      destroyChart('calibration');

      const cal = this.calData.calibration || {};
      const predicted = cal.predicted || [];
      const actual = cal.actual || [];

      if (predicted.length === 0) return;

      // Perfect line
      const perfect = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

      charts['calibration'] = new Chart(el, {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: 'Perfect Calibration',
              data: perfect.map(v => ({ x: v, y: v })),
              borderColor: COLORS.gray,
              borderDash: [6, 4],
              showLine: true,
              pointRadius: 0,
              borderWidth: 1,
            },
            {
              label: this.calData.is_sample ? 'Sample Model' : 'Model',
              data: predicted.map((p, i) => ({ x: p, y: actual[i] })),
              borderColor: COLORS.accent,
              backgroundColor: COLORS.accent + '40',
              showLine: true,
              tension: 0.3,
              pointRadius: 6,
              pointHoverRadius: 9,
              pointBackgroundColor: COLORS.accent,
              borderWidth: 2,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: {
              min: 0, max: 1,
              title: { display: true, text: 'Predicted Probability' },
              ticks: { callback: v => (v * 100) + '%' },
              grid: { color: 'rgba(255,255,255,0.03)' },
            },
            y: {
              min: 0, max: 1,
              title: { display: true, text: 'Actual Outcome Rate' },
              ticks: { callback: v => (v * 100) + '%' },
              grid: { color: 'rgba(255,255,255,0.03)' },
            },
          },
          plugins: {
            tooltip: { callbacks: { label: ctx => 'Predicted: ' + (ctx.parsed.x * 100).toFixed(0) + '%, Actual: ' + (ctx.parsed.y * 100).toFixed(0) + '%' } },
          },
        },
      });
    },
  };
}
