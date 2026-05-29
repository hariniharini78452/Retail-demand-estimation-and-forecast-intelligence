// ======================================================
// GLOBAL CHART INSTANCES
// ======================================================
let categoryChartInstance = null;
let monthlyChartInstance  = null;
let forecastChartInstance = null;

// ── Shared Chart Defaults (dark theme) ────────────────
Chart.defaults.color          = '#64748b';
Chart.defaults.borderColor    = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family    = 'Inter, sans-serif';
Chart.defaults.font.size      = 12;

// ======================================================
// DOM READY
// ======================================================
document.addEventListener("DOMContentLoaded", function () {

    // Dashboard
    if (window.location.pathname === "/dashboard") {
        loadKPIs();
        loadCategoryChart();
        loadMonthlyChart();
        loadCategoryFilter();
        loadInsights();

        const startInput    = document.getElementById("startDate");
        const endInput      = document.getElementById("endDate");
        const categoryInput = document.getElementById("categoryFilter");

        if (startInput && endInput && categoryInput) {
            startInput.addEventListener("change", applyFilters);
            endInput.addEventListener("change", applyFilters);
            categoryInput.addEventListener("change", applyFilters);
        }
    }


});

// ======================================================
// KPI LOADER
// ======================================================
function loadKPIs(url = "/api/kpis") {
    fetch(url)
        .then(r => r.json())
        .then(data => {
            if (data.error) return;
            document.getElementById("kpi-transactions").innerText = data.transactions.toLocaleString();
            document.getElementById("kpi-revenue").innerText      = "₹ " + data.revenue.toLocaleString();
            document.getElementById("kpi-quantity").innerText     = data.quantity.toLocaleString();
            document.getElementById("kpi-categories").innerText   = data.categories;
        });
}

// ======================================================
// CATEGORY CHART
// ======================================================
function loadCategoryChart(url = "/api/chart/category") {
    fetch(url)
        .then(r => r.json())
        .then(data => {
            const canvas = document.getElementById("categoryChart");
            if (!canvas) return;
            const ctx = canvas.getContext("2d");

            if (categoryChartInstance) categoryChartInstance.destroy();

            // Generate shades of blue → violet for bars
            const colors = data.labels.map((_, i) => {
                const hue = 210 + (i * 18) % 120;
                return `hsla(${hue}, 70%, 55%, 0.85)`;
            });

            categoryChartInstance = new Chart(ctx, {
                type: "bar",
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: "Revenue",
                        data: data.values,
                        backgroundColor: colors,
                        borderRadius: 6,
                        borderSkipped: false,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#111827',
                            borderColor: 'rgba(255,255,255,0.1)',
                            borderWidth: 1,
                            titleColor: '#e2e8f0',
                            bodyColor: '#94a3b8',
                            callbacks: {
                                label: ctx => " ₹ " + ctx.parsed.y.toLocaleString()
                            }
                        }
                    },
                    onClick: function (evt, elements) {
                        if (elements.length > 0) {
                            const selectedCategory = this.data.labels[elements[0].index];
                            const dropdown = document.getElementById("categoryFilter");
                            if (dropdown) { dropdown.value = selectedCategory; applyFilters(); }
                        }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { maxRotation: 30, minRotation: 0, color: '#64748b' }
                        },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: {
                                color: '#64748b',
                                callback: v => "₹ " + (v / 1000000).toFixed(1) + "M"
                            }
                        }
                    }
                }
            });
        });
}

// ======================================================
// MONTHLY CHART
// ======================================================
function loadMonthlyChart(url = "/api/chart/monthly") {
    fetch(url)
        .then(r => r.json())
        .then(data => {
            const canvas = document.getElementById("monthlyChart");
            if (!canvas) return;
            const ctx = canvas.getContext("2d");

            if (monthlyChartInstance) monthlyChartInstance.destroy();

            // Gradient fill
            const gradient = ctx.createLinearGradient(0, 0, 0, 300);
            gradient.addColorStop(0, 'rgba(59,130,246,0.25)');
            gradient.addColorStop(1, 'rgba(59,130,246,0.01)');

            monthlyChartInstance = new Chart(ctx, {
                type: "line",
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: "Monthly Revenue",
                        data: data.values,
                        borderColor: "#3b82f6",
                        backgroundColor: gradient,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        pointBackgroundColor: "#3b82f6",
                        borderWidth: 2,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#111827',
                            borderColor: 'rgba(255,255,255,0.1)',
                            borderWidth: 1,
                            titleColor: '#e2e8f0',
                            bodyColor: '#94a3b8',
                            callbacks: {
                                label: ctx => " ₹ " + ctx.parsed.y.toLocaleString()
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { color: '#64748b', maxRotation: 30 }
                        },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: {
                                color: '#64748b',
                                callback: v => "₹ " + (v / 1000000).toFixed(1) + "M"
                            }
                        }
                    }
                }
            });
        });
}

// ======================================================
// APPLY FILTERS
// ======================================================
function applyFilters() {
    const start    = document.getElementById("startDate").value;
    const end      = document.getElementById("endDate").value;
    const category = document.getElementById("categoryFilter").value;
    const query    = `?start=${start}&end=${end}&category=${category}`;

    loadKPIs("/api/dashboard" + query);
    loadCategoryChart("/api/chart/category" + query);
    loadMonthlyChart("/api/chart/monthly" + query);
    loadInsights("/api/insights" + query);
}

// ======================================================
// CATEGORY FILTER DROPDOWN
// ======================================================
function loadCategoryFilter() {
    fetch("/api/categories")
        .then(r => r.json())
        .then(data => {
            const dropdown = document.getElementById("categoryFilter");
            if (!dropdown) return;
            dropdown.innerHTML = '<option value="all">All Categories</option>';
            data.categories.forEach(cat => {
                const opt = document.createElement("option");
                opt.value = cat; opt.textContent = cat;
                dropdown.appendChild(opt);
            });
        });
}

// ======================================================
// INSIGHTS
// ======================================================
function loadInsights(url = "/api/insights") {
    fetch(url)
        .then(r => r.json())
        .then(data => {
            const topList    = document.getElementById("top5List");
            const bottomList = document.getElementById("bottom5List");
            topList.innerHTML = "";
            bottomList.innerHTML = "";

            data.top5.forEach(item => {
                const li = document.createElement("li");
                li.textContent = item;
                topList.appendChild(li);
            });
            data.bottom5.forEach(item => {
                const li = document.createElement("li");
                li.textContent = item;
                bottomList.appendChild(li);
            });

            document.getElementById("peakMonth").innerText = data.peak_month;
        });
}

