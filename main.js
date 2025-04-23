document.addEventListener('DOMContentLoaded', function() {
    const carrierFreq = document.getElementById('carrierFreq');
    const pilotFreq = document.getElementById('pilotFreq');
    const msgFreq = document.getElementById('msgFreq');
    const carrierFreqVal = document.getElementById('carrierFreqVal');
    const pilotFreqVal = document.getElementById('pilotFreqVal');
    const msgFreqVal = document.getElementById('msgFreqVal');

    function updateLabels() {
        carrierFreqVal.textContent = carrierFreq.value;
        pilotFreqVal.textContent = pilotFreq.value;
        msgFreqVal.textContent = msgFreq.value;
    }

    function generateSignals() {
        const fs = 10000; // sample rate
        const t = [];
        const N = 1000;
        for (let i = 0; i < N; i++) t.push(i / fs);

        const fc = parseFloat(carrierFreq.value);
        const fp = parseFloat(pilotFreq.value);
        const fm = parseFloat(msgFreq.value);

        // Message signal (sine)
        const msg = t.map(time => Math.sin(2 * Math.PI * fm * time));
        // Carrier
        const carrier = t.map(time => Math.cos(2 * Math.PI * fc * time));
        // Pilot
        const pilot = t.map(time => 0.2 * Math.cos(2 * Math.PI * fp * time));
        // Modulated signal (AM with pilot)
        const modulated = t.map((time, i) =>
            (1 + msg[i]) * carrier[i] + pilot[i]
        );

        return { t, msg, carrier, pilot, modulated };
    }

    let chart = null;
    function plot() {
        const ctx = document.getElementById('modulationChart');
        if (!ctx || !ctx.getContext) return;
        const { t, msg, carrier, pilot, modulated } = generateSignals();
        const labels = t.map(time => time.toFixed(4));
        if (chart) chart.destroy();
        chart = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Message', data: msg, borderColor: 'blue', fill: false, pointRadius: 0 },
                    { label: 'Carrier', data: carrier, borderColor: 'green', fill: false, pointRadius: 0 },
                    { label: 'Pilot', data: pilot, borderColor: 'orange', fill: false, pointRadius: 0 },
                    { label: 'Modulated', data: modulated, borderColor: 'red', fill: false, pointRadius: 0 }
                ]
            },
            options: {
                animation: false,
                elements: { line: { tension: 0 } },
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: false },
                    y: { min: -2, max: 2 }
                }
            }
        });
        window.modulationChart = chart;
    }

    if (carrierFreq && pilotFreq && msgFreq) {
        carrierFreq.oninput = pilotFreq.oninput = msgFreq.oninput = function() {
            updateLabels();
            plot();
        };
        updateLabels();
        setTimeout(plot, 200);
    }
});
