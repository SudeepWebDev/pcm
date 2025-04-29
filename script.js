document.addEventListener('DOMContentLoaded', function() {
    // Hamburger menu functionality
    const navToggle = document.getElementById('navToggle');
    const sideNav = document.getElementById('sideNav');
    const overlay = document.getElementById('overlay');
    
    navToggle.addEventListener('click', toggleNav);
    overlay.addEventListener('click', closeNav);
    
    // Close menu when clicking a nav link (for better mobile UX)
    document.querySelectorAll('.side-nav a').forEach(link => {
        link.addEventListener('click', closeNav);
    });
    
    function toggleNav() {
        navToggle.classList.toggle('active');
        sideNav.classList.toggle('active');
        
        if (sideNav.classList.contains('active')) {
            document.body.style.overflow = 'hidden'; // Prevent scrolling while menu is open
            overlay.style.display = 'block';
        } else {
            closeNav();
        }
    }
    
    function closeNav() {
        navToggle.classList.remove('active');
        sideNav.classList.remove('active');
        document.body.style.overflow = '';
        overlay.style.display = 'none';
    }
    
    // Check if Canva iframe loads properly
    const canvaFrame = document.getElementById('canvaFrame');
    const canvaFallback = document.getElementById('canvaFallback');
    const canvaContainer = document.getElementById('canvaContainer');
    
    // Set a timeout to check if iframe loaded
    setTimeout(function() {
        try {
            // Try to access iframe content - this will fail if CSP blocks it
            const frameContent = canvaFrame.contentWindow || canvaFrame.contentDocument;
            if (!frameContent) {
                showFallback();
            }
        } catch (e) {
            // If any error occurs, show fallback
            showFallback();
        }
    }, 3000); // Check after 3 seconds
    
    // Also listen for iframe load errors
    canvaFrame.onerror = showFallback;
    
    function showFallback() {
        canvaContainer.style.display = 'none';
        canvaFallback.style.display = 'block';
    }
    
    // Existing code for the demo functionality
    const carrierFreq = document.getElementById('carrierFreq');
    const pilotFreq = document.getElementById('pilotFreq');
    const msgFreq = document.getElementById('msgFreq');
    const quantBits = document.getElementById('quantBits');
    const carrierFreqVal = document.getElementById('carrierFreqVal');
    const pilotFreqVal = document.getElementById('pilotFreqVal');
    const msgFreqVal = document.getElementById('msgFreqVal');
    const quantBitsVal = document.getElementById('quantBitsVal');
    const calcElements = {
        carrierFreq: document.getElementById('calcCarrierFreq'),
        pilotFreq: document.getElementById('calcPilotFreq'),
        msgFreq: document.getElementById('calcMsgFreq'),
        eqMsgFreq: document.getElementById('eqMsgFreq'),
        eqMsgFreq2: document.getElementById('eqMsgFreq2'),
        eqMsgFreq3: document.getElementById('eqMsgFreq3'),
        eqCarrierFreq: document.getElementById('eqCarrierFreq'),
        eqCarrierFreq2: document.getElementById('eqCarrierFreq2'),
        eqPilotFreq: document.getElementById('eqPilotFreq'),
        powerEfficiency: document.getElementById('powerEfficiency'),
        powerRatio: document.getElementById('powerRatio'),
        msgBandwidth: document.getElementById('msgBandwidth'),
        pilotComponents: document.getElementById('pilotComponents'),
        noPilotComponents: document.getElementById('noPilotComponents')
    };
    function updateLabels() {
        carrierFreqVal.textContent = carrierFreq.value;
        pilotFreqVal.textContent = pilotFreq.value;
        msgFreqVal.textContent = msgFreq.value;
        quantBitsVal.textContent = quantBits.value;
        updateMathCalculations();
    }
    function updateMathCalculations() {
        const fc = parseInt(carrierFreq.value);
        const fp = parseInt(pilotFreq.value);
        const fm = parseInt(msgFreq.value);
        updateWithHighlight(calcElements.carrierFreq, fc);
        updateWithHighlight(calcElements.pilotFreq, fp);
        updateWithHighlight(calcElements.msgFreq, fm);
        updateWithHighlight(calcElements.eqCarrierFreq, fc);
        updateWithHighlight(calcElements.eqCarrierFreq2, fc);
        updateWithHighlight(calcElements.eqPilotFreq, fp);
        updateWithHighlight(calcElements.eqMsgFreq, fm);
        updateWithHighlight(calcElements.eqMsgFreq2, fm);
        updateWithHighlight(calcElements.eqMsgFreq3, fm);
        const powerRatio = Math.round(Math.pow(10, 16/10));
        const efficiency = (1 / (1 + 1/powerRatio) * 100).toFixed(1);
        updateWithHighlight(calcElements.powerEfficiency, efficiency);
        updateWithHighlight(calcElements.powerRatio, powerRatio);
        updateWithHighlight(calcElements.msgBandwidth, fm);
        updateWithHighlight(calcElements.pilotComponents, `${fc} ± ${fm} Hz, ${fp} Hz`);
        updateWithHighlight(calcElements.noPilotComponents, `${fc} ± ${fm} Hz`);
    }
    function updateWithHighlight(element, value) {
        if (element) {
            if (element.textContent != value) {
                element.textContent = value;
                element.classList.add('highlight');
                setTimeout(() => {
                    element.classList.remove('highlight');
                }, 1000);
            }
        }
    }
    function quantizeSignal(signal, bits) {
        const levels = Math.pow(2, bits);
        const min = Math.min(...signal);
        const max = Math.max(...signal);
        const step = (max - min) / (levels - 1);
        return signal.map(x => {
            const q = Math.round((x - min) / step) * step + min;
            return q;
        });
    }
    function generateSignals() {
        const fs = 10000; // sample rate
        const t = [];
        const N = 1000;
        for (let i = 0; i < N; i++) t.push(i / fs);

        const fc = parseFloat(carrierFreq.value);
        const fp = parseFloat(pilotFreq.value);
        const fm = parseFloat(msgFreq.value);
        const bits = parseInt(quantBits.value);

        const msg = t.map(time => Math.sin(2 * Math.PI * fm * time));
        const quantMsg = quantizeSignal(msg, bits);
        const quantNoise = msg.map((v, i) => quantMsg[i] - v);
        const carrier = t.map(time => Math.cos(2 * Math.PI * fc * time));
        const pilot = t.map(time => 0.2 * Math.cos(2 * Math.PI * fp * time));

        const modulatedWithPilot = t.map((time, i) => (1 + msg[i]) * carrier[i] + pilot[i]);
        const modulatedWithoutPilot = t.map((time, i) => (1 + msg[i]) * carrier[i]);

        return { t, msg, quantMsg, quantNoise, carrier, pilot, modulatedWithPilot, modulatedWithoutPilot };
    }

    let chart1 = null;
    let chart2 = null;
    let chart3 = null;
    function plot() {
        const ctx1 = document.getElementById('modulationChart');
        const ctx2 = document.getElementById('modulationChartNoPilot');
        const ctx3 = document.getElementById('quantizationChart');
        if (!ctx1 || !ctx2 || !ctx1.getContext || !ctx2.getContext) return;
        const { t, msg, quantMsg, quantNoise, carrier, pilot, modulatedWithPilot, modulatedWithoutPilot } = generateSignals();
        const labels = t.map(time => time.toFixed(4));

        if (chart1) chart1.destroy();
        if (chart2) chart2.destroy();
        if (chart3) chart3.destroy();

        chart1 = new Chart(ctx1.getContext('2d'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Message', data: msg, borderColor: 'blue', fill: false, pointRadius: 0 },
                    { label: 'Carrier', data: carrier, borderColor: 'green', fill: false, pointRadius: 0 },
                    { label: 'Pilot', data: pilot, borderColor: 'orange', fill: false, pointRadius: 0 },
                    { label: 'Modulated (With Pilot)', data: modulatedWithPilot, borderColor: 'red', fill: false, pointRadius: 0 }
                ]
            },
            options: {
                animation: false,
                elements: { line: { tension: 0 } },
                responsive: true,
                maintainAspectRatio: false,
                scales: { x: { display: false }, y: { min: -2, max: 2 } }
            }
        });

        chart2 = new Chart(ctx2.getContext('2d'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Modulated (Without Pilot)', data: modulatedWithoutPilot, borderColor: 'purple', fill: false, pointRadius: 0 }
                ]
            },
            options: {
                animation: false,
                elements: { line: { tension: 0 } },
                responsive: true,
                maintainAspectRatio: false,
                scales: { x: { display: false }, y: { min: -2, max: 2 } }
            }
        });

        if (ctx3 && ctx3.getContext) {
            chart3 = new Chart(ctx3.getContext('2d'), {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'Original Message', data: msg, borderColor: 'blue', fill: false, pointRadius: 0 },
                        { label: 'Quantized Message', data: quantMsg, borderColor: 'orange', fill: false, pointRadius: 0 },
                        { label: 'Quantization Noise', data: quantNoise, borderColor: 'red', fill: false, pointRadius: 0 }
                    ]
                },
                options: {
                    animation: false,
                    elements: { line: { tension: 0 } },
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { x: { display: false }, y: { min: -1.2, max: 1.2 } }
                }
            });
        }
    }

    if (carrierFreq && pilotFreq && msgFreq && quantBits) {
        carrierFreq.oninput = pilotFreq.oninput = msgFreq.oninput = quantBits.oninput = function() {
            updateLabels();
            plot();
        };
        updateLabels();
        setTimeout(plot, 200);
    }
    updateMathCalculations();

    // Topic filter logic
    const topicFilter = document.getElementById('topicFilter');
    const topicSections = {
        modulation: [document.getElementById('presentationSection'), document.getElementById('demoPanel'), document.getElementById('exampleSection')],
        quantization: [document.getElementById('demoPanel'), document.getElementById('quantQuizSection')],
        quiz: [document.getElementById('quizSection'), document.getElementById('quantQuizSection')],
        math: [document.getElementById('mathSection')],
        references: [document.getElementById('referencesSection')]
    };
    function showSectionsForTopic(topic) {
        // Hide all topic sections
        document.querySelectorAll('section, .demo-panel').forEach(sec => sec.style.display = 'none');
        if (topic === 'all') {
            document.querySelectorAll('section, .demo-panel').forEach(sec => sec.style.display = '');
        } else {
            (topicSections[topic] || []).forEach(sec => { if (sec) sec.style.display = ''; });
        }
    }
    topicFilter.addEventListener('change', function() {
        showSectionsForTopic(this.value);
    });
    // Show all by default
    showSectionsForTopic('all');

    // Quiz logic
    function setupQuiz() {
        document.querySelectorAll('.quiz-options').forEach(optionsDiv => {
            optionsDiv.querySelectorAll('.option').forEach(option => {
                option.addEventListener('click', function() {
                    // Remove previous selection and feedback
                    optionsDiv.querySelectorAll('.option').forEach(opt => {
                        opt.classList.remove('selected', 'correct', 'incorrect');
                    });
                    const feedback = optionsDiv.parentElement.querySelector('.feedback');
                    feedback.textContent = '';
                    feedback.classList.remove('correct', 'incorrect');

                    // Mark selected
                    option.classList.add('selected');
                    // Show feedback
                    if (option.dataset.correct === "true") {
                        option.classList.add('correct');
                        feedback.textContent = "Correct!";
                        feedback.classList.add('correct');
                    } else {
                        option.classList.add('incorrect');
                        feedback.textContent = "Incorrect. Try again!";
                        feedback.classList.add('incorrect');
                    }
                });
            });
        });
    }

    setupQuiz();

    // Accordion logic for proof/power efficiency (check if exists)
    document.querySelectorAll('.accordion-button').forEach(btn => {
        btn.addEventListener('click', function() {
            const content = btn.nextElementSibling;
            if (content) {
                if (content.classList.contains('active')) {
                    content.classList.remove('active');
                } else {
                    document.querySelectorAll('.accordion-content').forEach(c => c.classList.remove('active'));
                    content.classList.add('active');
                }
            }
        });
    });

    // Redraw charts on window resize for responsiveness
    window.addEventListener('resize', function() {
        // Only plot if all chart containers/canvases exist
        if (
            document.getElementById('modulationChart') &&
            document.getElementById('modulationChartNoPilot') &&
            document.getElementById('quantizationChart')
        ) {
            plot();
        }
    });
});