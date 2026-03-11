/* ============================================
   VØX — Main JS
   Interactivity, animations, scroll effects
   ============================================ */

import './style.css'

// ── Nav scroll effect ──
const nav = document.getElementById('nav')
let lastScroll = 0

window.addEventListener('scroll', () => {
    const currentScroll = window.scrollY
    if (currentScroll > 50) {
        nav.classList.add('scrolled')
    } else {
        nav.classList.remove('scrolled')
    }
    lastScroll = currentScroll
}, { passive: true })

// ── Smooth scroll for anchor links ──
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
        e.preventDefault()
        const target = document.querySelector(anchor.getAttribute('href'))
        if (target) {
            const navHeight = nav.offsetHeight
            const targetPosition = target.getBoundingClientRect().top + window.scrollY - navHeight - 20
            window.scrollTo({ top: targetPosition, behavior: 'smooth' })
        }
    })
})

// ── Speed bar animation ──
const barSlow = document.getElementById('barSlow')
const barFast = document.getElementById('barFast')
const speedCard = document.getElementById('speedCard')

const speedObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            setTimeout(() => {
                barSlow.classList.add('animate')
                barFast.classList.add('animate')
            }, 300)
            speedObserver.unobserve(entry.target)
        }
    })
}, { threshold: 0.3 })

if (speedCard) speedObserver.observe(speedCard)

// ── Counter animation ──
function animateCounter(element, target, duration = 1500) {
    let start = 0
    const startTime = performance.now()

    function update(currentTime) {
        const elapsed = currentTime - startTime
        const progress = Math.min(elapsed / duration, 1)
        // Ease out quad
        const eased = 1 - (1 - progress) * (1 - progress)
        const current = Math.round(start + (target - start) * eased)
        element.textContent = current
        if (progress < 1) {
            requestAnimationFrame(update)
        }
    }

    requestAnimationFrame(update)
}

const speedCounter = document.getElementById('speedCounter')
const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            setTimeout(() => animateCounter(speedCounter, 5, 1200), 500)
            counterObserver.unobserve(entry.target)
        }
    })
}, { threshold: 0.3 })

if (speedCounter) {
    speedCounter.textContent = '0'
    counterObserver.observe(speedCounter)
}

// ── Scroll reveal ──
const revealElements = document.querySelectorAll('.reveal')

const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible')
            revealObserver.unobserve(entry.target)
        }
    })
}, {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
})

revealElements.forEach((el, i) => {
    el.style.transitionDelay = `${i % 3 * 100}ms`
    revealObserver.observe(el)
})

// ── FAQ accordion ──
document.querySelectorAll('.faq-question').forEach(question => {
    question.addEventListener('click', () => {
        const item = question.parentElement
        const isOpen = item.classList.contains('open')

        // Close all
        document.querySelectorAll('.faq-item').forEach(faq => {
            faq.classList.remove('open')
        })

        // Open clicked if it wasn't open
        if (!isOpen) {
            item.classList.add('open')
        }
    })
})

// ── Mobile nav toggle ──
const navToggle = document.getElementById('navToggle')
const navLinks = document.querySelector('.nav-links')

if (navToggle) {
    navToggle.addEventListener('click', () => {
        navLinks.classList.toggle('active')
        navToggle.classList.toggle('active')
    })
}

// ── Typing effect for demo ──
function typeWriter(element, text, speed = 30) {
    let i = 0
    element.textContent = ''

    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i)
            i++
            setTimeout(type, speed)
        }
    }

    type()
}

const outputText = document.querySelector('.demo-card.output .demo-speech')
const originalText = outputText ? outputText.textContent : ''

const demoObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && outputText) {
            setTimeout(() => typeWriter(outputText, originalText, 25), 800)
            demoObserver.unobserve(entry.target)
        }
    })
}, { threshold: 0.3 })

const demoSection = document.getElementById('demo')
if (demoSection) demoObserver.observe(demoSection)

    // ── VØX Interactive LED Visualizer ──
    ; (function initVoxVisualizer() {
        const TEAL = [0, 240, 255]
        const GRID_SIZE = 7
        const CENTER = 3

        const grid = document.getElementById('voxLedGrid')
        const capsule = document.getElementById('voxCapsule')
        const micBtn = document.getElementById('tryMicBtn')
        const slider = document.getElementById('trySlider')
        const statusText = document.querySelector('.vox-status-text')

        if (!grid || !capsule) return

        // Diamond mask: which cells can light up
        function getDiamondMask() {
            const mask = []
            for (let row = 0; row < GRID_SIZE; row++) {
                mask[row] = []
                const dist = Math.abs(row - CENTER)
                const activeCols = GRID_SIZE - (dist * 2)
                const startCol = dist
                for (let col = 0; col < GRID_SIZE; col++) {
                    mask[row][col] = col >= startCol && col < startCol + activeCols
                }
            }
            return mask
        }

        const diamondMask = getDiamondMask()
        const bricks = []

        // Build 7×7 grid
        for (let row = 0; row < GRID_SIZE; row++) {
            bricks[row] = []
            for (let col = 0; col < GRID_SIZE; col++) {
                const brick = document.createElement('div')
                brick.className = 'vox-led-brick'
                grid.appendChild(brick)
                bricks[row][col] = brick
            }
        }

        let currentAmplitude = 0
        let targetAmplitude = 0
        let useMic = false
        let analyser = null
        let dataArray = null

        function getRowOpacity(row, amplitude) {
            const dist = Math.abs(row - CENTER)
            const threshold = dist / CENTER
            if (amplitude <= threshold * 0.3) return 0
            const rowIntensity = Math.max(0, 1 - (dist * 0.22))
            const ampFactor = Math.min(1, (amplitude - threshold * 0.2) / 0.6)
            return rowIntensity * ampFactor
        }

        function updateGrid(amplitude) {
            if (amplitude > 0.05) {
                capsule.classList.add('active')
                if (statusText) statusText.textContent = 'Dictating...'
            } else {
                capsule.classList.remove('active')
                if (statusText) statusText.textContent = 'Ready'
            }

            for (let row = 0; row < GRID_SIZE; row++) {
                const rowOpacity = getRowOpacity(row, amplitude)
                const isCenterRow = row === CENTER

                for (let col = 0; col < GRID_SIZE; col++) {
                    const brick = bricks[row][col]
                    const inDiamond = diamondMask[row][col]

                    if (!inDiamond) {
                        brick.style.background = `rgba(${TEAL.join(',')}, 0.03)`
                        brick.style.boxShadow = 'none'
                        continue
                    }

                    // Center row always has soft idle glow
                    if (isCenterRow && amplitude <= 0.05) {
                        const idleGlow = 0.12 + Math.sin(Date.now() / 800 + col * 0.3) * 0.03
                        brick.style.background = `rgba(${TEAL.join(',')}, ${idleGlow})`
                        brick.style.boxShadow = `0 0 4px rgba(${TEAL.join(',')}, ${idleGlow * 0.3})`
                        continue
                    }

                    if (rowOpacity > 0) {
                        const variation = Math.sin(Date.now() / 120 + col * 1.7 + row * 2.3) * 0.15
                        const colDist = Math.abs(col - CENTER)
                        const colFade = 1 - (colDist * 0.05)
                        const finalOpacity = Math.min(1, Math.max(0, rowOpacity * colFade + variation * amplitude))
                        const bg = finalOpacity * 0.7 + 0.06
                        const glow = finalOpacity * 0.5
                        brick.style.background = `rgba(${TEAL.join(',')}, ${bg.toFixed(3)})`
                        brick.style.boxShadow = glow > 0.1
                            ? `0 0 ${6 + glow * 8}px rgba(${TEAL.join(',')}, ${(glow * 0.6).toFixed(3)})`
                            : 'none'
                    } else {
                        brick.style.background = `rgba(${TEAL.join(',')}, 0.05)`
                        brick.style.boxShadow = 'none'
                    }
                }
            }
        }

        function animate() {
            currentAmplitude += (targetAmplitude - currentAmplitude) * 0.25
            if (useMic && analyser) {
                analyser.getByteFrequencyData(dataArray)
                let sum = 0
                for (let i = 0; i < dataArray.length; i++) sum += dataArray[i]
                const avg = sum / dataArray.length / 255
                targetAmplitude = Math.min(1, Math.pow(avg, 0.6) * 2.5)
            }
            updateGrid(currentAmplitude)
            requestAnimationFrame(animate)
        }

        // Mic button
        if (micBtn) {
            micBtn.addEventListener('click', async () => {
                if (useMic) {
                    useMic = false
                    micBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg> Enable Mic`
                    micBtn.classList.remove('recording')
                    targetAmplitude = 0
                    return
                }
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
                    const audioCtx = new AudioContext()
                    const source = audioCtx.createMediaStreamSource(stream)
                    analyser = audioCtx.createAnalyser()
                    analyser.fftSize = 256
                    analyser.smoothingTimeConstant = 0.7
                    source.connect(analyser)
                    dataArray = new Uint8Array(analyser.frequencyBinCount)
                    useMic = true
                    micBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg> Mic On ●`
                    micBtn.classList.add('recording')
                } catch (err) {
                    console.error('Mic access denied:', err)
                    micBtn.textContent = 'Mic Denied'
                }
            })
        }

        // Slider
        if (slider) {
            slider.addEventListener('input', (e) => {
                if (!useMic) targetAmplitude = e.target.value / 100
            })
        }

        animate()
    })()

console.log('⚡ VØX powered by AXIOM — Ø⟳')

