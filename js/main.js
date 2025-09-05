// js/main.js (robust version that works on all pages)
(function () {
  function qs(sel, root = document) { return root.querySelector(sel); }
  function qsa(sel, root = document) { return Array.from(root.querySelectorAll(sel)); }

  // ---- Night mode (auto + toggle) ----
  // ---- Night mode (auto + toggle + persistence, no double-toggle) ----
function initNightMode() {
  const body   = document.body;
  const input  = document.querySelector('#switch');           // the checkbox
  const wrapper = document.querySelector('#toggle')           // optional wrapper
                || document.querySelector('.toggle-wrapper')  // your current markup
                || document.querySelector('.switch-wrapper'); // fallback

  if (!body || !input) return;

  // A) Load saved preference first (if any)
  const saved = localStorage.getItem('theme'); // 'night' | 'day' | null
  if (saved === 'night') {
    input.checked = true;
    body.classList.add('night');
  } else if (saved === 'day') {
    input.checked = false;
    body.classList.remove('night');
  } else {
    // B) No saved preference → auto-night 7pm–7am (first load only)
    const hours = new Date().getHours();
    const night = hours >= 19 || hours <= 7;
    input.checked = night;
    body.classList.toggle('night', night);
  }

  // C) Clicking the wrapper should NOT flip twice
  if (wrapper) {
    wrapper.addEventListener('click', (ev) => {
      // If you clicked the checkbox itself, let the browser handle it.
      if (ev.target && ev.target.id === 'switch') return;

      // If you clicked label/wrapper, prevent label’s native toggle and forward once.
      ev.preventDefault();
      input.click(); // this fires the 'change' handler below
    });
  }

  // D) React to checkbox changes (mouse/keyboard) and persist preference
  input.addEventListener('change', () => {
    const isNight = input.checked;
    body.classList.toggle('night', isNight);
    localStorage.setItem('theme', isNight ? 'night' : 'day');
  });
}

  // ---- Back-to-top button ----
  function initTopButton() {
    const topButton = qs('#top-button');
    if (!topButton) return;

    // Determine where to start showing the button
    const intro = qs('.intro');
    const introHeight = intro ? intro.offsetHeight : 0;

    function show(btn) {
      if (window.jQuery) { jQuery(btn).fadeIn(); } else { btn.style.display = 'block'; }
    }
    function hide(btn) {
      if (window.jQuery) { jQuery(btn).fadeOut(); } else { btn.style.display = 'none'; }
    }

    function onScroll() {
      if (window.scrollY > introHeight) show(topButton);
      else hide(topButton);
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    topButton.addEventListener('click', function () {
      if (window.jQuery) {
        jQuery('html, body').animate({ scrollTop: 0 }, 500);
      } else {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  }

  // ---- Hand wave emoji ----
  function initWaveHand() {
    const hand = qs('.emoji.wave-hand');
    if (!hand) return;

    function waveOnLoad() {
      hand.classList.add('wave');
      setTimeout(() => hand.classList.remove('wave'), 2000);
    }
    setTimeout(waveOnLoad, 1000);

    hand.addEventListener('mouseover', () => hand.classList.add('wave'));
    hand.addEventListener('mouseout',  () => hand.classList.remove('wave'));
  }

  // ---- ScrollReveal fade-ins ----
  function initScrollReveal() {
    if (!window.ScrollReveal) return;

    const sr = ScrollReveal({
      reset: false,
      duration: 600,
      easing: 'cubic-bezier(.694,0,.335,1)',
      scale: 1,
      viewFactor: 0.3,
    });

    // Only reveal what actually exists on the current page
    [
      '.background',
      '.skills',
      '.experience',
      '.featured-projects',
      '.other-projects',
      '.certifications',
      '.project-detail', // project subpage column
      '.project-hero'    // project subpage hero
    ].forEach(sel => {
      if (qs(sel)) sr.reveal(sel, { viewFactor: 0.1 });
    });
  }

  // ---- Boot everything safely ----
  function init() {
    try { initNightMode(); }     catch (e) { console.warn('NightMode init failed', e); }
    try { initTopButton(); }     catch (e) { console.warn('TopButton init failed', e); }
    try { initWaveHand(); }      catch (e) { console.warn('Wave init failed', e); }
    try { initScrollReveal(); }  catch (e) { console.warn('ScrollReveal init failed', e); }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();