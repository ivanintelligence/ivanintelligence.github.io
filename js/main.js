// js/main.js (clean, with Swup scroll-restore to homepage)
(function () {
  // Shorthands
  const qs  = (sel, root = document) => root.querySelector(sel);
  const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const isExternalHref = (href) => /^(?:[a-z]+:)?\/\//i.test(href);
  const isHomePage = () => !!document.querySelector('.intro'); // intro exists only on home
  const SCROLL_KEY = 'homeScrollY';

  // -------------------------------
  // Night mode (auto + toggle + persistence, no double-toggle)
  // -------------------------------
  function initNightMode() {
    const body    = document.body;
    const input   = qs('#switch'); // the checkbox
    const wrapper = qs('#toggle') || qs('.toggle-wrapper') || qs('.switch-wrapper');

    if (!body || !input) return;

    // Load saved preference first (if any)
    const saved = localStorage.getItem('theme'); // 'night' | 'day' | null
    if (saved === 'night') {
      input.checked = true;
      body.classList.add('night');
    } else if (saved === 'day') {
      input.checked = false;
      body.classList.remove('night');
    } else {
      // No saved preference → auto-night 7pm–7am (first load only)
      const hours = new Date().getHours();
      const night = hours >= 19 || hours <= 7;
      input.checked = night;
      body.classList.toggle('night', night);
    }

    // Click the wrapper (not the checkbox) to toggle once
    if (wrapper) {
      wrapper.addEventListener('click', (ev) => {
        if (ev.target && ev.target.id === 'switch') return; // let native change happen
        ev.preventDefault();
        input.click(); // fires 'change'
      });
    }

    input.addEventListener('change', () => {
      const isNight = input.checked;
      document.body.classList.toggle('night', isNight);
      localStorage.setItem('theme', isNight ? 'night' : 'day');
    });
  }

  // -------------------------------
  // Back-to-top button
  // -------------------------------
  function initTopButton() {
    const topButton = qs('#top-button');
    if (!topButton) return;

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

    topButton.addEventListener('click', () => {
      if (window.jQuery) jQuery('html, body').animate({ scrollTop: 0 }, 500);
      else window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // -------------------------------
  // Hand wave emoji
  // -------------------------------
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

  // -------------------------------
  // ScrollReveal fade-ins
  // -------------------------------
  function initScrollReveal() {
    if (!window.ScrollReveal) return;

    const sr = ScrollReveal({
      reset: false,
      duration: 600,
      easing: 'cubic-bezier(.694,0,.335,1)',
      scale: 1,
      viewFactor: 0.3,
    });

    [
      '.background',
      '.skills',
      '.experience',
      '.featured-projects',
      '.other-projects',
      '.certifications',
      '.project-detail',
      '.project-hero'
    ].forEach(sel => {
      if (qs(sel)) sr.reveal(sel, { viewFactor: 0.1 });
    });
  }

  // -------------------------------
  // Swup setup (handles internal nav + restore scroll to where you left on home)
  // -------------------------------
  function initSwup() {
    if (!window.Swup) return;

    const swup = new Swup({ containers: ['#swup'] });

    // Add/remove a CSS flag if you later add page-transition CSS
    swup.hooks.before('animation:out:start', () => {
      document.documentElement.classList.add('is-animating');
    });
    swup.hooks.on('animation:in:end', () => {
      document.documentElement.classList.remove('is-animating');
    });

    // Save scroll when leaving the homepage via an internal link
    swup.hooks.on('link:click', ({ el }) => {
      const href = el && el.getAttribute && el.getAttribute('href') || '';
      const external = isExternalHref(href);
      if (!external && isHomePage()) {
        try { sessionStorage.setItem(SCROLL_KEY, String(window.scrollY)); } catch (_) {}
      }
    });

    // Re-init widgets after every swap
    swup.hooks.on('content:replace', () => {
      initWidgets();
    });

    // On every page view (including browser Back), restore home scroll if saved
    swup.hooks.on('page:view', () => {
      if (!isHomePage()) return;
      const yStr = sessionStorage.getItem(SCROLL_KEY);
      if (!yStr) return;
      const y = parseInt(yStr, 10);
      if (Number.isNaN(y)) { sessionStorage.removeItem(SCROLL_KEY); return; }

      let tries = 0;
      const attempt = () => {
        window.scrollTo(0, y);
        if (Math.abs(window.scrollY - y) > 2 && tries < 8) {
          tries++;
          setTimeout(attempt, 60);
        } else {
          sessionStorage.removeItem(SCROLL_KEY);
        }
      };
      requestAnimationFrame(attempt);
    });

    // Let Swup own history scrolling
    try { history.scrollRestoration = 'manual'; } catch (_) {}
  }

  // -------------------------------
  // Init all widgets on the current DOM
  // -------------------------------
  function initWidgets() {
    try { initNightMode(); }     catch (e) { console.warn('NightMode init failed', e); }
    try { initTopButton(); }     catch (e) { console.warn('TopButton init failed', e); }
    try { initWaveHand(); }      catch (e) { console.warn('Wave init failed', e); }
    try { initScrollReveal(); }  catch (e) { console.warn('ScrollReveal init failed', e); }
  }

  // -------------------------------
  // First load
  // -------------------------------
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      initWidgets();
      initSwup();
    }, { once: true });
  } else {
    initWidgets();
    initSwup();
  }
})();