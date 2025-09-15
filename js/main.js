// js/main.js (clean, Swup + reliable night-mode across pages)
(function () {
  // Shorthands
  const qs  = (sel, root = document) => root.querySelector(sel);
  const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const isExternalHref = (href) => /^(?:[a-z]+:)?\/\//i.test(href);
  const isHomePage = () => !!document.querySelector('.intro'); // intro exists only on home
  const SCROLL_KEY = 'homeScrollY';

  // -------------------------------
  // Night mode (auto + toggle + persistence; binds only once)
  // -------------------------------
  function initNightMode() {
    const body    = document.body;
    const html    = document.documentElement;
    const input   = qs('#switch'); // the checkbox
    const wrapper = qs('#toggle') || qs('.toggle-wrapper') || qs('.switch-wrapper');

    if (!body || !html || !input) return;

    function updateFooterIcons(isNight) {
      // Update ALL social icons (footer + intro)
      document.querySelectorAll('img.social-icon, .intro__photo').forEach(img => {
        const next = isNight ? img.dataset.dark : img.dataset.light;
        if (next) img.setAttribute('src', next);
      });
    }

    // Prevent duplicate listeners when Swup swaps content (toggle lives outside #swup)
    if (!input.dataset.bound) {
      input.dataset.bound = '1';
      input.addEventListener('change', () => {
        applyTheme(input.checked);
      });
    }
    if (wrapper && !wrapper.dataset.bound) {
      wrapper.dataset.bound = '1';
      wrapper.addEventListener('click', (ev) => {
        if (ev.target && (ev.target.id === 'switch' || ev.target.tagName === 'LABEL')) return;
        input.checked = !input.checked;
        input.dispatchEvent(new Event('change', { bubbles: true }));
      });
    }

    function applyTheme(isNight) {
      body.classList.toggle('night', isNight);
      html.classList.toggle('night', isNight);
      updateFooterIcons(isNight);
      try { localStorage.setItem('theme', isNight ? 'night' : 'day'); } catch (_) {}
    }

    // Load saved preference first; else auto 7pm–7am
    let saved = null;
    try { saved = localStorage.getItem('theme'); } catch (_) {}
    if (saved === 'night') {
      input.checked = true;  applyTheme(true);
    } else if (saved === 'day') {
      input.checked = false; applyTheme(false);
    } else {
      const hours = new Date().getHours();
      const night = hours >= 19 || hours <= 7;
      input.checked = night;
      applyTheme(night);
    }
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

    // Apply standard reveal to homepage sections
    const homeSections = [
      '.intro',
      '.profile',
      '.projects',
      '.certifications',
      '.professional-certificates',
      '.skills',
      '.internship',
      '.education',
      '.conferences',
      '.publications',
      '.awards',
      '.leadership'
    ];
                         
    homeSections.forEach(sel => {
      if (qs(sel)) sr.reveal(sel, { viewFactor: 0.1 });
    });
    
    // Apply IMMEDIATE reveal to subpage content (no scroll required)
    if (qs('.project-detail')) {
      sr.reveal('.project-detail', { 
        delay: 0,
        viewFactor: 0,
        duration: 300
      });
      
      sr.reveal('.project-hero', { 
        delay: 0,
        viewFactor: 0,
        duration: 300
      });
    }
  }

  // -------------------------------
  // External links in new tab
  // -------------------------------
  function initExternalLinks() {
    // Apply target="_blank" to all external links
    qsa('a').forEach(link => {
      const href = link.getAttribute('href') || '';
      
      // Skip links with no href or that already have target set
      if (!href || link.hasAttribute('target')) return;
      
      // If external link, open in new tab
      if (isExternalHref(href)) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener');
      }
    });
  }

  // -------------------------------
  // Initialize/re-initialize MathJax
  // -------------------------------
  function initMathJax() {
    if (typeof MathJax !== 'undefined') {
      // If MathJax already loaded, process the new content
      MathJax.typesetPromise && MathJax.typesetPromise();
    } else if (document.querySelector('.equation') || document.body.textContent.includes('$')) {
      // If MathJax not loaded but equations exist, load it
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\(', '\\)']],
          displayMath: [['$$', '$$'], ['\\[', '\\]']],
          processEscapes: true,
          processEnvironments: true
        },
        options: {
          skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
        }
      };
      
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
      script.id = 'MathJax-script';
      script.async = true;
      document.head.appendChild(script);
    }
  }

  // -------------------------------
  // Hero Parallax Effect
  // -------------------------------
  function initParallax() {
    const heroImage = qs('.project-hero img');
    if (!heroImage) return;
    
    // Add parallax class to the container
    const hero = qs('.project-hero');
    hero.classList.add('project-hero--parallax');
    
    // Calculate parallax on scroll
    let ticking = false;
    
    const updateParallax = () => {
      const scrollPosition = window.scrollY;
      const scrollRange = window.innerHeight;
      const moveAmount = (scrollPosition / scrollRange) * 32; // Adjust intensity
      
      // Move the image based on scroll position
      heroImage.style.transform = `translateY(${moveAmount}%)`;
      
      ticking = false;
    };
    
    // Call immediately to set initial position
    updateParallax();
    
    // Use requestAnimationFrame for smooth scrolling
    window.addEventListener('scroll', () => {
      if (!ticking) {
        window.requestAnimationFrame(updateParallax);
        ticking = true;
      }
    });
  }

  // -------------------------------
  // Swup (internal nav + restore home scroll)
  // -------------------------------
  function initSwup() {
    if (!window.Swup) return;

    const swup = new Swup({ containers: ['#swup'] });

    swup.hooks.before('animation:out:start', () => {
      document.documentElement.classList.add('is-animating');
    });
    swup.hooks.on('animation:in:end', () => {
      document.documentElement.classList.remove('is-animating');
    });

    // Save scroll when leaving the homepage via an internal link
    swup.hooks.on('link:click', ({ el }) => {
      const href = el && el.getAttribute && el.getAttribute('href') || '';
      if (!isExternalHref(href) && isHomePage()) {
        try { sessionStorage.setItem(SCROLL_KEY, String(window.scrollY)); } catch (_) {}
      }
    });

    // Re-init page widgets after content swap
    swup.hooks.on('content:replace', () => {
      initWidgets(); // This already includes initParallax
    });

    // On every page view (including Back), restore home scroll if saved
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

    // Let Swup manage history scrolling
    try { history.scrollRestoration = 'manual'; } catch (_) {}
  }

  // -------------------------------
  // Init all widgets on the current DOM
  // -------------------------------
  function initWidgets() {
    try { initNightMode(); }      catch (e) { console.warn('NightMode init failed', e); }
    try { initWaveHand(); }       catch (e) { console.warn('Wave init failed', e); }
    try { initScrollReveal(); }   catch (e) { console.warn('ScrollReveal init failed', e); }
    try { initExternalLinks(); }  catch (e) { console.warn('ExternalLinks init failed', e); }
    try { initMathJax(); }        catch (e) { console.warn('MathJax init failed', e); }
    try { initParallax(); }       catch (e) { console.warn('Parallax init failed', e); }
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