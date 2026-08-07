(() => {
  const TRANSITION_MS = 180;

  function navigateWithTransition(url) {
    if (document.body.classList.contains('page-leaving')) return;

    document.body.classList.add('page-leaving');
    window.setTimeout(() => window.location.assign(url), TRANSITION_MS);
  }

  window.navigateWithTransition = navigateWithTransition;

  document.addEventListener('click', (event) => {
    const link = event.target.closest('a[href]');
    if (!link || event.defaultPrevented || link.target || link.hasAttribute('download')) return;
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;

    const destination = new URL(link.href, window.location.href);
    const current = new URL(window.location.href);
    const isSamePage = destination.href === current.href ||
      (destination.pathname === current.pathname && destination.search === current.search && destination.hash);

    if (destination.origin !== current.origin || isSamePage) return;

    event.preventDefault();
    navigateWithTransition(destination.href);
  });

  // A browser back/forward restore can retain the leaving class from bfcache.
  window.addEventListener('pageshow', () => document.body.classList.remove('page-leaving'));
})();
