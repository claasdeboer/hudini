"use strict";
// The router: hash-based, two views, one data source. A bake carries
// exactly one video, so the embedded app opens on its timeline and has
// no index to go back to.

(function main() {
  const data = makeDataSource();
  const root = document.getElementById("app");
  let teardown = null;

  function route() {
    if (teardown) teardown();
    const match = window.location.hash.match(/^#\/v\/(.+)$/);
    if (data.embedded) {
      const name = window.HUDINI_EMBEDDED.video.name;
      teardown = renderTimelineView(root, data, name, null);
      return;
    }
    if (match) {
      const name = decodeURIComponent(match[1]);
      teardown = renderTimelineView(root, data, name, () => {
        window.location.hash = "";
      });
      return;
    }
    teardown = renderIndexView(root, data, (name) => {
      window.location.hash = `#/v/${encodeURIComponent(name)}`;
    });
  }

  window.addEventListener("hashchange", route);
  route();
})();
