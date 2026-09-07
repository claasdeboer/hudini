"use strict";
// The one data-source switch. A bake sets window.HUDINI_EMBEDDED to
// {video, intervals}; the served app leaves it null and talks to its
// own local server. Nothing else in the app knows which mode it is in.
// The served source also remembers the last listing in localStorage,
// so a fresh page paints the previous inventory while the fetch runs.

const LISTING_KEY = "hudini.index.listing";

function makeDataSource() {
  const embedded = window.HUDINI_EMBEDDED;
  if (embedded) {
    return {
      embedded: true,
      async listVideos() {
        return { directory: null, videos: [embedded.video] };
      },
      async refresh() {
        return this.listVideos();
      },
      cachedListing() {
        return null;
      },
      saveListing() {},
      async intervals() {
        return embedded.intervals;
      },
      videoUrl(name) {
        return embedded.video_file || `${name}.mp4`;
      },
    };
  }
  return {
    embedded: false,
    async listVideos(query) {
      const url = query ? `/api/videos?q=${encodeURIComponent(query)}` : "/api/videos";
      const response = await fetch(url);
      return response.json();
    },
    async refresh() {
      const response = await fetch("/api/refresh", { method: "POST" });
      return response.json();
    },
    cachedListing() {
      try {
        return JSON.parse(window.localStorage.getItem(LISTING_KEY));
      } catch {
        return null;
      }
    },
    saveListing(listing) {
      try {
        window.localStorage.setItem(LISTING_KEY, JSON.stringify(listing));
      } catch {
        // A full or unavailable localStorage only costs the warm start.
      }
    },
    async intervals(name) {
      const response = await fetch(`/api/intervals/${encodeURIComponent(name)}`);
      if (!response.ok) return [];
      return response.json();
    },
    videoUrl(name) {
      return `/video/${encodeURIComponent(name)}`;
    },
  };
}
