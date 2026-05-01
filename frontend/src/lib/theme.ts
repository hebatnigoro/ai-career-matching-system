"use client";

import { useEffect, useState } from "react";

export type Theme = "light" | "dark";

const STORAGE_KEY = "pd-theme";

/** Read/write the active theme. Persists to localStorage and the
 *  ``data-theme`` attribute on <html>. The pre-hydration script in
 *  layout.tsx applies the persisted value before React mounts.
 *
 *  Hydration: ``theme`` is always ``"light"`` and ``mounted`` is always
 *  ``false`` on the very first render — both on the server and on the
 *  client — so React's hydration tree matches. After mount we read the
 *  *real* theme from the ``data-theme`` attribute that the pre-hydration
 *  script already applied. Components rendering theme-dependent visuals
 *  (icons, colors) should gate on ``mounted`` to avoid a paint flash. */
export function useTheme(): {
  theme: Theme;
  toggle: () => void;
  set: (t: Theme) => void;
  mounted: boolean;
} {
  const [theme, setTheme] = useState<Theme>("light");
  const [mounted, setMounted] = useState(false);

  // Sync from the DOM once on mount. This pulls in whatever the pre-
  // hydration script in layout.tsx applied (persisted choice or
  // prefers-color-scheme fallback) without forcing it during SSR.
  useEffect(() => {
    const attr = document.documentElement.getAttribute("data-theme");
    setTheme(attr === "dark" ? "dark" : "light");
    setMounted(true);
  }, []);

  // After mount, push changes back to the DOM and localStorage.
  useEffect(() => {
    if (!mounted) return;
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // localStorage unavailable (private mode etc.) — ignore.
    }
  }, [theme, mounted]);

  return {
    theme,
    mounted,
    set: setTheme,
    toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")),
  };
}
