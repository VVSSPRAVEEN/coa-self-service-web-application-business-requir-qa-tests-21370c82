import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright E2E configuration for the COA portal.
 *
 * Self-contained contract: the e2e suite runs against the compiled
 * frontend (frontend/dist) served by Vite's preview server, so no
 * backend is required for the portal UI smoke tests.
 *
 * frontend/dist must be the bootstrap-gate bundle, i.e. built with
 * VITE_AUTH_STUB_ENABLED=true (see webServer.env below); a production
 * `vite build` without either that flag or Cognito refuses to compile.
 */
export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: "html",
  use: {
    baseURL: "http://localhost:4173",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },
  ],
  webServer: {
    command: "npm run build && npm run preview -- --port 4173 --strictPort",
    cwd: "../frontend",
    port: 4173,
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
    // The suite exercises the non-OIDC bootstrap gate, so the served bundle has
    // to be the one compiled with VITE_AUTH_STUB_ENABLED=true ? the same opt-in
    // frontend/vite.config.ts requires of any production `vite build` without
    // Cognito. Serving the stub bundle is what puts "Operator identity" on
    // screen for signIn(). Set here as well as at build time so the server this
    // config boots can never disagree with how the bundle was compiled.
    env: {
      VITE_AUTH_STUB_ENABLED: "true",
    },
  },
});
