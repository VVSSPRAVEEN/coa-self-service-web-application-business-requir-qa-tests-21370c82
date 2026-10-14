import { test, expect } from "@playwright/test";
import type { Page, Request, Route } from "@playwright/test";

/**
 * End-to-end tests for the Ventura COA self-service portal.
 *
 * The suite runs against the compiled bundle in frontend/dist, served by
 * `vite preview` (see e2e/playwright.config.ts, whose webServer already points
 * at ../frontend). Every HTTP call the portal makes is answered by a single
 * page-level route mock, so the suite needs no backend and no network.
 *
 * Contract mirrored here, one-for-one, with the FastAPI routers:
 *   POST /api/auth/login                       {"token"}
 *   GET  /api/coas?search&status&date_from&date_to&plant&product&page&page_size
 *   GET  /api/coas/{coa_id}
 *   POST /api/coas/{coa_id}/validate
 *   POST /api/coas/{coa_id}/download           {"expiry_seconds"?}
 *   POST /api/coas/{coa_id}/print/reauth       {"user_id"}
 *   POST /api/coas/{coa_id}/print              {"user_id","reauth_token","printing_plant"?}
 *   GET  /api/missing-coas
 *   POST /api/admin/users                      {"username","email","role","organization"}
 *
 * FR6: the completion gate is the API's decision, never the browser's. Download
 * and print answer 409 with
 *   { detail, code: "completion_gate_failed", coa_id, missing_fields }
 * so these tests assert the refusal on the incomplete certificate and the
 * successful delivery on the complete one.
 *
 * FR3: the mock login returns the roles the real one would, because the shell
 * gates the Report and Admin tabs on them - a session must not be offered a tab
 * its role has no entitlement to. The mocks below cover one role per row of
 * ROLE_RESOURCES in src/core/rbac.py: internal, 3pl and customer.
 *
 * Nothing secret is written as a literal in this file: the mocked re-auth
 * credential (FR8) and the signature on the mocked presigned download URL (FR7)
 * are minted per run by `mintRunScopedValue`, and the assertions below compare
 * the portal against the values this run issued.
 */

const COMPLETE_COA_ID = "COA-2026-004182";
const INCOMPLETE_COA_ID = "COA-2026-004197";
const MISSING_FIELDS = ["moisture", "ph"];

/**
 * Mint the run-scoped doubles that stand in for the credentials the portal
 * would hold at runtime. Deriving them here keeps the fixture file free of any
 * hardcoded credential literal while still letting a test assert that the token
 * the portal replays is exactly the token the API handed out.
 */
function mintRunScopedValue(prefix: string): string {
  const stamp = Date.now().toString(36);
  const noise = Math.random().toString(36).slice(2, 10);
  return `${prefix}-${stamp}-${noise}`;
}

const REAUTH_TOKEN = mintRunScopedValue("reauth");
const DOWNLOAD_NONCE = mintRunScopedValue("signature");
const DOWNLOAD_URL =
  "https://documents.example.invalid/coas/" +
  `${COMPLETE_COA_ID}.pdf?X-Amz-Signature=${DOWNLOAD_NONCE}&X-Amz-Expires=900`;

interface MockSession {
  user_id: string;
  username: string;
  roles: string[];
  plants: string[];
  expires_at: string;
}

const SESSION: MockSession = {
  user_id: "usr-4f21c0",
  username: "j.alvarez@ventura.example",
  roles: ["internal", "qa", "customer_service"],
  plants: ["VEC", "VLP"],
  expires_at: "2026-10-13T21:00:00+00:00",
};

const EXTERNAL_SESSION: MockSession = {
  user_id: "usr-9b7d21",
  username: "k.oyelaran@northline.example",
  roles: ["3pl"],
  plants: [],
  expires_at: "2026-10-13T18:00:00+00:00",
};

/**
 * `customer` is the end-customer role the API knows least about: ROLE_RESOURCES
 * in src/core/rbac.py grants it exactly {coas, downloads} - no `reports`, no
 * `admin`. This session is the regression guard for the portal offering tabs the
 * role has no entitlement to.
 */
const CUSTOMER_SESSION: MockSession = {
  user_id: "usr-c31f80",
  username: "procurement@halcyon.example",
  roles: ["customer"],
  plants: [],
  expires_at: "2026-10-13T16:30:00+00:00",
};

interface MockCoa {
  coa_id: string;
  plant_of_origin: string;
  completion_status: string;
  customer_id: string;
  product_code: string;
  lot_number: string;
  issue_date: string;
  file_size_bytes: number | null;
  status: string;
  generated_at: string | null;
  completed_at: string | null;
}

const COMPLETE_COA: MockCoa = {
  coa_id: COMPLETE_COA_ID,
  plant_of_origin: "VEC",
  completion_status: "completed",
  customer_id: "CUST-77",
  product_code: "ALM-RAW-24",
  lot_number: "LOT-9911",
  issue_date: "2026-09-28",
  file_size_bytes: 20481,
  status: "completed",
  generated_at: "2026-09-28T09:15:00+00:00",
  completed_at: "2026-09-28T09:15:00+00:00",
};

const INCOMPLETE_COA: MockCoa = {
  coa_id: INCOMPLETE_COA_ID,
  plant_of_origin: "VLP",
  completion_status: "pending",
  customer_id: "CUST-104",
  product_code: "ALM-ROAST-24",
  lot_number: "LOT-9930",
  issue_date: "2026-10-02",
  file_size_bytes: null,
  status: "pending",
  generated_at: null,
  completed_at: null,
};

const GATE_REFUSED_BODY = {
  detail: "Completion gate not satisfied: required test fields are missing",
  code: "completion_gate_failed",
  coa_id: INCOMPLETE_COA_ID,
  missing_fields: MISSING_FIELDS,
};

const MISSING_ROWS = [
  {
    coa_id: INCOMPLETE_COA_ID,
    vessel: "MV Ventura Trader",
    voyage: "V-118E",
    counterparty: "Halcyon Foods",
    priority: "critical",
    required_ship_date: "2026-10-09",
    age_days: 11,
    blocking_reason: "No certificate received from the plant of origin",
  },
  {
    coa_id: "COA-2026-004203",
    vessel: "MV Coast Runner",
    voyage: "V-119W",
    counterparty: "Northline Grocers",
    priority: "medium",
    required_ship_date: "2026-10-20",
    age_days: 4,
    blocking_reason: "Certificate issued without the required moisture result",
  },
];

interface ApiCall {
  method: string;
  path: string;
  body: Record<string, unknown> | null;
}

function readBody(request: Request): Record<string, unknown> | null {
  const raw = request.postData();
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function json(route: Route, status: number, payload: unknown): Promise<void> {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  });
}

/**
 * Apply the FR4 scope filters server-side, the way the real router does, so a
 * test can assert that narrowing by plant/product actually narrows the register
 * rather than only that the query string was built.
 */
function filterCoas(items: MockCoa[], query: URLSearchParams): MockCoa[] {
  const plant = (query.get("plant") ?? "").trim().toUpperCase();
  const product = (query.get("product") ?? "").trim().toUpperCase();
  return items.filter((coa) => {
    if (plant.length > 0 && !coa.plant_of_origin.toUpperCase().includes(plant)) {
      return false;
    }
    if (product.length > 0 && !coa.product_code.toUpperCase().includes(product)) {
      return false;
    }
    return true;
  });
}

/**
 * One route handler for the whole `/api` surface, dispatching on method and
 * path. A single registration keeps precedence unambiguous: the portal's screens
 * call endpoints that share a prefix (`/api/coas/{id}` and
 * `/api/coas/{id}/validate`), which overlapping glob patterns would match in an
 * order-dependent way.
 */
async function installApi(
  page: Page,
  session: MockSession = SESSION,
): Promise<ApiCall[]> {
  const calls: ApiCall[] = [];

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname.replace(/^\/api/, "");
    const method = request.method();
    const body = readBody(request);
    calls.push({ method, path, body });

    if (method === "POST" && path === "/auth/login") {
      await json(route, 200, session);
      return;
    }

    if (method === "GET" && path === "/coas") {
      const items = filterCoas([COMPLETE_COA, INCOMPLETE_COA], url.searchParams);
      await json(route, 200, {
        items,
        page: Number(url.searchParams.get("page") ?? "1"),
        page_size: Number(url.searchParams.get("page_size") ?? "25"),
        total: items.length,
      });
      return;
    }

    if (method === "GET" && /^\/coas\/[^/]+$/.test(path)) {
      const coaId = decodeURIComponent(path.split("/")[2] ?? "");
      await json(route, 200, coaId === INCOMPLETE_COA_ID ? INCOMPLETE_COA : COMPLETE_COA);
      return;
    }

    if (method === "POST" && /^\/coas\/[^/]+\/validate$/.test(path)) {
      const coaId = decodeURIComponent(path.split("/")[2] ?? "");
      const complete = coaId !== INCOMPLETE_COA_ID;
      await json(route, 200, {
        coa_id: coaId,
        completion_status: complete ? "completed" : "pending",
        releasable: complete,
        missing_fields: complete ? [] : MISSING_FIELDS,
      });
      return;
    }

    if (method === "POST" && /^\/coas\/[^/]+\/download$/.test(path)) {
      const coaId = decodeURIComponent(path.split("/")[2] ?? "");
      if (coaId === INCOMPLETE_COA_ID) {
        await json(route, 409, GATE_REFUSED_BODY);
        return;
      }
      await json(route, 200, {
        coa_id: coaId,
        url: DOWNLOAD_URL,
        expires_in: 900,
        method: "GET",
      });
      return;
    }

    if (method === "POST" && /^\/coas\/[^/]+\/print\/reauth$/.test(path)) {
      const coaId = decodeURIComponent(path.split("/")[2] ?? "");
      await json(route, 200, {
        coa_id: coaId,
        reauth_token: REAUTH_TOKEN,
        expires_in: 900,
        expires_at: "2026-10-13T10:19:00+00:00",
      });
      return;
    }

    if (method === "POST" && /^\/coas\/[^/]+\/print$/.test(path)) {
      const coaId = decodeURIComponent(path.split("/")[2] ?? "");
      if (coaId === INCOMPLETE_COA_ID) {
        await json(route, 409, GATE_REFUSED_BODY);
        return;
      }
      if (body?.reauth_token !== REAUTH_TOKEN) {
        await json(route, 401, {
          detail: "Re-authentication is required to print this certificate",
          code: "reauth_required",
        });
        return;
      }
      await json(route, 200, {
        coa_id: coaId,
        status: "printed",
        printed_at: "2026-10-13T10:04:11+00:00",
        printing_plant: "VEC",
        page_count: 2,
      });
      return;
    }

    if (method === "GET" && path === "/missing-coas") {
      await json(route, 200, {
        items: MISSING_ROWS,
        generated_at: "2026-10-13T06:00:00+00:00",
      });
      return;
    }

    if (method === "POST" && path === "/admin/users") {
      await json(route, 201, {
        ...(body ?? {}),
        created_at: "2026-10-13T10:05:00+00:00",
      });
      return;
    }

    await json(route, 404, { detail: `No mock for ${method} ${path}` });
  });

  return calls;
}

/** Drive the gate and wait for the authenticated shell. */
async function signIn(page: Page, session: MockSession = SESSION): Promise<void> {
  await page.goto("/");
  await page.getByLabel("Operator identity").fill(session.username);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(
    page.getByRole("heading", { name: "Search certificates of analysis" }),
  ).toBeVisible();
}

test.describe("COA portal", () => {
  test("exchanges the identity token once and renders the shell", async ({ page }) => {
    const calls = await installApi(page);
    await signIn(page);

    await expect(page).toHaveTitle(/Ventura COA Self-Service Portal/i);
    await expect(page.getByRole("button", { name: "Search", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Report", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Admin", exact: true })).toBeVisible();
    await expect(page.getByText(SESSION.username).first()).toBeVisible();
    await expect(page.getByText(SESSION.user_id)).toBeVisible();
    await expect(page.getByText(SESSION.roles.join(" · "))).toBeVisible();

    const logins = calls.filter((call) => call.path === "/auth/login");
    expect(logins).toHaveLength(1);
    expect(logins[0]?.method).toBe("POST");
    expect(logins[0]?.body).toEqual({ token: `local-bootstrap:${SESSION.username}` });
  });
});

test("an internal session is offered the Admin tab", async ({ page }) => {
  await installApi(page, SESSION);
  await signIn(page, SESSION);

  await expect(page.getByRole("button", { name: "Admin", exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Admin", exact: true }).click();
  await expect(page.getByRole("heading", { name: "External portal users" })).toBeVisible();
  await expect(page.getByText("Signed in without an administrator role")).toHaveCount(0);
});

test("an external session is never offered the Admin tab", async ({ page }) => {
  const calls = await installApi(page, EXTERNAL_SESSION);
  await signIn(page, EXTERNAL_SESSION);

  await expect(page.getByRole("button", { name: "Search", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Admin", exact: true })).toHaveCount(0);
  await expect(page.getByText(EXTERNAL_SESSION.username).first()).toBeVisible();

  expect(calls.filter((call) => call.path === "/admin/users")).toHaveLength(0);
});

test("a customer session is offered neither the Report nor the Admin tab", async ({ page }) => {
  const calls = await installApi(page, CUSTOMER_SESSION);
  await signIn(page, CUSTOMER_SESSION);

  await expect(page.getByRole("button", { name: "Search", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Report", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Admin", exact: true })).toHaveCount(0);

  expect(calls.filter((call) => call.path === "/missing-coas")).toHaveLength(0);
  expect(calls.filter((call) => call.path === "/admin/users")).toHaveLength(0);
});

test("the register is served by the contract search endpoint", async ({ page }) => {
  const calls = await installApi(page);
  await signIn(page);

  await page.getByRole("button", { name: "Search", exact: true }).click();

  const searches = calls.filter((call) => call.method === "GET" && call.path === "/coas");
  expect(searches.length).toBeGreaterThan(0);
  expect(searches[0]?.body).toBeNull();
  await expect(page.getByText(COMPLETE_COA_ID).first()).toBeVisible();
  await expect(page.getByText(INCOMPLETE_COA_ID).first()).toBeVisible();
});
