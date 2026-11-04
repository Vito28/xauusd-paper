---

Issue 1

Title: Set up monorepo structure: React (Vite) frontend & NestJS backend

┌───────────────────┬───────────────┬──────────┬──────┐
│       Area        │   Work Type   │ Priority │ Size │
├───────────────────┼───────────────┼──────────┼──────┤
│ Frontend, Backend │ Setup / Chore │ High     │ S    │
└───────────────────┴───────────────┴──────────┴──────┘

## Description
Set up the initial project structure as a monorepo with two separate applications: a **frontend** built with React (Vite + TypeScript) and a **backend** built with NestJS.

## Tasks
- [ ] Create root folders `frontend/` and `backend/`
- [ ] Initialize `frontend/` with Vite (`react-ts` template) + Tailwind CSS
- [ ] Initialize `backend/` with NestJS CLI (TypeScript, strict mode)
- [ ] Add root `.gitignore` (node_modules, dist, .env*, coverage)
- [ ] Add `.env.example` for each app (no real secrets committed)
- [ ] Configure ESLint + Prettier consistently across both apps
- [ ] Add a root `README.md` explaining the folder layout and how to run each app locally

## Target Structure
```
.
├── frontend/          # React + Vite
│   └── src/
├── backend/           # NestJS
│   └── src/
├── docs/              # thesis documents (existing .docx files can move here)
├── .gitignore
└── README.md
```

## Acceptance Criteria
- `cd frontend && npm run dev` serves the default page on `http://localhost:5173`
- `cd backend && npm run start:dev` serves `GET /` on `http://localhost:3000`
- Both apps pass `npm run lint` and `npm run build`

---

Issue 2

Title: Dockerize stack with Docker Compose (PostgreSQL, React, NestJS)

┌────────┬───────────────┬──────────┬──────┐
│  Area  │   Work Type   │ Priority │ Size │
├────────┼───────────────┼──────────┼──────┤
│ DevOps │ Setup / Chore │ High     │ M    │
└────────┴───────────────┴──────────┴──────┘

## Description
Containerize the application so the whole stack (React frontend, NestJS backend, PostgreSQL database) command.

**Depends on:** #1

## Tasks
- [ ] `backend/Dockerfile` — multi-stage build (deps → build → slim runtime), runs `node dist/main`
- [ ] `frontend/Dockerfile` — multi-stage build: build with Node, serve static `dist/` with Nginx
- [ ] `frontend/nginx.conf` — SPA fallback (`try_files $uri /index.html`) and `/api` reverse proxy to backend
- [ ] `.dockerignore` for each app (node_modules, dist, .env)
- [ ] Root `docker-compose.yml` with services:
  - `db` — PostgreSQL 16 (named volume for data, healthcheck with `pg_isready`)
  - `backend` — NestJS, `depends_on: db (service_healthy)`, reads `DATABASE_URL` from env
  - `frontend` — Nginx serving the React build
- [ ] Root `.env.example` with `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `DATABASE_URL`, `VITE_API_URL`
- [ ] Optional `docker-compose.dev.yml` with bind mounts for hot reload
- [ ] Document `docker compose up --build` in README

## Acceptance Criteria
- `docker compose up --build` starts all three services without errors
- Frontend reachable at `http://localhost:8080`, backend at `http://localhost:3000`
- Backend can connect to PostgreSQL (simple `GET /health` endpoint returns DB status)
- Database data persists after `docker compose down` / `up`

---

Issue 3

Title: Integrate shadcn-admin components with feature-based architecture

┌────────────────────┬───────────┬──────────┬──────┐
│        Area        │ Work Type │ Priority │ Size │
├────────────────────┼───────────┼──────────┼──────┤
│ Frontend (Website) │ Feature   │ Medium   │ L    │
└────────────────────┴───────────┴──────────┴──────┘

## Description
Use [satnaing/shadcn-admin](https://github.com/satnaing/shadcn-admin) (React + Vite + shadcn/ui + TanStd bring only its reusable components into our `frontend/` with a clean, feature-based folderarchitecture.

**Depends on:** #1

## Tasks
- [ ] Clone shadcn-admin locally for reference (do **not** commit the whole template into this repo)
- [ ] Initialize shadcn/ui in `frontend/` (`npx shadcn@latest init`) and add base primitives via the CL
- [ ] Set up routing with TanStack Router (same as the template)
- [ ] Port only what we need:
  - [ ] App layout: sidebar, header, breadcrumb, profile menu
  - [ ] Theme provider (light/dark) and theme switch
  - [ ] Data table (TanStack Table: pagination, sorting, filtering, column toggle)
  - [ ] Form patterns (React Hook Form + Zod)
  - [ ] Auth layout (sign-in page) — UI only
  - [ ] Error pages (404 / 500)
- [ ] Remove demo-only content (apps, chats, tasks, etc.) that is not relevant to GOR
- [ ] Replace sidebar menu with GOR module placeholders: Dashboard, Courts, Bookings, Schedule, Reports
- [ ] Keep the MIT license notice / credit for ported code

## Proposed Folder Architecture (feature-based)
```
frontend/src/
├── routes/                    # TanStack Router route files
│   ├── __root.tsx
│   ├── (auth)/sign-in.tsx
│   └── _authenticated/
│       ├── index.tsx          # dashboard
│       ├── courts/
│       └── bookings/
├── components/
│   ├── ui/                    # shadcn primitives (generated by CLI)
│   ├── layout/                # sidebar, header, nav
│   └── shared/                # data-table, page-header, etc.
├── features/                  # domain modules (components, hooks, schemas, api per feature)
│   ├── dashboard/
│   ├── courts/
│   └── bookings/
├── hooks/
├── lib/                       # utils, api client, constants
├── context/                   # theme, providers
├── types/
└── main.tsx
```

## References
- https://github.com/satnaing/shadcn-admin — main UI base
- https://github.com/Kiranism/next-shadcn-dashboard-starter — reference for feature-based folder structure
- https://ui.shadcn.com/blocks — official sidebar/dashboard blocks

## Acceptance Criteria
- Dashboard shell (sidebar + header + theme toggle) renders after sign-in route
- A sample data table page works with dummy data
- No unused demo pages/features remain from the template
- `npm run build` and `npm run lint` pass

---