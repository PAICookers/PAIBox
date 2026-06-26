# PAIBox Visualizer UI

This directory contains the React/Vite frontend source for the packaged
PAIBox visualizer.

The installed visualizer does not run from this directory. Packaging copies the
compiled static assets into `paibox/visualizer/static/`, and the Python
FastAPI service serves that directory from the wheel.

## Development

Install local UI dependencies once:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`. Start a backend
service separately when using the dev server:

```bash
uv run paiviz --artifact /path/to/config.pb --host 127.0.0.1 --port 8000 --no-browser
```

## Code Style

Use the local scripts before packaging UI changes:

- `npm run format`: apply Prettier to TS/TSX/CSS/config files.
- `npm run format:check`: verify Prettier formatting.
- `npm run lint`: run ESLint over TypeScript, React TSX, Vite config, and Node
  helper scripts.
- `npm run lint:css`: run Stylelint over CSS.
- `npm run check`: run formatting checks, TS/React lint, CSS lint, and the
  production build.

Keep frontend types aligned with the Python viewer JSON schema. The frontend
consumes `/api/...`; it must not parse protobuf files or raw frame files
directly.

## Package Build

Build the frontend and refresh packaged static assets:

```bash
npm run build:package
```

This writes Vite output to `paibox/visualizer/static/`. Commit both the source
changes under `tools/visualizer_ui/` and the refreshed static files under
`paibox/visualizer/static/` when changing UI behavior.

## Installed User Path

End users do not need Node.js. After installing the optional visualizer extra,
they can start the packaged UI directly:

```bash
pip install "paibox[visualizer]"
paiviz --artifact /path/to/config.pb
```

The Python service loads the artifact, exposes `/api/...`, and serves the
prebuilt static UI bundled in the Python package.
