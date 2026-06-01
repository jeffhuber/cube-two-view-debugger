# Railway Deploy Guardrail

Do not run `railway up` for `ctvd-recognizer` from the primary working checkout.
Do not run `railway down` to clean up a bad deployment; it can remove the last good deployment. Prefer a clean redeploy through this script, then verify the latest deployment metadata.

Use:

```bash
tools/deploy_railway_ctvd_recognizer.sh
```

The script deploys `origin/main` by default from a fresh temporary git worktree, so local untracked directories such as `.worktrees/`, caches, and in-progress edits cannot change the upload archive or hide `railway.json`.

To deploy a specific committed ref:

```bash
tools/deploy_railway_ctvd_recognizer.sh <ref> "Deploy message"
```

After the deploy starts, verify the latest Railway deployment has:

- `configFile: /railway.json`
- `builder: NIXPACKS`
- `startCommand: python railway_start.py`
- `healthcheckPath: /api/diag`

Then smoke-test:

```bash
curl -fsS https://ctvd-recognizer-production.up.railway.app/api/diag
```

## CubeSnap iOS Debug Uploads

The optional iOS repro-bundle upload endpoint is disabled unless
`CUBE_IOS_REPRO_UPLOAD_TOKEN` is set on the `ctvd-recognizer` service. The
same token value is baked into internal CubeSnap iOS builds through Xcode
Cloud's `CUBESNAP_DEBUG_UPLOAD_TOKEN` secret; do not print it in logs or PRs.

Railway setup:

```bash
railway variable set CUBE_IOS_REPRO_UPLOAD_TOKEN --stdin --service ctvd-recognizer --environment production
railway variable set CUBE_IOS_REPRO_UPLOAD_RETENTION_DAYS=14 --service ctvd-recognizer --environment production --skip-deploys --json
tools/deploy_railway_ctvd_recognizer.sh
```

After deploy, run the tokened upload smoke from a trusted shell:

```bash
CUBE_IOS_REPRO_UPLOAD_TOKEN="$(railway variable list --service ctvd-recognizer --environment production --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["CUBE_IOS_REPRO_UPLOAD_TOKEN"])')" \
  .venv/bin/python tools/smoke_ios_repro_upload.py
```

The endpoint stores decoded bundles under `/runs/ios-repro-uploads/<id>/` by
default and prunes entries older than `CUBE_IOS_REPRO_UPLOAD_RETENTION_DAYS`
days. Set the value to `0` to disable pruning during a short investigation.

## Durable Recognition Event Logging

Production recognition metadata should be written to a Railway volume, not the
container filesystem. The app writes metadata-only SQLite events when
`CUBE_RECOGNITION_EVENT_DB_PATH` points at a database path. The event records
include status, recognition category, failed checks, constrained-recognizer
decision fields, stage timings, image hashes/sizes/dimensions, and optional
CubeSnap client metadata. They do **not** include image bytes.

Railway setup:

```bash
railway volume add --service ctvd-recognizer --environment production --mount-path /data --json
railway variable set CUBE_RECOGNITION_EVENT_DB_PATH=/data/recognition_events.sqlite3 --service ctvd-recognizer --environment production --skip-deploys --json
tools/deploy_railway_ctvd_recognizer.sh
```

Verify with:

```bash
curl -fsS https://api.cubesnap.app/api/diag
```

The response includes `recognitionEvents.totalEvents`, status/category counts,
and the latest event timestamp. For a production-safe operational report over
HTTPS, use the metadata-only API:

```bash
curl -fsS 'https://api.cubesnap.app/api/recognition-events/report?sinceHours=24&recentLimit=20'
```

For a daily/adhoc production report without shell access, render the endpoint
response through the report tool:

```bash
.venv/bin/python tools/report_recognition_events.py \
  --endpoint https://api.cubesnap.app/api/recognition-events/report \
  --since-hours 24 \
  --recent-limit 20 \
  --report runs/recognition_event_report.md
```

For a structured report from a copied or local database, run:

```bash
.venv/bin/python tools/report_recognition_events.py \
  --db /data/recognition_events.sqlite3 \
  --since-hours 24 \
  --report runs/recognition_event_report.md
```

For deeper ad hoc queries, connect to the service shell and inspect
`/data/recognition_events.sqlite3` with SQLite.
