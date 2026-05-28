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
