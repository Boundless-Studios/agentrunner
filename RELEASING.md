# Releasing

This repo publishes to [PyPI](https://pypi.org/project/boundless-agentrunner/)
via **GitHub Actions + PyPI Trusted Publishing (OIDC)** — no API tokens or stored
secrets. A push of a `v*` tag builds the package and publishes it.

This document is the template Boundless Studios uses for **every** OSS Python
repo. To set up a brand-new repo, follow [One-time setup](#one-time-setup); to
ship a release, follow [Cutting a release](#cutting-a-release).

## Distribution name vs import name

The PyPI **distribution name** is `boundless-agentrunner` — the plain
`agentrunner` name was already taken on PyPI. The **import name is unchanged**:

```bash
pip install boundless-agentrunner
```
```python
import agentrunner
```

This split is intentional and supported: `pyproject.toml` sets
`name = "boundless-agentrunner"` (what `pip install` uses) while
`[tool.hatch.build.targets.wheel] packages = ["src/agentrunner"]` keeps the
import package `agentrunner`. **When a name is taken, namespace the distribution,
not the import.** For a new Boundless repo, prefer the bare name if available and
fall back to a `boundless-` prefix.

## How it works

`.github/workflows/publish.yml` triggers on any `v*` tag push. It runs in the
`pypi` GitHub Environment, requests an OIDC token (`id-token: write`), builds the
sdist + wheel with `python -m build`, and uploads with
[`pypa/gh-action-pypi-publish`](https://github.com/pypa/gh-action-pypi-publish).
PyPI verifies the OIDC token against a **trusted publisher** registered for the
project — so the action authenticates as the repo, not as a person, and there is
no long-lived token to leak.

## One-time setup

Do this **once per package**. It is the only manual, web-UI step — PyPI has no
API for configuring trusted publishers.

### 1. Register the trusted publisher on PyPI

If the project **does not exist on PyPI yet** (first release), register a
**pending publisher** — it creates the project on first successful publish:

1. Sign in to <https://pypi.org> with the Boundless Studios publishing account.
2. Go to **Account settings → Publishing** (<https://pypi.org/manage/account/publishing/>).
3. Under **Add a new pending publisher**, choose **GitHub** and fill in:

   | Field | Value |
   |-------|-------|
   | PyPI Project Name | `boundless-agentrunner` *(the **distribution** name, which may differ from both the repo and the import name)* |
   | Owner | `Boundless-Studios` |
   | Repository name | `agentrunner` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

4. **Add**.

If the project **already exists on PyPI**, register the publisher from the
project page instead: **Manage → Settings → Publishing → Add a new publisher**
(same five fields, minus the project name).

> **The publisher binds to the distribution name.** PyPI authorizes the upload
> against the project named in `pyproject.toml`'s `name`, not the repo name. If
> the build produces `agentrunner` but the publisher is registered for
> `boundless-agentrunner` (or vice versa), the upload fails. Keep
> `pyproject` `name` and the registered PyPI Project Name identical.
>
> The four GitHub-side fields must exactly match the OIDC claims the workflow
> emits. If a publish fails with `invalid-publisher: valid token, but no
> corresponding publisher`, the failed run's logs print the exact claims it sent
> (`repository_owner`, `repository`, `workflow_ref`, `environment`) — compare
> them against the publisher you registered.

### 2. Confirm the GitHub Environment exists

The workflow pins `environment: pypi`. Create it once at
**Settings → Environments → New environment → `pypi`** (no secrets or protection
rules required; add reviewers here later if you want a manual approval gate
before each publish).

### 3. Confirm `publish.yml` is present

Copy [`.github/workflows/publish.yml`](.github/workflows/publish.yml) verbatim
into the new repo. It needs no edits beyond matching the `environment` name to
the one you registered above.

## Cutting a release

1. **Bump the version.** Edit `version` in `pyproject.toml` (e.g. `0.2.0` →
   `0.3.0`). Follow [SemVer](https://semver.org/). Open a PR, get it merged to
   `main`.
2. **Tag the merged commit.** The tag **must** match the `pyproject.toml`
   version, prefixed with `v`:

   ```bash
   git checkout main && git pull
   git tag v0.3.0          # must equal pyproject version
   git push origin v0.3.0
   ```

   > **The tag determines what gets built.** `publish.yml` checks out the tag's
   > tree, so it publishes whatever `version` is in `pyproject.toml` *at that
   > tag* — not what's on `main`. Tagging a commit whose `pyproject` version
   > doesn't match the tag name ships a mislabeled release. Always tag the
   > commit you bumped.

3. **CI publishes.** The tag push starts the **Publish to PyPI** workflow. Watch
   it:

   ```bash
   gh run watch --repo Boundless-Studios/agentrunner
   ```

4. **Verify.** Once green, the release is live within a minute or two:

   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/boundless-agentrunner/json   # 200
   python -m venv /tmp/ar && /tmp/ar/bin/pip install boundless-agentrunner==0.3.0
   /tmp/ar/bin/python -c "import agentrunner; print('ok')"
   ```

## Re-running a failed publish

A publish can fail before the trusted publisher is configured (the very first
release) without anything being wrong with the build. After fixing the
publisher config, **re-run the same workflow run** — the OIDC token is minted at
job runtime, so a re-run picks up the new publisher without re-tagging:

```bash
gh run rerun <run-id> --repo Boundless-Studios/agentrunner
```

PyPI rejects re-uploading a version that already published, so if the build step
succeeded but only the upload failed, a re-run is safe and idempotent. A re-run
only helps when the *built artifact* is unchanged — if you renamed the
distribution or changed the code, bump to a new version and tag again, because a
published version is immutable.

## Reuse checklist (new Boundless OSS repo)

- [ ] `pyproject.toml` has a correct `name` (bare name if free on PyPI, else
      `boundless-<name>`), `version`, `license`, and build backend
- [ ] Import package under `src/<import_name>/` with
      `[tool.hatch.build.targets.wheel] packages = ["src/<import_name>"]`
- [ ] `.github/workflows/publish.yml` copied in (tag-triggered, `environment: pypi`,
      `id-token: write`)
- [ ] `pypi` GitHub Environment created
- [ ] PyPI pending trusted publisher registered for the **distribution name**
      (owner / repo / `publish.yml` / `pypi`) — see
      [step 1](#1-register-the-trusted-publisher-on-pypi)
- [ ] `README.md` shows `pip install <dist-name>` and, if it differs, the import name
- [ ] First version tagged; workflow green; `pip install` verified
- [ ] This `RELEASING.md` copied in and the names updated
