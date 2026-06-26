# Releasing

This repo publishes to [PyPI](https://pypi.org/project/agentrunner/) via **GitHub
Actions + PyPI Trusted Publishing (OIDC)** — no API tokens or stored secrets. A
push of a `v*` tag builds the package and publishes it.

This document is the template Boundless Studios uses for **every** OSS Python
repo. To set up a brand-new repo, follow [One-time setup](#one-time-setup); to
ship a release, follow [Cutting a release](#cutting-a-release).

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
   | PyPI Project Name | `agentrunner` *(the package name, not the repo, if they differ)* |
   | Owner | `Boundless-Studios` |
   | Repository name | `agentrunner` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

4. **Add**.

If the project **already exists on PyPI**, register the publisher from the
project page instead: **Manage → Settings → Publishing → Add a new publisher**
(same five fields, minus the project name).

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

1. **Bump the version.** Edit `version` in `pyproject.toml` (e.g. `0.1.0` →
   `0.2.0`). Follow [SemVer](https://semver.org/). Open a PR, get it merged to
   `main`.
2. **Tag the merged commit.** The tag **must** match the `pyproject.toml`
   version, prefixed with `v`:

   ```bash
   git checkout main && git pull
   git tag v0.2.0          # must equal pyproject version
   git push origin v0.2.0
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
   curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/agentrunner/json   # 200
   python -m venv /tmp/ar && /tmp/ar/bin/pip install agentrunner==0.2.0
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
succeeded but only the upload failed, a re-run is safe and idempotent. If you
need to change the *built artifact*, bump to a new version and tag again — a
published version is immutable.

## Reuse checklist (new Boundless OSS repo)

- [ ] `pyproject.toml` has a correct `name`, `version`, `license`, and build
      backend
- [ ] `.github/workflows/publish.yml` copied in (tag-triggered, `environment: pypi`,
      `id-token: write`)
- [ ] `pypi` GitHub Environment created
- [ ] PyPI pending trusted publisher registered (owner / repo / `publish.yml` /
      `pypi`) — see [step 1](#1-register-the-trusted-publisher-on-pypi)
- [ ] `README.md` shows `pip install <package>`
- [ ] First tag `v0.1.0` pushed; workflow green; `pip install` verified
- [ ] This `RELEASING.md` copied in and the package name updated
