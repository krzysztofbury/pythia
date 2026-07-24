# QNAP Deployment

This deployment pulls `linux/amd64` images from GHCR. It has no `build` key and
does not require a repository checkout on the NAS.

## Prerequisites

- QNAP Container Station with Docker Compose v2 and an amd64 CPU.
- Access to `ghcr.io/krzysztofbury/pythia`.
- A Kraken key without withdrawal permission.
- The repository's GitHub `production` environment configured with a required
  reviewer before stable promotion.

If the GHCR package is private, authenticate the QNAP Docker daemon before
pulling. Use a token with package read access only. `docker login` writes the
credential to the Docker config file; do not put the token in an env file:

```sh
docker login ghcr.io -u YOUR_GITHUB_USER
```

Set `WATCHTOWER_DOCKER_CONFIG_FILE` in `config.env` to the absolute path of
that Docker `config.json` before enabling Watchtower for a private package.

## Configure

Place this directory on the NAS, then create local configuration files:

```sh
cp config.env.example config.env
cp candidate.env.example candidate.env
cp live.env.example live.env
chmod 600 config.env candidate.env live.env
```

Replace every placeholder in the channel env files. Do not put secrets in
`config.env`; that file controls image channels and Watchtower selection.

Create the legacy archive directory before Compose evaluates its bind mount:

```sh
mkdir -p archive/v9
```

If a v9 `trades.db` is retained, place a copy under this directory while the
v9 process is stopped, then remove write permission from the archived copy.

The archive is mounted at `/archive/v9:ro`. No v10 state setting points there,
and v10 must not open it. The source tree ships an offline cutover utility
(`src/rebuild_v10.py`, run with `--confirm-services-stopped` while every bot
process is stopped) that archives v9 and rebuilds v10 state from exact Kraken
records, including seeding the LIVE fill cursor. It is not yet a supported
one-command migration — treat any cutover as an operator-attended, reviewed
procedure, and retain the archive indefinitely either way.

Validate without starting containers:

```sh
docker compose --env-file config.env --profile candidate --profile live config --quiet
```

After pulling each channel image and before starting the bot, preflight the
named state volume as the runtime UID. This verifies create, SQLite-style
rename, and delete permissions without starting WARP or the bot entrypoint:

```sh
docker compose --env-file config.env --profile candidate pull oracle-candidate
docker compose --env-file config.env --profile candidate run --rm --no-deps \
  --entrypoint /bin/sh oracle-candidate -ec \
  'test "$(id -u)" = 10001; f=/app/logs/.write-test.$$; printf x > "$f"; mv "$f" "$f.renamed"; rm "$f.renamed"'

docker compose --env-file config.env --profile live pull oracle-live
docker compose --env-file config.env --profile live run --rm --no-deps \
  --entrypoint /bin/sh oracle-live -ec \
  'test "$(id -u)" = 10001; f=/app/logs/.write-test.$$; printf x > "$f"; mv "$f" "$f.renamed"; rm "$f.renamed"'
```

If either check fails, correct the named-volume ownership for UID/GID 10001
before starting that channel. Do not work around it by running the bot as root.

## Run Candidate

Candidate is forced to PAPER by Compose even if `candidate.env` is edited
incorrectly:

```sh
docker compose --env-file config.env --profile candidate pull
docker compose --env-file config.env --profile candidate up -d
docker compose --env-file config.env logs -f oracle-candidate
```

Confirm the boot log says `mode=PAPER`. Observe representative cycles, API
failure behavior, health, and persistent execution/analytics writes. A quiet
period is not sufficient validation.

## Promote Stable

1. Open the successful `Publish candidate` workflow run on `main`.
2. Copy its full 40-character revision and complete `sha256:...` digest.
3. Dispatch `Promote stable` with both values and `PROMOTE` confirmation.
4. Approve the `production` environment deployment.
5. Confirm the workflow's remote stable digest matches the requested digest.

Promotion retags the registry manifest. It does not rebuild the image.

Start the stable profile in PAPER first:

```sh
docker compose --env-file config.env --profile live pull
docker compose --env-file config.env --profile live up -d
docker compose --env-file config.env logs -f oracle-live
```

When the stable image has passed PAPER validation, arm all four LIVE controls
in `live.env` deliberately:

```dotenv
TRADING_MODE=LIVE
TRADING_ACCOUNT_ID=your-expected-account-identifier
LIVE_TRADING_ACK=I_UNDERSTAND_LIVE_TRADING
LIVE_ENTRIES_ENABLED=true
```

Review the intended four arming values in the protected env file, then recreate
the live service and confirm the boot log reports `mode=LIVE`:

```sh
docker compose --env-file config.env --profile live up -d --force-recreate oracle-live
```

## Select Stable Watchtower Updates

Watchtower is behind the separate `watchtower` profile and is not started by
the `candidate` or `live` profile. It cannot update candidate, WARP, or itself.
It requires both the `enable=true` label and the `oracle-pythia-stable` scope.
Stable watching is also off by default.

To opt in after choosing the stable channel, set this in `config.env`:

```dotenv
WATCHTOWER_STABLE_SELECTED=true
```

Then recreate the selected service so the interpolated label is applied:

```sh
docker compose --env-file config.env --profile live --profile watchtower \
  up -d --force-recreate oracle-live watchtower
```

`WATCHTOWER_CLEANUP=false` is fixed in Compose, so previous images remain
available for rollback. The Docker socket grants Watchtower daemon-equivalent
control despite the read-only mount; treat Watchtower as privileged
infrastructure and protect access to its configuration.

Watchtower does not wait for Oracle Pythia's structured health status and does
not roll back an unhealthy replacement. `restart: unless-stopped` restarts a
process that exits; Docker does not restart a process merely because its
health status becomes `unhealthy`. Keep automated updates off unless an
operator can monitor and perform a manual digest rollback.

## Roll Back

Set `WATCHTOWER_STABLE_SELECTED=false`, then set `STABLE_REF` in `config.env` to
the previously recorded digest, including the leading `@`:

```dotenv
STABLE_REF=@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

`WATCHTOWER_CLEANUP=false` retains prior images. If the digest is still local,
recreate without pulling; otherwise authenticate to GHCR and pull that exact
digest before recreating only the live service:

```sh
docker compose --env-file config.env --profile live up -d --pull never \
  --force-recreate oracle-live
# If the retained image is unavailable locally:
docker compose --env-file config.env --profile live pull oracle-live
docker compose --env-file config.env --profile live up -d --force-recreate oracle-live
```

The named `live-logs` volume is preserved. Verify schema compatibility and the
effective PAPER/LIVE mode before relying on the rollback.

## Isolation and Operations

- No service publishes a host port.
- WARP and bot containers share only the project-scoped `proxy-egress` bridge.
- WARP is the read-only-root exception because its upstream entrypoint writes
  `/usr/bin/rws-cli-v6`; it still has all capabilities dropped, no new
  privileges, and ephemeral `/etc/sing-box` configuration.
- Watchtower uses a separate management bridge.
- Bot roots are read-only; `/tmp` is ephemeral, `/app/logs` is the only
  persistent writable runtime path, and `/archive/v9` is a separate read-only
  legacy mount.
- Container stdout uses explicit size and file-count rotation.
- The bot receives 90 seconds to stop cleanly.

Useful checks:

```sh
docker compose --env-file config.env ps
docker compose --env-file config.env logs --since 30m oracle-live
docker compose --env-file config.env exec oracle-live /app/.venv/bin/python --version
docker compose --env-file config.env exec oracle-live \
  /app/.venv/bin/python /app/src/health.py /app/logs/healthcheck \
  --max-age-seconds 4500
```
