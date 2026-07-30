# Oracle Pythia Repository and Release Specification

## Scope

This v10.0.0 specification defines runtime data authority, repository quality
gates, container production, release promotion, and QNAP runtime invariants.
It does not make PAPER execution equivalent to LIVE validation.

## Safety Invariants

1. Python is 3.13 in CI and in the runtime image.
2. CI and image smoke tests never set `TRADING_MODE=LIVE` and run the image
   smoke test with Docker networking disabled.
3. The local Compose stack overrides any env-file value with
   `TRADING_MODE=PAPER`.
4. The QNAP candidate service overrides any env-file value with
   `TRADING_MODE=PAPER` and `LIVE_ENTRIES_ENABLED=false`.
5. The image executes the existing flat-import entrypoint as
   `/app/.venv/bin/python /app/src/main.py`; no package namespace migration is
   implied.
6. The runtime image contains the virtual environment and `src/` only from the
   repository. Source trees, tests, local env files, and build tools are not
   copied into the final stage.
7. Runtime bot containers use UID/GID 10001, a read-only root filesystem, all
   Linux capabilities dropped, `no-new-privileges`, a writable `/tmp` tmpfs,
   and a dedicated persistent `/app/logs` state volume.
8. Neither local nor QNAP Compose publishes the WARP proxy port to the host.
   WARP is reachable only on the project-scoped bridge network.
9. QNAP pulls registry images only. It never builds application source.
10. No workflow or deployment emits or creates a `latest` tag.
11. DCA and partial exits remain disabled on the v10 coordinator path until
    protection quantities can be resized exactly and recovered safely.
12. LLM features default off. Enabled output remains shadow-only in LIVE unless
    influence is separately armed, and all paid calls use persistent caps.

## Runtime Data Authority

- `paper-execution-v10.db` and `live-execution-v10.db` are separate,
  scope-bound execution stores. The selected store is authoritative for orders,
  fills, positions, and protections.
- `analytics-v10.db` is a derived journal for signals, snapshots, reporting,
  and optimization. It is never execution authority.
- v9 `trades.db` is retained only as a read-only archive. It is mounted at
  `/archive/v9:ro` on QNAP, outside the writable state volume, and no v10 path
  setting points to it.
- No supported v9 import or analytics rebuild command exists. Such a command is
  future work and must not be inferred from the migration utilities present in
  the source tree.
- The structured health document defaults to `/app/logs/healthcheck` in the
  image and shares the persistent state volume with the execution databases.

## Quality Gate

`.github/workflows/_quality.yml` is the single reusable quality definition.
It performs the following mandatory checks:

- `uv sync --frozen` and the complete pytest suite.
- Ruff syntax and critical error families that pass on the current source.
- A Ruff source-only ratchet: the current baseline is 167 findings and a
  change may not increase that count. Reducing the baseline is encouraged;
  after cleanup, lower `RUFF_MAX_FINDINGS` in the workflow in the same change.
- Validation of both Compose definitions.
- A `linux/amd64` Docker build and network-disabled import smoke test of the
  current `src/main.py` file.
- A UID 10001 create/write/rename SQLite smoke and a synthetic READY document
  checked by the exact offline container health command.

The smoke test intentionally imports the runtime graph without calling
`main()`. PAPER mode still uses public exchange and model APIs, so starting the
loop would not be an offline smoke test.

## Image and Tags

The canonical image is `ghcr.io/krzysztofbury/pythia`.

- `candidate` is the moving pre-production tag.
- `sha-<40-character-git-sha>` is a one-time traceability tag produced in the
  same build as `candidate`; publication fails if that tag already exists.
- `stable` is the moving production tag.
- Image platform is exactly `linux/amd64`.
- OCI source, revision, version, and creation labels are populated at build
  time.

`publish-candidate.yml` runs only for `refs/heads/main`, builds once, pushes the
candidate and one-time SHA tags from that build, records its registry digest,
and publishes build provenance for that digest. The digest is the promotion and
rollback authority.

`promote-stable.yml` runs only for `refs/heads/main` and accepts a full tested
Git SHA plus `sha256:<64 hex characters>` digest. It verifies the digest against
that one-time SHA tag, does not check out source or invoke a build, copies the
exact manifest to `stable`, and verifies the remote stable digest.

## Production Approval

Stable promotion is manual (`workflow_dispatch`) and the promotion job targets
the GitHub `production` environment. Repository administrators must create
that environment, restrict deployment branches to `main`, and configure at
least one required reviewer. Without environment protection rules, GitHub
cannot provide the intended human approval gate.

Candidate publication and stable promotion have separate non-cancelling
concurrency groups. This prevents simultaneous tag writers while avoiding an
in-progress publication being cancelled after registry mutation.

## Permissions

Workflow permissions default to read-only. Package write and OIDC token write
are granted only to candidate publication. Package write is granted only to
stable promotion. Security-event write is granted only to CodeQL and SARIF
upload jobs.

All directly referenced GitHub Actions are pinned to full commit SHAs. The
adjacent comments retain the audited release tag for update visibility.

## QNAP Channels

`deploy/qnap/compose.yaml` targets `linux/amd64` and exposes three explicit
profiles:

- `candidate`: uses the `candidate` tag and is unconditionally PAPER.
- `live`: uses the `stable` tag; `TRADING_MODE` comes from `live.env` and must
  be changed to LIVE deliberately by the operator.
- `watchtower`: starts Watchtower only after a separate explicit opt-in.

Before either bot starts, the operator must verify create/write/rename access to
its named `/app/logs` volume as UID/GID 10001. The same read-only v9 archive is
mounted separately at `/archive/v9`; the runtime is never directed to open it.

Watchtower is label-enabled and scoped to `oracle-pythia-stable`. Candidate,
WARP, and Watchtower itself are explicitly excluded. Stable updates are also
disabled by default; the operator must set
`WATCHTOWER_STABLE_SELECTED=true`, enable the `watchtower` profile, and recreate
the live service to opt in. Private GHCR access requires a protected Docker
`config.json` mounted read-only at `/config.json`.
`WATCHTOWER_CLEANUP=false` retains old images for rollback.

Watchtower's Docker socket access is inherently daemon-equivalent privilege.
It is isolated on a separate network, given no additional capabilities, and
limited by both enable and scope labels, but the socket remains a trusted
administrative boundary.

Watchtower does not gate replacement on the application healthcheck and does
not automatically roll back an unhealthy image. Docker restart policy reacts
to process exit, not health status alone.

The upstream WARP v6 entrypoint generates `/etc/sing-box/config.json` and
`/usr/bin/rws-cli-v6` at every start, so WARP is the documented read-only-root
exception. Its generated configuration is on tmpfs, it receives no Linux
capabilities or host ports, and `no-new-privileges` remains enabled.

## Rollback

Before promotion, record the current stable digest. To roll back QNAP, set
`STABLE_REF=@sha256:<digest>`, set `WATCHTOWER_STABLE_SELECTED=false`, and
recreate only `oracle-live`. Use the retained local image with `--pull never`
when available, or authenticate and pull that exact digest. Persistent logs and
SQLite data remain in the named live volume.

## Known Follow-ups

- Third-party container base images use version tags rather than repository
  digests so Dependabot can propose updates. Pin them to reviewed multi-arch
  digests if the operational update process moves away from Dependabot.
- Trivy starts in reporting mode to establish a vulnerability baseline.
  Convert high/critical findings to a blocking gate after existing findings
  are triaged.
- The retained third-party WARP image downloads bootstrap data during startup.
  Its network/capability isolation limits exposure but does not make that
  upstream supply chain reproducible; replace it with a reviewed, immutable
  sidecar if a suitable maintained image becomes available.
- A quiet runtime window is not production verification. Candidate behavior
  must be observed under representative PAPER load before stable promotion.
