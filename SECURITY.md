# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 10.0.x  | Yes       |
| 9.x     | No - retain only as a read-only archive and upgrade to 10.0.0 |
| < 9.0   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in Oracle Pythia, please report it responsibly:

1. **Do NOT open a public issue** for security vulnerabilities
2. Email the maintainer directly or use GitHub's private vulnerability reporting feature
3. Include a description of the vulnerability and steps to reproduce
4. Allow reasonable time for a fix before any public disclosure

## Security Considerations

Oracle Pythia handles sensitive financial data and API credentials:

### API Keys
- All API keys are loaded from `.env` files using `pydantic.SecretStr`
- Keys are never logged, printed, or included in error messages. v9.1's `llm_client._sanitize` additionally scrubs anything resembling an API key from LLM error strings before they hit the log file or Discord webhook.
- Kraken API keys should be configured with **minimal permissions**: Query Funds + Modify Orders only. **Never enable Withdrawal permissions.**
- LLM features default off. Enabling one requires keys only for its selected
  primary/fallback providers. Provider calls go through LiteLLM; LLM output
  never submits orders and remains shadow-only in LIVE unless influence is
  separately armed.

### Database
- `paper-execution-v10.db` or `live-execution-v10.db` is authoritative and
  contains sensitive exact order, fill, position, and protection data.
- `analytics-v10.db`, `llm-budget.db`, PAPER checkpoints, and all database
  backups contain sensitive operational or financial data.
- A v9 `trades.db` is a read-only archive, never v10 execution authority. The
  QNAP layout mounts it separately at `/archive/v9:ro`; v10 is not configured
  to open it.
- The `logs/` directory and `deploy/qnap/*.env` are gitignored and must not be
  committed.

### Docker
- Env-file values are injected at container creation and are not baked into the image
- No secrets are baked into the Docker image
- The WARP proxy sidecar handles Polymarket geo-restriction only
- Bot containers run as UID/GID 10001 with a read-only root and a dedicated
  persistent `/app/logs` volume. Verify volume write/rename ownership before
  startup.
- Watchtower is optional, profile-gated, and has daemon-equivalent Docker socket
  access. Private GHCR use requires a protected read-only Docker config mount.
  It does not perform health-gated rollback.

### Network
- All Kraken API calls use HTTPS via CCXT
- LLM provider API calls (Grok/xAI, Anthropic, Gemini, OpenAI) use HTTPS via LiteLLM
- Optional funding-rate fetcher (v9.5) uses read-only CCXT to a secondary exchange (Binance by default). No orders are placed on the secondary exchange.
- Only Polymarket traffic routes through the WARP SOCKS5 proxy
- No inbound ports are exposed

## Best Practices for Operators

1. Use a dedicated Kraken API key with **no withdrawal permissions**
2. Set `MAX_POSITION_SIZE_USD` to limit per-trade exposure (ceiling $500 since v9.2; default remains $50 — only raise deliberately)
3. Set `MAX_POSITION_PCT_OF_BALANCE` (v9.2, default 40%) and `MAX_TOTAL_CRYPTO_POSITIONS` (v9.5, default 5) to cap portfolio concentration
4. Set `DAILY_LOSS_LIMIT_USD` to pause trading after losses
5. Monitor via Discord/Slack webhooks (`WEBHOOK_URL`, `ERROR_WEBHOOK_URL`); v9.1 HIGH-concern LLM critiques are routed to `ERROR_WEBHOOK_URL`
6. Review `logs/bot.log` and the structured health document regularly for anomalies
7. Keep your `.env` file permissions restricted: `chmod 600 .env`
8. Record image digests before promotion and rollback by exact digest; do not
   treat a moving tag as deployment authority
