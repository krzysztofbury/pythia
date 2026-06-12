# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 9.7.x   | Yes       |
| 9.6.x   | Yes       |
| 9.0.x – 9.5.x | No — upgrade to 9.7 (v9.6 fixed a fail-closed entry gate) |
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
- The primary sentiment model (v9.6+: Gemini via `SENTIMENT_MODEL`) requires its provider key (`GEMINI_API_KEY` by default). Other provider keys (Anthropic, OpenAI, xAI) are needed only for the configured fallback or the LLM Challenger. All provider HTTP calls go through LiteLLM, which is configured for read-only completions; the LLM never issues trade instructions.

### Database
- `trades.db` contains full trading history including prices, quantities, and P&L
- The `logs/` directory is gitignored and should never be committed
- Database backups (`.backup-*`) contain the same sensitive data

### Docker
- The `.env` file is mounted read-only into the container
- No secrets are baked into the Docker image
- The WARP proxy sidecar handles Polymarket geo-restriction only

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
6. Review `logs/bot.log` regularly for anomalies — the boot-time `Effective config:` line (v9.5) confirms the loaded universe and feature flags
7. Keep your `.env` file permissions restricted: `chmod 600 .env`
