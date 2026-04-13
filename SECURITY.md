# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 8.2.x   | Yes       |
| < 8.2   | No        |

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
- Keys are never logged, printed, or included in error messages
- Kraken API keys should be configured with **minimal permissions**: Query Funds + Modify Orders only. **Never enable Withdrawal permissions.**

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
- Grok (xAI) API calls use HTTPS
- Only Polymarket traffic routes through the WARP SOCKS5 proxy
- No inbound ports are exposed

## Best Practices for Operators

1. Use a dedicated Kraken API key with **no withdrawal permissions**
2. Set `MAX_POSITION_SIZE_USD` to limit per-trade exposure (hard cap: $100)
3. Set `DAILY_LOSS_LIMIT_USD` to pause trading after losses
4. Monitor via Discord/Slack webhooks (`WEBHOOK_URL`, `ERROR_WEBHOOK_URL`)
5. Review `logs/bot.log` regularly for anomalies
6. Keep your `.env` file permissions restricted: `chmod 600 .env`
