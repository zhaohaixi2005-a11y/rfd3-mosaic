# Security policy

## Sensitive data

Do not commit credentials, private keys, access tokens, cluster passwords,
unpublished structures, local `.env` files, model checkpoints, run outputs or
institution-specific private records.

Use `.env.example` as the environment template. Execution profiles committed
to the repository must contain placeholders or portable defaults rather than
personal paths, accounts or credentials.

## Reporting a vulnerability

Do not open a public issue containing credentials, unpublished structures or
security-sensitive logs. Contact the repository maintainers privately or use
GitHub's private security-advisory mechanism when it is available.

If a credential has been pasted into an issue, chat, log or commit, revoke or
rotate it immediately. Removing it from the current file is not sufficient
because Git history and external copies may still retain it.
