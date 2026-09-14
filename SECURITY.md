# Security policy

## Supported version

Security fixes are applied to the latest `v1.x` release.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability or exposed secret. Contact the repository owner privately with a concise reproduction and impact description. Rotate any API key that may have been exposed before reporting it.

## Secret-handling rules

- Use environment variables or a local `.env.local` file for credentials.
- Do not place production credentials in `config/settings.yaml`.
- Check `git status` before every push.
