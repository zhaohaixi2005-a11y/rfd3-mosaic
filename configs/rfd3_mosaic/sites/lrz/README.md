# LRZ deployment profiles

These profiles preserve site-specific deployment settings for existing
RFD3-Mosaic experiments. They are not portable defaults or public hardware
requirements, and they are not packaged as the installed distribution's
general execution profiles.

The short legacy profile identifiers `v100`, `p100`, `a100_80g` and `h100`
continue to resolve from a source checkout so frozen experiment YAML files do
not need to change.

Checkpoint fields use the portable `~/.foundry/checkpoints` convention. Copy
a profile to a site-local, ignored file when a deployment requires different
paths or shell setup. Never replace the committed placeholders with a user
account, credential or private filesystem path.
