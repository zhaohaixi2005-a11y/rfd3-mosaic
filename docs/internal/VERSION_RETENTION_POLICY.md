# RFD3-Mosaic version retention policy

This policy keeps historical work reproducible without allowing it to compete
with the current product path.

## Current development line

There is one active code line, mirrored to two remotes:

```text
personal: refactor/product-core-v1
lab:      hx/rfd3-mosaic-product-core
```

New code is committed to the local `refactor/product-core-v1` branch, tested,
then pushed to both names. The two remote heads must identify the same commit.
The branch names differ only because the lab repository requires an
initials-prefixed branch.

Old feature and snapshot branches are read-only references. They are not
alternative current implementations and must not receive new development.

## Repository files

- Maintained entry points stay in `scripts/rfd3_mosaic/`.
- Scripts that directly invoke the old adapter/RFD3 path stay in
  `scripts/rfd3_mosaic/archive/legacy_direct/`.
- Current and useful development YAMLs stay in `experiments/`.
- Inputs with a direct newer replacement stay in
  `experiments/archive/superseded/`.
- A historical file is moved with `git mv`, never copied. This preserves
  history and prevents two editable copies from drifting apart.

Repository-layout tests enforce that active scripts do not bypass the public
worker and current release gates never point into an archive.

## Run results

Run directories are immutable evidence. They remain organized by UTC day,
campaign, experiment and job/run ID. Their `software/` snapshots are not
source checkouts for new development; they are frozen records of what a run
actually executed.

Do not rename an old result to look current. A scientific comparison must
record the campaign manifest, repository commit, resolved configuration and
pose manifest. New runs use a new campaign directory rather than overwriting
old outputs.

## Promotion and retirement

To make a diagnostic experiment current:

1. add or update one active YAML;
2. add it to the maintained launcher mapping;
3. add CPU coverage and the required GPU evidence plan;
4. archive the directly superseded YAML;
5. update project status only after evidence is collected.

To retire a path, move it to the appropriate archive and update its internal
relative paths. Do not delete it merely to make the directory look cleaner.
Git history is the final recovery mechanism, but archived files should remain
readable whenever practical.

