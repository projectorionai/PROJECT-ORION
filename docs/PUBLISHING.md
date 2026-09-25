# Publishing ORION source

This procedure exports current source without the live installation's private
data or Git history. It does not publish anything automatically.

## 1. Review the source

```powershell
.venv\Scripts\python.exe -B tools/prepare_public_source.py
```

The check includes tracked and new, non-ignored files. Deleted files are excluded.
Paths outside the reviewed layout, build binaries and recognised credentials
block the export. No whole test folder is exempt; synthetic credential tests
construct their values at runtime.

For stronger local checking, add existing private credential stores:

```powershell
.venv\Scripts\python.exe -B tools/prepare_public_source.py `
  --secrets-file config/api_keys.json `
  --secrets-file config/messaging.json
```

Their sensitive values are checked in memory; stores and values are not printed
or exported. Optionally supply `--private-terms-file` pointing to a JSON list of
private names/emails, stored **outside this repository**. Findings show locations
and categories without matched values. Also review prose, examples and new files:
pattern checks cannot recognise all private information.

## 2. Create a source-only ZIP

```powershell
.venv\Scripts\python.exe -B tools/prepare_public_source.py `
  --secrets-file config/api_keys.json `
  --secrets-file config/messaging.json `
  --output release/ORION-source-2026-09-18.zip
```

Use a new filename each time: existing files are never overwritten. The ZIP
contains exactly the scanned bytes and a SHA-256 `SOURCE_MANIFEST.json` beside
the `ORION/` source folder. It
excludes `.git`, installed executables, virtual environments and live config.
The `release/` directory is ignored by Git.

## 3. Start a fresh publication directory

Extract the ZIP outside the live installation. Initialise a **new repository**
in its `ORION` folder. Do not copy the old `.git` directory. The neighbouring
`SOURCE_MANIFEST.json` records the archive's contents and is not an application
input; keep it outside the new repository.

```powershell
git init -b main
git add .
git status --short
```

Review, commit with the intended author identity and connect to the intended
GitHub destination. These instructions deliberately do not supply a remote URL,
push existing history or force-update an existing repository.

**History is a separate risk.** A clean ZIP does not repair old commits, existing
remotes, forks or distributed builds. Rotate credentials that appeared in them.
If preserving old history matters, sanitise it as a separate operation before
making it public. `.gitignore` does not provide that protection.

## 4. Build from clean source

Create a new environment and install requirements in the extracted directory.
`build_standalone.py` seeds only explicitly listed public defaults. It stops if
`dist/ORION` has personal configuration, preventing a rebuild from erasing or
distributing a personalised installation.

Docker uses a deny-by-default `.dockerignore`, limiting its context to code,
server requirements and the same public defaults. Supply private settings at
runtime. Old executables and container images do not become sanitised when
source rules change.

Downloaded engines/dependencies are absent from the source ZIP. Review their
redistribution terms separately for compiled builds. This cleanup does not
assign a new project licence.
