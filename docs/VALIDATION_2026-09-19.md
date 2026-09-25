# Source validation — 19 September 2026

The source accompanying the [18 September capabilities inventory](CAPABILITIES_2026-09-18.md)
was tested on Windows 11 AMD64, Python 3.13.14, with Qt's offscreen platform.
Tests ran from a separate source-only checkout, without the live application's
credentials, memories, browser profile or installed executable.

## Results

- **5,910 passed, 4 skipped, 0 failed, 0 errors** in 169.20 seconds.
- All **652 Python source files** compiled without writing bytecode caches.
- The publication audit found no recognised credential patterns, configured
  credential values or supplied private terms in the exported source.
- The archive contains a SHA-256 source manifest and excludes old Git history,
  runtime databases, private configuration, browser data and installed binaries.

Command used in the isolated checkout:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B -m pytest -q -p no:cacheprovider --tb=short
```

The four skips concern unavailable Windows symlink privileges, an unavailable
optional swarm model, deliberately absent private messaging configuration, and
an absent local self-repair plugin. Seven warnings came from the speaker-ID
dependency: a SciPy deprecation and numerical warnings when testing silence.

## Fixes covered

Regression coverage includes PCB inspection and camera lifecycle, startup,
privacy and export boundaries, profile-driven missions, process-action guards,
clipboard filtering, flight-search parsing and face animation. Recent additions
exercise bounded journal reads, repeated-fault sampling and test verification
that correctly rejects collection errors, nonzero exit codes and timeouts.
Signal-wiring inspection now works when the checkout lives below a directory
named `release`, and prunes generated directories before traversing them.

## Limits

This is automated source validation using the installed development dependencies,
not a fresh dependency installation or certification of every integration.
Live PCB hardware, microphones, cloud providers, browser flight results, remote
deployment and newly built executables were not tested end to end in this pass.
Only documentation changed after the tested code snapshot.

Pattern matching and known-value comparisons cannot establish that arbitrary
private prose or unknown secrets are absent. Old commits and existing builds
require separate handling; see [Publishing ORION](PUBLISHING.md).
