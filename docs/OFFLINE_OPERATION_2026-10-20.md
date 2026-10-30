# Running ORION offline

From the ORION project folder, start the installed application with:

```powershell
& '.\dist\ORION\ORION.exe' --offline
```

This selects local routing for that session without deleting or rewriting saved
cloud credentials. On first-run key setup, **Continue offline** starts the local
system without a Gemini key. Cancel still exits setup. With normal automatic
startup, ORION can also fall back when connectivity is lost.

The desktop ORB and face use the bundled Three.js files through a local Qt URL
scheme. They do not need a CDN. If the renderer or host fails, a software orb
keeps ORION visible, and the window reports the failure and attempts bounded
recovery. An animation counter alone is insufficient: readiness now depends on
completed renders. The separate compact orb is painted locally with Qt.

Offline mode suppresses connectivity probes and uncached cloud speech requests
and cache warming. Previously cached speech can still play. Kokoro supplies
local speech when its model is installed; local transcription uses an installed
Whisper or Vosk model. Missing models are reported as unavailable, without
silently fetching them during an offline turn. The standard local commands and
rule-based conversation do not require a language model or API key. A configured
local model can handle broader conversation; its server must be running.

Useful local functions include time/date and arithmetic, saved memory,
reminders/timers, local file lookup and application control. Existing write and
confirmation rules still apply. Ask for intelligence mode to see local versus
cloud availability. These shipped analysis plugins work from supplied data:

- `orion_latency_report`: summarises supplied stage timings; it does not invent
  measurements or monitor a live service.
- `orion_batch_planner`: groups supplied tasks by dependencies without executing
  them.
- `orion_dataset_compare`: compares supplied rows using their identifiers.

Structured workbook export, charts, supported formulas and document/data
inspection can also run locally. Generating a complex document specification
from prose may require a configured local language model. Internet searches,
fresh weather/prices, cloud voice and remote account services still need their
respective connections. A local/LAN integration and an internet service have
different requirements; an available local tool does not imply a remote account
is connected.

Verification should include actual offline ORB and face pixels, progressing
rendered frames, forced renderer/host failure, local command dispatch, cached
speech generation and transcription, and the installed executable. File presence
or an open host window alone is insufficient. Device-free audio tests establish
PCM generation/transcription; microphone and speaker quality require a separate
physical check.

The implementation follows Qt's requirement to register local URL schemes before
creating the application and uses Three.js's supported local-module deployment:
[Qt local URL scheme handlers](https://doc.qt.io/qt-6/qwebengineurlschemehandler.html),
[Three.js installation](https://threejs.org/manual/pages/installation.html).
