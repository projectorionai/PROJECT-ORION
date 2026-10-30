# ORION on Render Free: a disposable cloud trial

Prepared and checked against Render's documentation on **30 September 2026**.
The configuration runs ORION's existing headless brain behind Render's HTTPS
endpoint. It does not publish an APK, control the PC or grant access to phone
permissions. It is a single-owner service; separate customers must not share
this pairing registry, conversation memory or provider key.

## What is free, and what is not permanent

Render Free supplies managed HTTPS and a web-service container. Its current
allowance is 750 running hours per workspace per calendar month. The service
sleeps after 15 minutes without inbound traffic; waking takes about a minute.
Each sleep, restart and deployment loses local files. Free services cannot
attach persistent disks, and free Render Postgres expires after 30 days.
These are [Render's published Free limits](https://render.com/docs/free).

**This profile therefore loses ORION's SQLite memory, revocations and pairing
registry whenever the container is replaced.** Previously issued credentials
then fail authentication; the phone must pair again using the fresh code from
the private service log. Do not copy a previous signing secret alone: restoring
an old registry could undo device revocations. This is a trial, not a reliable
always-on home or customer deployment. There is no keep-alive job intended to
evade the free plan's sleep behaviour.

Hosted AI quotas and charges are separate. Render may suspend free services
with unusually high outbound API traffic. Its free tier blocks SMTP ports
25, 465 and 587, so mail integrations must use their authorised HTTPS APIs.
Check the account's spending limit and usage dashboard before deployment.

## Deploy the reviewed source

1. Sign in to [Render](https://dashboard.render.com/) and connect your Git
   provider to a **private, reviewed source repository**. Do not upload the
   desktop folder, `.env`, live `config/`, databases, logs, recordings,
   `.runtime-cache`, API keys or APK signing keys. The committed `.dockerignore`
   limits the build context; `Dockerfile.render` copies only source and public
   knowledge packs. Review the repository itself before granting access:
   Docker filtering does not remove files already committed to Git history.
2. Create **New → Blueprint** for the reviewed branch. Set the Blueprint path
   to **`deploy/render-free.yaml`**. Confirm the only resource is the
   **`orion-cloud-trial` Free web service**, with region **Frankfurt**. No paid
   disk, database, worker or preview resource is requested.
3. Enter your Gemini key privately in the requested
   **`ORION_GEMINI_API_KEY`** environment value. The Blueprint has `sync: false`;
   it contains no key. This places the key in the cloud account's runtime, which
   is a different credential location from the PC and paired phone. Anyone
   with access to its environment or service logs must be trusted.
4. Deploy. Render provides `RENDER=true`, `RENDER_EXTERNAL_URL` and `PORT`.
   `render_entrypoint.py` checks these before importing ORION, binds
   `0.0.0.0:$PORT` **inside the isolated container**, advertises only the assigned
   `https://…onrender.com:443/` origin and lets Render terminate TLS.
   Forwarded headers never authenticate a caller or select an endpoint.
5. Read the first-run nine-character pairing code from the **private** Render
   service log. It expires after ten minutes and is single-use. Use the service
   HTTPS address and port **443** in ORION's existing server pairing flow. Pairing
   grants access to the cloud brain, not to a switched-off PC.

The file [format](https://render.com/docs/blueprint-spec), Docker build and
[platform environment variables](https://render.com/docs/environment-variables)
are described in Render's official documentation. Auto-deployment is off;
subsequent releases are explicit manual deployments. Native phone-only mode
continues to use its own encrypted provider credential; selecting this service
requires testing its server-mode conversation path separately.

## Check the endpoint without sending a credential

Replace the example with the hostname Render actually allocated:

```powershell
$orionCloudUrl = 'https://YOUR-SERVICE.onrender.com'
Invoke-RestMethod -Uri "$orionCloudUrl/api/health" -TimeoutSec 90
try {
    Invoke-WebRequest -Uri "$orionCloudUrl/api/pc/power" -TimeoutSec 15
} catch {
    # Without a bearer token this must return HTTP 401.
    $_.Exception.Response.StatusCode
}
```

Health must return ORION JSON rather than a platform loading page. After sleep,
retry the health check once the service has completed its cold start. Do not
turn a failed check into an unbounded retry or send a pairing token to an
unverified hostname. A public health response does not prove voice, model
quota, long answers or phone actions work; those require a paired-client test.

## Before using cloud hosting permanently

Use a host with a durable filesystem and an adequate availability policy, or
implement a transactional durable datastore for **all** auth, revocation,
conversation, queue and action records. ORION currently uses local SQLite and
private files; adding a Postgres URL does not migrate those stores. On a paid
Render service, use a persistent disk and a reviewed backup/restore procedure,
and remove the trial launcher in favour of durable-state deployment settings.
The existing Oracle and VPS deployment scripts remain available for a
dedicated host. A consumed Oracle trial does not itself prove Always Free
compute is unavailable; eligibility and actual capacity are account-specific.

On 30 September the local Docker engine was started and the Linux trial image
built. A temporary loopback-only container returned health HTTP 200 in 1.204
seconds, refused unauthenticated PC power access with HTTP 401, ran as UID
10001 and stopped cleanly. Sampled memory was 38.21 MiB under a 512 MiB ceiling.
No credentials or private host files were supplied. This does not establish
Render cold-start behaviour, provider access or paired-phone compatibility.
No cloud resource was created, no repository pushed and no key sent to Render.

Verification completed: **58 focused tests passed**, critical Python checks
passed, and `render-free.yaml` passed Render's official
[Blueprint JSON Schema](https://render.com/schema/render.yaml.json). The tests
exercise pairing, revocation, bearer requirements, rejection of forged proxy
headers, invalid runtime settings and process replacement without putting a
provider key in command arguments. The separate local Linux acceptance above
does not establish real Render or phone acceptance.
