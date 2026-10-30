# Hosted AI providers

ORION's desktop provider router accepts OpenAI-compatible text providers. Its
existing options include Gemini, OpenRouter, Groq, Cerebras, SambaNova and
Together. The additional Cloudflare Workers AI, Mistral, Hugging Face and
NVIDIA profiles are disabled until their credentials are supplied. DeepSeek is
also available as an opt-in paid fallback. Free allowances have rate and usage
limits; none is an unlimited production service.

The `.env.example` file lists variable names only. ORION does **not** load a
`.env` file automatically. Set variables in the PowerShell session that launches
ORION, then restart it. For example:

```powershell
$env:ORION_CLOUDFLARE_ACCOUNT_ID = '<32-character-account-id>'
$env:ORION_CLOUDFLARE_API_TOKEN = '<Workers-AI-token>'
& python orion.py
```

Activate the project's working Python environment before the final command.

Cloudflare's default model is `@cf/google/gemma-4-26b-a4b-it`. The account ID
must be 32 hexadecimal characters. Its [OpenAI-compatible endpoint](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/)
requires both an account ID and a token. The [model page](https://developers.cloudflare.com/workers-ai/models/gemma-4-26b-a4b-it/)
describes its current pricing and capabilities.

For other providers, set the relevant key in the launch shell:

| Provider | Variable | Default model |
| --- | --- | --- |
| Mistral | `ORION_MISTRAL_API_KEY` | `mistral-small-latest` |
| Hugging Face | `ORION_HUGGINGFACE_API_KEY` | `openai/gpt-oss-120b:cheapest` |
| NVIDIA NIM | `ORION_NVIDIA_API_KEY` | `meta/llama-3.3-70b-instruct` |
| DeepSeek, paid | `ORION_DEEPSEEK_API_KEY` | `deepseek-flash` |

Each provider has an optional `ORION_<PROVIDER>_MODEL` variable. A saved profile
with a non-empty key and an explicit enabled setting takes precedence over
later environment changes. The provider router tries profiles in its saved
order and falls back when one fails; enabling more profiles may increase the
time before a failure is reported. Review the order in ORION's provider settings
if you want to keep paid providers behind free allowances.

These profiles serve the **desktop backend**. The Android app's standalone
PC-off mode uses its own provider configuration and does not inherit desktop
environment variables. Version 3.14 supports these additional services through
Options → Connections → Text AI provider; see
[phone integration setup](PHONE_INTEGRATIONS_2026-09-30.md). New desktop defaults
use Cerebras `gpt-oss-120b` and Together `Qwen/Qwen3.5-9B` at
`https://api.together.ai/v1`. Saved explicit model and endpoint choices remain
unchanged; update an obsolete saved profile deliberately in provider settings.

Routine desktop/headless replies also reduce optional model thinking on the
recognised official endpoints: DeepSeek `deepseek-flash` and `deepseek-v4-pro`
send `thinking: {"type": "disabled"}`; Cerebras `gpt-oss-120b` sends
`reasoning_effort: "low"`. Custom hosts and unrecognised models do not inherit
those service-specific fields. Set `ORION_MODEL_THINKING=1` before starting
ORION to retain the provider's normal thinking policy. This removes avoidable
reasoning work; it is not a live latency benchmark or a guarantee of faster
answers for every request.

Current provider documentation: [Mistral free mode](https://docs.mistral.ai/getting-started/quickstarts/studio/activate-and-generate-api-key),
[Hugging Face credits](https://huggingface.co/docs/inference-providers/en/pricing),
[NVIDIA development endpoints](https://build.nvidia.com/explore/discover?api-key=true),
and [DeepSeek API](https://api-docs.deepseek.com/guides/harness).
