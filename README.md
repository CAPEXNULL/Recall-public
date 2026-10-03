# Recall

Recall links a suggested action to a user-reported outcome and reuses successful experience in later, lexically similar situations. Learning updates an in-memory experience store; it does not train model weights. Actions are suggestions and are never executed by the application.

Public repository: https://github.com/CAPEXNULL/Recall-public

## Architecture

`Situation -> Decision -> Outcome -> Experience -> Transfer -> New Decision`

- `app.py`: English Flask UI, isolated server-side browser sessions, signed cookies and one-time form tokens.
- `recall_core.py`: outcome linkage, lexical experience retrieval, limited service-name adaptation, explicit action-constraint checks and provenance.
- `model_client.py`: real NVIDIA Nemotron inference through Nebius Token Factory, JSON-schema output validation and no automatic retries.
- `demo/scenario.json`: simulated Alpha-to-Beta incident and reported successful outcome.
- `tests/test_runtime.py`: original Recall regression cases for service adaptation, blocked transfer, state preservation, the Flask UI and mocked provider requests.

All runtime code and required assets are included. There is no database, semantic extraction adapter, background worker or bundled model weights. Memory lasts within one server process; restart or 24 hours of inactivity clears it. Each session permits at most 100 decisions; the server permits at most 1,000 sessions.

## Matching and transfer

Features are Unicode alphanumeric words/numbers, normalized with NFC and case-folding, deduplicated and sorted. Similarity is Jaccard, `|A intersection B| / |A union B|`. An experience is eligible when reported success is true, outcome score is at least 0.50 and similarity is at least 0.50. Ties resolve by similarity, score, normalized action and source experience ID.

Recall selects one eligible experience. A limited English/Russian grammar detects newly stated prohibitions or unavailability of the proposed action. If it detects a critical mismatch, transfer is blocked and a fresh model recommendation is requested; the source experience and blocking reason remain visible. This grammar is not a general semantic applicability evaluator.

For an unambiguous service-name substitution with matching surrounding text, Recall updates the service-qualified name in the reused action. Ambiguous names or changed context leave the original action intact. Users must review every suggestion, including object names.

## NVIDIA and Nebius

`model_client.py` sends HTTPS requests to `https://api.tokenfactory.nebius.com/v1/chat/completions`. Nebius Token Factory hosts inference; an NVIDIA Nemotron model returns structured `features`, `action` and `reason`. The adapter requires an NVIDIA Nemotron model ID, requests JSON-schema output, checks canonical features and validates the response before saving state. Transferring an experience makes no model request.

The original project's recorded live integration check used `nvidia/Nemotron-3_5-Lightning`. That historical observation does not establish present account availability or general recommendation quality. Configure an NVIDIA Nemotron model enabled in your account. Packaging and publication do not run paid model checks.

## Setup and run

Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$recallCredential = Read-Host 'Nebius API key' -AsSecureString
$env:NEBIUS_API_KEY = [System.Net.NetworkCredential]::new('', $recallCredential).Password
$env:NEBIUS_MODEL_ID = 'nvidia/Nemotron-3_5-Lightning'
$env:RECALL_SESSION_SECRET = '<random private signing secret>'
python app.py
```

Open http://127.0.0.1:5000. `.env.example` lists variable names; `.env` is not loaded automatically. Without credentials the UI reports the missing connection and does not substitute a fake model. Live provider requests incur charges.

For hosting, use HTTPS termination and one process:

```text
waitress-serve --host=0.0.0.0 --port=8080 --call app:create_app
```

Set `RECALL_HTTPS=1` when browsers access the site through HTTPS. Supply secrets in hosting environment settings. Memory is volatile and is not shared between workers.

## Demo

1. Submit the prefilled service Alpha incident. The first decision has source MODEL.
2. Report the simulated success, score 0.8 and feedback from `demo/scenario.json`.
3. Submit `situation_b`, changing Alpha to Beta. The source becomes EXPERIENCE; inspect the source experience, similarity, score and adapted action.
4. Change `Rollback is available` to `Rollback is unavailable`. For a rollback action, Recall blocks transfer and asks the model for a new recommendation.

The operational incident and outcome are simulated; no infrastructure action is executed. The real model may choose an action other than rollback, so the blocking step applies only to an action covered by the stated constraint. Local tests use an explicit scripted rollback action to verify this path deterministically.

## Offline verification

```powershell
python -m unittest discover -s tests -p 'test_runtime.py' -v
python examples/transfer_demo.py
```

The tests retain the original Recall regression bodies and use scripted model responses, Flask's test client and mocked HTTPS transport. No API credentials, browser installation, private manifests or paid calls are required. They check local behavior, not live recommendation quality. The example verifies MODEL -> successful outcome -> EXPERIENCE and explicit blocked transfer -> MODEL.

## Limits, license and submission

Lexical overlap can miss synonyms, changed circumstances and unrecognized negation. Absence of a recognized prohibition does not prove an action is appropriate. Outcomes are user reports, not independent measurements. A later failure does not invalidate an earlier successful experience. Memory is per-browser and temporary; no production reliability or unrestricted-domain generalization is claimed.

MIT license, copyright CAPEXNULL: see [LICENSE](LICENSE). [Hackathon updates](HACKATHON_UPDATES.md) describe the original project's recorded changes. Source file origin and checksums are in [SOURCE_ORIGIN.json](SOURCE_ORIGIN.json) and [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). Internal logs, credentials and author materials are excluded.

A deployed working demo, video and Devpost submission still require their published URLs. Repository publication alone does not complete the competition submission.
