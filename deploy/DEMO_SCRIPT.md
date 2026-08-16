# Demo Video Script — 3:35

Commands are for **Command Prompt (cmd.exe)** with the `mlops-a4` conda environment active —
that's what makes plain `python` resolve correctly. Run everything from the repo root.

> PowerShell: replace `set X=1` with `$env:X = "1"` and `timeout /t 20 /nobreak` with
> `Start-Sleep 20`. Chain with `;` instead of `&&`.

**The arc:** here's the problem → here's what we built and how the pieces fit → here's the model
working → here's proof it's the real model → watch it break → watch the system catch it by itself.

**Only one command runs after the 1:50 mark.** Everything in Parts 4, 5 and 6 happens on the
Grafana and Prometheus screens — the API detects its own drift, so there is no analysis script to
run on camera.

Narration is written to be **read out loud**, in the team's voice ("we"). Every tool gets
explained in plain English before it gets used, so the audience is never looking at something
they don't have a name for.

---

## STEP 0 — Prep (BEFORE recording)

```cmd
cd "c:\Users\wasee\OneDrive\Desktop\college\Quarter4\MLOps\finalP\repo"
set PIP_TRUSTED_HOSTS=1
docker compose build
docker compose up -d
timeout /t 20 /nobreak
python src/prepare_data.py
python monitoring/drift_simulation.py
docker compose down -v
```

**Do not record the build** — it's two minutes of pip output.

Open five tabs in this order, then come back to tab 1:

1. `http://localhost:8000/docs` — the API
2. `http://localhost:3000/d/readmission-monitoring` — Grafana, **Last 15 minutes**, refresh **5s**
3. `http://localhost:9090/alerts` — Prometheus
4. `deploy/demo_request.json` open in your editor, ready to copy

Only three browser tabs are used on camera. Evidently runs *inside the API container*, so its
detection shows up on the Grafana tiles rather than needing a tab of its own. The standalone
Evidently HTML reports in `monitoring/reports/` are the deeper offline analysis — a repo
deliverable that belongs on a presentation slide, not in the video.

Terminal on the left, browser on the right, record the whole desktop. **Do a full dry run first.**

**Turn the terminal font up** before you record — Part 3 is pure terminal output, and default
console text is unreadable once the video is compressed and played at anything under fullscreen.
In Windows Terminal: Ctrl+`+` a few times, or Settings → Appearance → font size 16–18.

---

# PART 1 · The problem and the system — 0:00 to 0:50

**Start on the terminal.**

```cmd
docker compose up -d
timeout /t 20 /nobreak
docker compose ps
```

> "This project predicts whether a diabetic patient will end up back in hospital within thirty
> days of being discharged. Hospitals care about that, because a readmission usually means the
> patient wasn't quite ready to go home — and it's expensive for everyone.
>
> So we trained a model that scores that risk. But a trained model sitting in a notebook isn't
> much use to anyone. What we're showing today is the other half — actually running it, and
> knowing whether it's still working once it's live.
>
> That's one command, and it starts three containers."

*Let `docker compose ps` land, then explain the architecture over it.*

> "There are really two things running here.
>
> The first is the model itself, wrapped in a web API so anything can send it a patient and get a
> risk score back. That's the product.
>
> The second is the monitoring around it, and it's three layers.
>
> **EvidentlyAI** is our drift detector, and it runs *inside* the container alongside the model.
> Every batch of patients the API serves, Evidently compares against a sample of the training
> data and decides whether the incoming data has actually changed. So the service diagnoses
> itself — nobody has to run an analysis.
>
> **Prometheus** is the collector. Every five seconds it pulls Evidently's verdict off the API,
> along with live serving statistics, and stores the history — so we can see not just what's
> happening now, but when it started.
>
> And **Grafana** is the screen. It doesn't calculate or store anything itself — it just reads
> from Prometheus and puts it all in one place.
>
> So: Evidently detects, Prometheus remembers, Grafana shows us."

*Click through the three browser tabs, about five seconds each, and refresh them.*

> "Here they are. The model API. The dashboard — empty, because nothing's happened yet. And this
> is where alerts appear if something goes wrong. Nothing so far, everything's clean.
>
> Let's start with the model itself."

---

# PART 2 · The model, working — 0:50 to 1:25

**Tab 1, Swagger.** Expand **POST /predict** → Try it out → paste from `demo_request.json` → Execute.

> "This is the live service. Let's send it two patients.
>
> Both are real records from our test data. The first is a fairly ordinary hospital stay. The
> second is a similar patient — but this one has been admitted six times before, and been to the
> emergency room three times."

*Execute. Point at the two numbers.*

> "Twenty-seven percent for the first. Ninety-eight percent for the second.
>
> How many times someone's been admitted before is the strongest signal the model has, and you can
> see how hard it pushes the risk up. That's the product — a doctor could look at this at
> discharge and decide who needs a follow-up before they go home."

---

# PART 3 · Proof it's the real model — 1:25 to 1:50

**Screen: the terminal.** Full screen it, or at least make it the focus — this section has no
browser component. The output *is* the visual.

```cmd
python monitoring/baseline_validation.py
```

> "Two patients doesn't prove much though. So this is the real check.
>
> It takes our whole test set — almost fourteen thousand records the model has never seen — sends
> every one of them through the running container, and compares the results against the scores we
> got back when we trained it."

*Takes about 10 seconds. This is what appears:*

```
[baseline] API health: {'status': 'ok', 'model': 'XGBClassifier', ...}
[baseline] clean test set: (13998, 16)

[baseline] Live API vs recorded training metrics:
  f1_score   live=0.436  recorded=0.436  Δ=0.000  OK      <-- point here
  auc_roc    live=0.648  recorded=0.648  Δ=0.000  OK
  precision  live=0.563  recorded=0.563  Δ=0.000  OK
  recall     live=0.355  recorded=0.355  Δ=0.000  OK

[baseline] Drift (clean train vs clean test): drifted_cols=0/18 share=0.000 dataset_drift=False
[baseline] Combined drift + performance report -> monitoring/reports/00_baseline.html

============================================================
  Performance parity : PASS                                 <-- and here
  Clean = no drift   : PASS
============================================================
```

**Point at two things, in this order:**

1. The **`Δ=0.000`** column — `live` vs `recorded` matching on all four metrics.
2. The **two PASS lines** at the bottom.

*Then say:*

> "Zero difference, on every metric. So the model inside the container is exactly the model we
> trained — nothing got lost or changed along the way.
>
> It also checks whether this clean data looks any different from the training data. It doesn't.
> And that matters, because this run becomes our reference point — the picture of *normal* that
> everything from here gets compared against.
>
> So right now, the system is healthy. Let's break it."

---

# PART 4 · Breaking it — 1:50 to 2:55

**Start the command, then switch to tab 2, Grafana, and stay there for the whole minute.** This
is the only command in the rest of the demo — everything after this happens on the dashboard.

```cmd
python deploy/demo_traffic.py --duration 60
```

> "Back to the dashboard, and now it's live. Watch these panels.
>
> This dashed line is normal — out of any group of patients, the model usually flags about a
> quarter of them as at risk. And up here, zero drifted columns. That's healthy.
>
> We're now sending a steady stream of patients through the API. And as it runs, we're slowly
> corrupting that stream — feeding it a much older, much sicker population than the model was
> ever trained on."

*[~20s in, as the line lifts off the baseline]*

> "And there it goes. The line's climbing away from normal.
>
> Now here's the part that really matters. **The API is not erroring.** Not once. It's happily
> handing back confident predictions for every one of these patients. It has no idea anything
> changed.
>
> That's how models actually fail in the real world. Nothing crashes. Nothing shows up in the
> logs. The model just quietly starts being wrong — and if nobody's watching, you find out months
> later."

*[~40s in, as the drift tiles flip red. Measured — it happens between 35 and 40 seconds and
stays red for the rest of the run.]*

> "And now the top row's caught it. Drift status has gone red. Evidently has flagged four
> columns as significantly different from the training data — length of stay, emergency visits,
> prior admissions, and age. Which is exactly the population shift we injected.
>
> Nobody ran anything to make that happen. Evidently is running inside the container, testing
> every batch as it arrives. It noticed on its own."

*[~55s in, line near the top]*

> "Ninety-five percent now. It's flagging almost every patient as high risk, which is useless to
> a doctor. But from inside the API, the requests still all look completely fine."

---

# PART 5 · The alerts — 2:55 to 3:20

**Switch to tab 3, Prometheus. No terminal, no new commands.**

> "And because Prometheus is storing all of this, it can act on it.
>
> Three alerts firing. Too many columns have changed. Too much of the data has changed. And the
> predictions themselves have shifted too far outside the normal range.
>
> We watch those last two separately on purpose — some kinds of bad data change what goes *in*
> without changing what comes *out*. Watch only one, and you'd miss half your problems.
>
> In a real deployment these would page somebody, or automatically roll back to the previous
> model version. Here they just go red on screen — but the detection is the hard part, and that's
> done."

---

# PART 6 · Close — 3:20 to 3:35

**Back to tab 2, Grafana, fully red.**

> "So that's the full picture. The model in a Docker container, served through FastAPI. Validated
> against data it had never seen. EvidentlyAI detecting drift from inside the running service,
> Prometheus keeping the history, Grafana putting it on one screen, and alerts firing the moment
> it goes wrong.
>
> And all of it comes up with one command. Repo link's on the slide."

---

## STEP 8 — Stop everything

```cmd
docker compose down -v
```

Removes containers, network, and metric volumes. The image stays cached, so the next
`docker compose up -d` takes about five seconds. To delete the image too:
`docker rmi diabetes-readmission-api:1.0.0`

---

## Between takes

```cmd
docker compose down -v && docker compose up -d && timeout /t 20 /nobreak
```

The only reliable reset — it wipes Prometheus history so the graphs start empty.

## Delivery notes

- **Part 1 is the load-bearing section.** If the audience doesn't leave it understanding
  "Evidently detects, Prometheus remembers, Grafana shows us", nothing after it lands properly.
  Slow down there. It's the one place worth over-explaining.
- **The one line that matters** is in Part 4: *the API never errors*. If you land nothing else,
  land that. It's the whole reason the monitoring half of this project exists.
- **Say the transitions.** "So right now the system is healthy — let's break it." "Nobody ran
  anything to make that happen." Those are what make it a demo instead of a tour of screens.
- **Don't fill Part 4's sixty seconds with talking.** Three beats, silence in between. Let the
  graph move — dead air while something visibly happens is good pacing, not a mistake.
- **Numbers out loud, not read off.** "About a quarter" beats "zero point two five seven". Point
  at the exact figure on screen instead.
- **Don't poke Swagger between takes** and then point at Grafana — a single-record request pins
  the ratio to 0% or 100% and makes the graph look like noise.
- **The drift tiles lag the graph by design.** The API recomputes drift every 1,000 rows served,
  so the top row flips red roughly 15-20 seconds after the line starts climbing. That gap is
  useful — narrate the climb first, then let the detection land as its own beat.
- **If you run long,** trim Part 3's middle paragraph. Never trim Part 1.
- **Individual contributions** are stated during the live presentation, per the assignment — this
  video stays on the product.
- **Live numbers vs. the offline reports.** Both are Evidently with the same configuration, but
  the in-container monitor sees the 16 input features (4 drift) while the offline run also sees
  the prediction and the target columns, 18 in total (5 drift). Same conclusion, different
  denominators — worth knowing if someone compares the video against `DRIFT_SUMMARY.md`.
