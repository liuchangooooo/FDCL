# ICRA Teaser (Fig. 1) — Assembly Blueprint

Hand-assembled in a vector tool (Illustrator / Inkscape). Code only produces the
real ingredient assets; layout, arrows, labels, typography are done by hand.

Design follows four reference Fig.1s:
- **CoEvolve** — 3-row paradigm ladder + ✓/✗ property bar + feedback loop + a
  "Signal Extraction" panel (Forgotten/Rare/**Boundary** trajectories). ← main template.
- **EnvGen** — numbered loop steps + "Loop ×N" + real env thumbnails.
- **Difficulty-aware nav** — visceral ✓/✗ success-vs-failure on real renders.
- **DIVO (our base)** — Push-T visual grammar: **red initial T, green goal T,
  orange obstacles, black end-effector dot, dashed trajectories**. Our assets now
  match this grammar so DIVO-familiar reviewers map on instantly.

## Chosen structure: 3-row ladder (CoEvolve-style)

Three stacked rows, each a left→right icon pipeline with labeled arrows, and a
grey ✓/✗ property bar beneath. Row (c) adds the skill-library signal panel + loop.

```
(a) DIVO: random obstacle deployment
    [init T + random obstacles grid] --deploy--> [policy] --0/1--> [success only]
    bar:  ✓ auto env    ✗ unguided    ✗ no difficulty signal

(b) Single-policy feedback curriculum (Eureka / EnvGen style)
    [LLM gen env] --> [policy rollout] --success/regret--> [rewrite]  ⟲
    bar:  ✓ feedback-guided    ✗ 0/1 signal saturates once policy matures

(c) OURS: skill-library learnability loop
    [LLM rewrites generator G_t] --> [Push-T scene] --probe K skills-->
    [Signal Extraction: skill fan (solve/fail) + p(1-p) band] -->
    [paired verifier] --keep in band--> (feedback loop back to G_t)
    bar:  ✓ feedback-guided    ✓ continuous difficulty signal    ✓ evolving
```

Row (c) is the differentiator — give it the most vertical space and the only
color-rich content; keep (a)/(b) lighter/greyer so "ours" pops (CoEvolve does this).

## Real assets (in `figure_outputs/teaser_assets/`, DIVO grammar)

| File | What it is | Use |
|---|---|---|
| `asset_B_skillfan.svg/png` | same layout, **K=6 skill rollouts**, 3 solid (solve) / 3 dashed (fail) | row (c) Signal-Extraction panel (the money asset) |
| `asset_C_learnability.svg/png` | clean `lv=p(1-p)` curve, τ-band, **real point p=0.5** | row (c), right of the fan |
| `asset_A_deployed.svg/png` | same layout, single deployed `w_0` rollout (succeeds) | row (a)/(b) "0/1 success" icon, or a motivation inset |
| `asset_mujoco_scene.png` | real MuJoCo photo — *not generated* (no headless GL) | optional, see bottom |
| `scene_meta.json` | exact layout + per-skill success | provenance |

Real numbers baked in (`best430k.pt`): **realized p = 3/6 = 0.5**, deployed `w_0`
succeeds, τ = 1/(2K) = 0.083. Assets are transparent, axis-free `.svg` + 300-dpi `.png`.
Preview: `figure_outputs/teaser_assets/_contact_sheet.png`.

## Canvas

- Double-column, top of page 1. Width **18.2 cm** (7.16 in). Height **~6–7 cm**
  (3 rows). Row heights ≈ (a) 1.6 cm, (b) 1.6 cm, (c) 2.6 cm.
- Thin dashed dividers between rows (CoEvolve uses dashed grey rules).

## Property bars (✓ green `#2e7d32` / ✗ red `#c0392b`)

Put a light-grey rounded bar under each row with 2–3 tags, e.g.:
- (a) `✗ unguided`  `✗ no difficulty signal`
- (b) `✓ feedback-guided`  `✗ 0/1 saturates at maturity`
- (c) `✓ feedback-guided`  `✓ continuous learnability signal`  `✓ evolving curriculum`

## Palette (hex) — aligned to DIVO

- initial T `#d62728` (red) · goal T `#2ca02c` (green) · obstacles `#ff7f0e` (orange)
- end-effector dot `#111111` · learnable band `#f4b400`
- loop / arrows blue `#1a73e8` · "ours" signal-guided arrow red `#c0392b`
- skill fan = matplotlib **tab10** (already in the SVG) — keep, it distinguishes skills
- row washes: keep (a)/(b) near-white/grey, (c) faint blue `#eef4fb`

## Typography

- Sans (Helvetica / Arial / TeX Gyre Heros). Row titles ~9 pt bold; arrow labels
  ~7 pt; property tags ~7 pt. Math italic (`p`, `lv=p(1-p)`, `w_0`, `G_t`).

## Layer order (bottom → top)

1. row washes + dashed dividers
2. icon-boxes / scene assets / curve
3. arrows (blue loop; red signal-guided) + "Loop ×N" curl on row (c)
4. ✓/✗ property bars, band label, "boundary" callout on the fan
5. row titles + caption

## Caption (ready to use)

> **Fig. 1.** From unguided obstacles to a skill-library learnability curriculum.
> (a) DIVO deploys random obstacles — automatic but unguided, with no per-layout
> difficulty signal. (b) Single-policy feedback curricula rank layouts by 0/1 success,
> which saturates once the policy matures and can no longer tell an already-mastered
> layout from a still-learnable one. (c) We probe each layout with a library of K skills:
> the fraction that solve, `realized p`, is a continuous signal peaking on the learnable
> band `lv = p(1−p)` (here p = 3/6). This drives an LLM generate–verify loop that rewrites
> the obstacle generator and keeps the training distribution on the band, improving
> zero-shot generalization.

## Tooling / workflow

1. New 18.2×6.5 cm artboard; open the `.svg` assets (transparent, axis-free).
2. Lay the 3 rows; drop `asset_B`/`asset_C` into row (c); reuse `asset_A` as the
   "0/1 success" icon in (a)/(b).
3. Draw icon-boxes + labeled arrows for (a)/(b); add the loop + "Loop ×N" on (c).
4. Add grey ✓/✗ property bars under each row.
5. Export vector PDF for submission + 300-dpi PNG for slides.

## To add the real MuJoCo photo later (optional)

Headless GL failed here so `asset_mujoco_scene.png` was not produced. Re-run
`figure_outputs/teaser_assets.py` on a machine with a display or EGL/OSMesa
(`MUJOCO_GL=egl`); the script already enables `env._record_frame` and grabs
`env.frames`, and `scene_meta.json["mujoco_render"]` will flip to `true`. Swap the
photo behind the schematic scene for extra polish.
