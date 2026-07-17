# Object-color stress test for LIBERO rollouts

Add an `ObjectColor` stress test alongside `RobotColor` in
`notebooks/stress_test/01_robot_color.ipynb`, plus a scene-inspection helper
so the user can read the names + current RGBAs of every geom in the scene
before authoring a preset.

## Motivation

`RobotColor` already perturbs the robot's color by overwriting
`model.geom_rgba` for geoms matching `robot0_*_vis` / `gripper0_*_visual`.
The natural sibling is a stress test that perturbs the *scene objects'*
colors — both task-relevant objects (e.g. `alphabet_soup_1`) and other
scene geometry (fixtures, table). The blocker is that the user does not
know the exact MuJoCo geom / material names per task and cannot author a
preset blind, so this work also adds a discovery cell.

## Components

All four additions live in §1 of `01_robot_color.ipynb` alongside the
existing `apply_robot_color` / `StressTest` / `RobotColor`. No changes
needed outside that notebook.

### `apply_object_color(env, rgba_by_prefix, *, also_patch_material=True)`

Mirror of `apply_robot_color`. For each `(prefix, rgba)` pair, scans every
geom in `env.sim.model`. If `geom_name == prefix` or
`geom_name.startswith(prefix + '_')`, patches `model.geom_rgba[gid]`.

If `also_patch_material=True` and the matched geom has a material
assigned (`model.geom_matid[gid] >= 0`), patches `model.mat_rgba[matid]`
too. This is needed because LIBERO objects typically use textured
materials, and the renderer prefers material color/texture over
`geom_rgba` — so a plain `geom_rgba` patch is a visual no-op for those
objects.

Side-effect note: `mat_rgba` is per-material, not per-geom. If two geoms
share a material, both get repainted. In LIBERO this is rare (materials
are object-scoped), but the helper docstring documents it.

Returns `dict[prefix -> count]` so the caller can detect typo'd keys
(zero match = no-op, easy to miss otherwise). `ObjectColor.apply_to_env`
prints a `[warn]` for any zero-match key.

### `list_scene_geoms(env, *, include_robot=False, include_arena=False)`

Returns a list of dicts, one per geom, with keys: `owner`, `kind`,
`geom_name`, `geom_group`, `rgba`, `mat_id`, `mat_name`, `mat_rgba`,
`has_texture`.

`owner` resolution (in order):
1. If `geom_name` starts with any key in `env.env.objects_dict` →
   `owner = <object_name>`, `kind = 'task_object'`.
2. If it starts with any key in `env.env.fixtures_dict` →
   `kind = 'fixture'`.
3. If it starts with `robot0_` → `kind = 'robot'`. Hidden unless
   `include_robot=True`.
4. If it starts with `gripper0_` → `kind = 'gripper'`. Hidden with
   the same flag.
5. Otherwise → `kind = 'other'`. Hidden unless `include_arena=True`.

This handles the case where `env.env` isn't the LIBERO problem object
(some wrappers nest deeper) by walking `.env` up to a small depth and
falling back to empty dicts.

### `print_scene_geoms(env, ...)`

Pretty-prints `list_scene_geoms` output grouped by owner. For each owner
shows: geom count (visual / collision split via `geom_group != 3`),
representative `rgba`, material info, and a ⚠ flag when a material is
likely overriding `geom_rgba` (i.e. `mat_id >= 0` and either has a
texture or has a non-default `mat_rgba`).

### `ObjectColor(StressTest)`

```python
@dataclass
class ObjectColor(StressTest):
    rgba_by_geom_prefix: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    also_patch_material: bool = True
    name_hint: str = 'objcolor'
```

- `slug` = `f'{name_hint}_n{len(...)}_{hash6}'`, hash over
  sorted `(prefix, rgba)` items so dict-insertion order doesn't change
  the slug.
- `apply_to_env(env)` calls `apply_object_color`, prints
  per-prefix patched counts, warns on zero matches.
- `manifest()` returns `{'kind': 'ObjectColor', 'slug', 'rgba_by_geom_prefix',
  'also_patch_material'}`.

## Notebook flow change

§3 ("Build env + apply the stress test") splits in two:

- **§3a — Build env + inspect scene.** Builds the env, calls
  `print_scene_geoms(env)`. No stress test applied. Output is the
  reference the user reads to author `ObjectColor` presets.
- **§3b — Apply stress test + sanity render.** Existing reset /
  `set_init_state` / `apply_to_env` / dummy-step / `plt.imshow` flow,
  but the single dummy step becomes a 10-step settling loop. The
  agentview camera observable in robosuite samples at 20 Hz, so a
  single `env.step()` after `apply_to_env` often returns the cached
  pre-perturbation frame. Stepping ten times (matching the rollout's
  `num_steps_wait`) reliably gives a re-sampled, perturbed image.

Iteration loop: run §1 → §2 → §3a, read geom names, edit `PRESETS` in
§1 with the right names + RGBAs, re-run from §2 → §3b to verify.

## Out of scope

- Animated color changes during a rollout (one-shot only, mirrors
  `RobotColor`).
- Per-geom (not per-prefix) granularity. If a user really needs to
  tint one specific `alphabet_soup_1_g3` geom, the existing prefix
  matcher handles it because that full geom name is a valid prefix
  (starts-with semantics + exact match).
- Texture replacement. We only patch RGBA channels of materials; the
  texture remains.

## Testing

Smoke test in a one-off scratch cell:

1. Build the env for `libero_10` task 0.
2. Capture `model.geom_rgba.copy()` and `model.mat_rgba.copy()`.
3. Call `apply_object_color(env, {'alphabet_soup_1': (1.0, 0.0, 0.0, 1.0)})`.
4. Assert at least one row in `geom_rgba` changed for an
   `alphabet_soup_1_*` geom; if `also_patch_material=True` and that
   geom had a material, assert the corresponding `mat_rgba` row changed.
5. Render `obs['agentview_image']` and visually confirm the soup looks
   red.

This is notebook code; the test cell is the "test" and is included as
part of §3a or a small adjacent verification cell rather than as a
pytest file.
