# AprilTag field layouts

Verbatim copies of WPILib's published field layouts. Do not edit them: refresh
with the commands below and let the upstream numbers change on their own.

Source: <https://github.com/wpilibsuite/allwpilib>, directory
`fields/src/main/native/resources/org/wpilib/fields/frc/`.

| file | upstream |
| --- | --- |
| `2025-reefscape-welded.json` | `2025-reefscape-welded.json` |
| `2025-reefscape-andymark.json` | `2025-reefscape-andymark.json` |
| `2026-rebuilt-welded.json` | `2026-rebuilt-welded.json` |
| `2026-rebuilt-andymark.json` | `2026-rebuilt-andymark.json` |

Fetched from `main` at commit `4817543253e343afcf2b630537e995d74d21db62`
(2026-10-26).

## Refreshing

```sh
base=https://raw.githubusercontent.com/wpilibsuite/allwpilib/main/fields/src/main/native/resources/org/wpilib/fields/frc
for f in 2025-reefscape-welded 2025-reefscape-andymark \
         2026-rebuilt-welded 2026-rebuilt-andymark; do
  curl -fsS -o "assets/tags/$f.json" "$base/$f.json"
done
```

Then run `cargo test --lib pose::layout` — the bundled-layout tests check the
field size and a known tag position against WPILib's numbers, so a silent
change upstream shows up as a failure.

To use a layout that is not bundled (an offseason field, or a layout a team
maintains themselves), pass it directly:

```sh
camcorder pose --video match.mp4 --tag-layout path/to/layout.json
```

Both WPILib schemas are accepted: the current one (`field-dimensions` and
`field-tags`) and the older `AprilTagFieldLayout` one (`field` and `tags`).
