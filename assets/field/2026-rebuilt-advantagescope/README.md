# AdvantageScope 2026 field asset

`model.glb`, `model_0.glb`, and `config.json` are unmodified from
[AdvantageScopeAssets Field3d_2026FRCFieldV2](https://github.com/Mechanical-Advantage/AdvantageScopeAssets/releases/tag/default-assets-v2).
The downloaded archive had SHA-256
`4d5f90415f34d218a75df0a9d6098f331d647949c61196313e5ce5f308122fa4`.
See `LICENSE` for the repository's redistribution terms. The model is a field
representation; its geometric accuracy and rights should be checked against the
upstream asset and official field drawings before relying on it for measurements.

AdvantageScope's configuration specifies a 90-degree rotation about X. The
model is centered on the field. After that rotation, convert its centered
coordinates `(x, y, z)` to the WPILib blue-wall origin as
`(field_length / 2 - x, field_width / 2 - y, z)`. We checked the configuration's
32 AprilTag centres against the bundled WPILib welded layout: the maximum error
is 0.992 mm (the configuration rounds positions to millimetres). The glTF
mesh's ID 16 tag base also lands within 1 mm of its WPILib centre after the
same conversion. For a glTF point `(x, y, z)`, the combined transform is
`(field_length / 2 - x, field_width / 2 + z, y)`. This checks the transform,
not the dimensions of every mesh or the AndyMark field variant.

## Derived geometry

`scripts/field_model.py` turns the mesh into the two files the pose estimator
reads, both in WPILib field coordinates. Regenerate them with:

```sh
.venv/bin/python scripts/field_model.py markings  > assets/field/2026-rebuilt-advantagescope/markings.json
.venv/bin/python scripts/field_model.py structure > assets/field/2026-rebuilt-advantagescope/structure.json
```

* `markings.json` — the long red and blue alliance tape, whose colour identifies
  which field line it is, so a match needs no pose to be identified.
* `structure.json` — a box outline of what each of the 32 tags is mounted on,
  keyed by tag id. A decoded tag already says which structure is in view, so
  these can never be matched to the wrong thing; they are also the only evidence
  that breaks a single small tag's scale ambiguity, because the mount does not
  lie in the tag's plane.

Both are geometry, not observations: the matcher still has to find an image edge
where the model says one is.
