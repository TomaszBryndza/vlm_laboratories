# Multi-View Route Sequence Samples

Place your paired images here for the route planning experiment.

Naming convention (exact):
- frame01_cam.png
- frame01_map.png
- frame02_cam.png
- frame02_map.png
- frame03_cam.png
- frame03_map.png
- frame04_cam.png
- frame04_map.png
- frame05_cam.png
- frame05_map.png

Optional additional frames: continue numbering (frame06_cam.png, etc.).

You may also include a `route_example.json` copy if sequence differs from the default in `data/`.

## Optional Annotation Files
For each frame you can add a textual descriptor to guide retrieval:
- frame01_context.txt (free-form notes, keywords)

## Usage in script
The sequence experiment script will scan this folder, pair `*_cam.png` with corresponding `*_map.png` (same prefix), sort by frame number, and build a composite prompt referencing all frames.

Images should be reasonably small (<=512px on each side) to keep inference latency manageable.
