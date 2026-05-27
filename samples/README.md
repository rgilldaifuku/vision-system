# Runtime Simulation Samples

Use this folder for local laptop testing before Raspberry Pi or USB camera hardware is available.

Place test images here:

```text
samples/
  test.jpg
  test_images/
    part_001.jpg
    part_002.jpg
```

Then run the runtime against a single image:

```bash
python -m app.runtime.detector_service --profile yellow_daifuku --camera-source samples/test.jpg --dry-run
```

Or cycle through a folder of images:

```bash
python -m app.runtime.detector_service --profile yellow_daifuku --camera-source samples/test_images/ --dry-run
```

`--dry-run` uses simulated detections and is clearly marked as simulation mode in the dashboard and API. Remove `--dry-run` when testing with a real trained YOLO model.
