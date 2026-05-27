# Runtime Profile Rules

Optional runtime inspection rules live under:

```text
profiles/<profile>/config.yaml
```

The runtime still loads model artifacts from `models/<profile>/`. YAML files here only override or extend runtime inspection behavior, such as acceptable classes, reject classes, confidence thresholds, stable-frame counts, ROI, and simulation policy.
