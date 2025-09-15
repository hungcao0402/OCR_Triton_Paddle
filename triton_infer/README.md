# Triton Inference with Monitoring

Docker Compose provides services to run the Triton Inference Server together with Prometheus and Grafana for monitoring.

## Run the services

```bash
docker compose up -d
```

After the containers start, access the dashboards:

- **Prometheus:** <http://localhost:9090>
- **Grafana:** <http://localhost:3000> (default credentials `admin`/`admin`)

In Grafana, add a Prometheus data source at `http://prometheus:9090` and import the "Triton Inference Server" dashboard from the [triton-inference-server/server](https://github.com/triton-inference-server/server/tree/main/qa/metrics) repository.

## Additional information

Metrics are enabled with the `--allow-metrics=true` and `--allow-gpu-metrics=true` flags when starting `tritonserver`.

