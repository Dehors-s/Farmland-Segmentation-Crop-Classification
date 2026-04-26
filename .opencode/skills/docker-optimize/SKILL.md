---
name: docker-optimize
description: Optimize Dockerfiles and docker-compose configurations
license: MIT
compatibility: opencode
metadata:
  audience: developers, devops
  workflow: deployment
---

## What I do

- Review Dockerfiles for layer caching, size reduction, and security
- Analyze docker-compose files for service structure and networking
- Suggest multi-stage builds and dependency pinning
- Check for common Docker antipatterns

## Dockerfile checklist

### Layer caching
- Order layers from least to most frequently changing
- Combine `RUN apt-get update && apt-get install` into one layer
- Copy `requirements.txt` before source code for pip cache reuse

### Size reduction
```dockerfile
# Use slim variants
FROM python:3.11-slim

# Multi-stage builds for compiled deps
FROM python:3.11-slim AS builder
RUN pip install --user --no-cache-dir <heavy-deps>

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local

# Clean up temp files in the same RUN layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    <packages> \
    && rm -rf /var/lib/apt/lists/*
```

### Security
- Pin base image digests: `python:3.11-slim@sha256:...`
- Don't run as root — use `USER nobody` or create a dedicated user
- Scan images with `docker scout` or `trivy`

## docker-compose best practices

- Use named volumes instead of bind mounts for production
- Set `restart: unless-stopped` for services
- Use health checks for dependency ordering
- Limit resources with `deploy.resources`

## Usage

```
/load docker-optimize
```

Then: "Review my Dockerfile for a Python web app"
