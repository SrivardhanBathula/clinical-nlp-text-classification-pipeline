# Deployment Guide

## Prerequisites
- Python 3.11+
- CUDA 12.0+ (for GPU inference)
- Docker 24.0+

## Quick Start

```bash
docker-compose up --build
```

API available at `http://localhost:8000`

## Performance
- Inference latency: <120ms (GPU)
- Throughput: 500K+ records/day
- Memory: ~4GB GPU RAM
