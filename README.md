# Network Intrusion Detection

[![Tests](https://github.com/GMCavalheri/network-intrusion-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/GMCavalheri/network-intrusion-detection/actions/workflows/tests.yml)

A distributed network intrusion detection pipeline built for a security data
engineering portfolio: the two standard public IDS benchmarks — **NSL-KDD**
and **CICIDS2017** — flow through a real Spark cluster for ETL and feature
engineering, train Spark MLlib classifiers, get served through FastAPI, and
are visualized in a Streamlit dashboard — all wired together with Docker
Compose.

**Stack:** Apache Spark 3.5 (standalone cluster) · MinIO (S3-compatible storage) ·
PostgreSQL · FastAPI · Streamlit · Docker Compose

Sibling project: [fraud-detection-spark](https://github.com/GMCavalheri/fraud-detection-spark)
(same stack, transaction fraud instead of network intrusions).

> This README is being built up incrementally alongside the code — see the
> commit history for progress. Full architecture diagram, dataset write-up,
> model metrics, and running instructions land as each piece is built.
