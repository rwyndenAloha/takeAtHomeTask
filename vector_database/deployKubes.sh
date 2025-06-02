#!/bin/bash
kubectl apply -f statefulset.yaml
kubectl apply -f headless-service.yaml
kubectl apply -f read-service.yaml
kubectl apply -f rbac.yaml
