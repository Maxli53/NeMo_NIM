# 🚀 Deployment Guide

## Table of Contents
- [Overview](#overview)
- [Deployment Options](#deployment-options)
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Configuration](#production-configuration)
- [Monitoring & Observability](#monitoring--observability)
- [Scaling Strategies](#scaling-strategies)
- [Security Hardening](#security-hardening)
- [Backup & Recovery](#backup--recovery)
- [CI/CD Pipeline](#cicd-pipeline)
- [Troubleshooting](#troubleshooting)

## Overview

The Multi-Agent Discussion System supports multiple deployment strategies:
- **Local**: Development and testing
- **Docker**: Containerized deployment
- **Kubernetes**: Orchestrated scaling
- **Cloud**: AWS, GCP, Azure deployment

Each deployment option is optimized for different use cases and scale requirements.

## Deployment Options

### Deployment Comparison

| Option | Use Case | Scalability | Complexity | Cost |
|--------|----------|-------------|------------|------|
| Local | Development | Low | Low | Free |
| Docker | Single server | Medium | Medium | Low |
| Docker Compose | Multi-service | Medium | Medium | Low |
| Kubernetes | Production | High | High | Medium |
| Cloud (Managed) | Enterprise | Very High | Low | High |
| Cloud (Self-managed) | Enterprise | Very High | Very High | Medium |

## Local Deployment

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourorg/ai-multi-agent.git
cd ai-multi-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models
python scripts/download_models.py

# 5. Initialize databases
python scripts/init_db.py

# 6. Run application
python main.py --mode ui
```

### Systemd Service (Linux)

```ini
# /etc/systemd/system/ai-agent.service

[Unit]
Description=AI Multi-Agent Discussion System
After=network.target

[Service]
Type=simple
User=aiagent
Group=aiagent
WorkingDirectory=/opt/ai-agent
Environment="PATH=/opt/ai-agent/venv/bin"
ExecStart=/opt/ai-agent/venv/bin/python /opt/ai-agent/main.py --mode api
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ai-agent
sudo systemctl start ai-agent
```

### Windows Service

```python
# windows_service.py
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
import os

class AIAgentService(win32serviceutil.ServiceFramework):
    _svc_name_ = "AIMultiAgent"
    _svc_display_name_ = "AI Multi-Agent Discussion System"
    _svc_description_ = "Multi-agent discussion system with RAG"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                            servicemanager.PYS_SERVICE_STARTED,
                            (self._svc_name_, ''))
        self.main()

    def main(self):
        # Run the application
        import main
        main.run_api_mode()

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(AIAgentService)
```

## Docker Deployment

### Single Container

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download models
RUN python scripts/download_models.py

# Expose ports
EXPOSE 8000 8501

# Run application
CMD ["python", "main.py", "--mode", "api"]
```

```bash
# Build and run
docker build -t ai-agent:latest .
docker run -d \
  --name ai-agent \
  -p 8000:8000 \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --gpus all \
  ai-agent:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    container_name: ai-agent-api
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - MODEL_PATH=/models/gpt-oss-20b
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/aiagent
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - redis
      - postgres
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    container_name: ai-agent-ui
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: ai-agent-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: ai-agent-postgres
    environment:
      - POSTGRES_USER=aiagent
      - POSTGRES_PASSWORD=secure_password
      - POSTGRES_DB=aiagent
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: ai-agent-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
      - ui
    restart: unless-stopped

volumes:
  redis-data:
  postgres-data:

networks:
  default:
    name: ai-agent-network
```

### Multi-Stage Build

```dockerfile
# Dockerfile.multistage
# Stage 1: Builder
FROM python:3.10 AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Model downloader
FROM python:3.10-slim AS model-downloader

WORKDIR /models
COPY scripts/download_models.py .
RUN pip install transformers huggingface-hub && \
    python download_models.py

# Stage 3: Final
FROM python:3.10-slim

WORKDIR /app

# Copy wheels and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy models
COPY --from=model-downloader /models /app/models

# Copy application
COPY . .

EXPOSE 8000
CMD ["python", "main.py"]
```

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-agent
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-agent-config
  namespace: ai-agent
data:
  config.yaml: |
    model:
      provider: local
      model_name: gpt-oss-20b
      local:
        device: cuda
        quantization: 8bit
    embedding:
      provider: local
      model_name: all-mpnet-base-v2
    mcp:
      enabled: true
      port: 8000
```

### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-agent-api
  namespace: ai-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-agent-api
  template:
    metadata:
      labels:
        app: ai-agent-api
    spec:
      containers:
      - name: api
        image: ai-agent:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: api
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: models
          mountPath: /app/models
        - name: data
          mountPath: /app/data
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: ai-agent-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: data
        persistentVolumeClaim:
          claimName: data-pvc
```

### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-agent-service
  namespace: ai-agent
spec:
  selector:
    app: ai-agent-api
  ports:
  - port: 8000
    targetPort: 8000
    name: api
  type: ClusterIP
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-agent-ingress
  namespace: ai-agent
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.ai-agent.example.com
    secretName: ai-agent-tls
  rules:
  - host: api.ai-agent.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-agent-service
            port:
              number: 8000
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-agent-hpa
  namespace: ai-agent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-agent-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: pending_requests
      target:
        type: AverageValue
        averageValue: "30"
```

### Persistent Volumes

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: ai-agent
spec:
  accessModes:
    - ReadOnlyMany
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
  namespace: ai-agent
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: standard
  resources:
    requests:
      storage: 50Gi
```

### Helm Chart

```yaml
# helm/Chart.yaml
apiVersion: v2
name: ai-agent
description: AI Multi-Agent Discussion System
type: application
version: 1.0.0
appVersion: "1.0.0"
```

```yaml
# helm/values.yaml
replicaCount: 3

image:
  repository: ai-agent
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.ai-agent.example.com
      paths:
        - path: /
          pathType: ImplementationSpecific
  tls:
    - secretName: ai-agent-tls
      hosts:
        - api.ai-agent.example.com

resources:
  limits:
    cpu: 4000m
    memory: 16Gi
    nvidia.com/gpu: 1
  requests:
    cpu: 2000m
    memory: 8Gi
    nvidia.com/gpu: 1

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

persistence:
  models:
    enabled: true
    storageClass: fast-ssd
    accessMode: ReadOnlyMany
    size: 100Gi
  data:
    enabled: true
    storageClass: standard
    accessMode: ReadWriteMany
    size: 50Gi

redis:
  enabled: true
  architecture: standalone
  auth:
    enabled: true
    password: "secure-password"

postgresql:
  enabled: true
  auth:
    postgresPassword: "secure-password"
    database: "aiagent"
```

## Cloud Deployment

### AWS Deployment

#### EC2 with Auto Scaling

```terraform
# terraform/aws/main.tf
provider "aws" {
  region = "us-west-2"
}

# Launch template
resource "aws_launch_template" "ai_agent" {
  name_prefix   = "ai-agent-"
  image_id      = "ami-0c55b159cbfafe1f0"  # Deep Learning AMI
  instance_type = "g4dn.xlarge"  # GPU instance

  user_data = base64encode(templatefile("userdata.sh", {
    s3_model_bucket = aws_s3_bucket.models.bucket
  }))

  vpc_security_group_ids = [aws_security_group.ai_agent.id]

  iam_instance_profile {
    arn = aws_iam_instance_profile.ai_agent.arn
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = 100
      volume_type = "gp3"
      iops        = 3000
      throughput  = 125
    }
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "ai_agent" {
  name               = "ai-agent-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns  = [aws_lb_target_group.ai_agent.arn]
  health_check_type  = "ELB"
  min_size          = 2
  max_size          = 10
  desired_capacity  = 3

  launch_template {
    id      = aws_launch_template.ai_agent.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "ai-agent"
    propagate_at_launch = true
  }
}

# Application Load Balancer
resource "aws_lb" "ai_agent" {
  name               = "ai-agent-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = aws_subnet.public[*].id

  enable_deletion_protection = true
  enable_http2              = true
}

# Target Group
resource "aws_lb_target_group" "ai_agent" {
  name     = "ai-agent-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout            = 5
    interval           = 30
    path               = "/health"
    matcher            = "200"
  }
}
```

#### ECS Fargate

```terraform
# terraform/aws/ecs.tf
resource "aws_ecs_cluster" "ai_agent" {
  name = "ai-agent-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_task_definition" "ai_agent" {
  family                   = "ai-agent"
  network_mode            = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                     = "4096"
  memory                  = "30720"
  execution_role_arn      = aws_iam_role.ecs_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "ai-agent"
    image = "${aws_ecr_repository.ai_agent.repository_url}:latest"
    
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = [
      {
        name  = "ENVIRONMENT"
        value = "production"
      },
      {
        name  = "MODEL_PATH"
        value = "/models/gpt-oss-20b"
      }
    ]

    mountPoints = [
      {
        sourceVolume  = "models"
        containerPath = "/models"
        readOnly      = true
      }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ai_agent.name
        "awslogs-region"        = "us-west-2"
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])

  volume {
    name = "models"
    efs_volume_configuration {
      file_system_id = aws_efs_file_system.models.id
      root_directory = "/"
    }
  }
}

resource "aws_ecs_service" "ai_agent" {
  name            = "ai-agent-service"
  cluster         = aws_ecs_cluster.ai_agent.id
  task_definition = aws_ecs_task_definition.ai_agent.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [aws_security_group.ecs.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ai_agent.arn
    container_name   = "ai-agent"
    container_port   = 8000
  }
}
```

### GCP Deployment

#### Google Kubernetes Engine

```terraform
# terraform/gcp/gke.tf
provider "google" {
  project = "ai-agent-project"
  region  = "us-central1"
}

resource "google_container_cluster" "ai_agent" {
  name     = "ai-agent-cluster"
  location = "us-central1"

  initial_node_count = 1
  
  node_config {
    preemptible  = false
    machine_type = "n1-standard-4"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  node_pool {
    name       = "gpu-pool"
    node_count = 3

    node_config {
      machine_type = "n1-standard-8"
      
      guest_accelerator {
        type  = "nvidia-tesla-t4"
        count = 1
      }

      disk_size_gb = 100
      disk_type    = "pd-ssd"
    }

    autoscaling {
      min_node_count = 2
      max_node_count = 10
    }
  }
}
```

### Azure Deployment

#### Azure Container Instances

```terraform
# terraform/azure/aci.tf
provider "azurerm" {
  features {}
}

resource "azurerm_container_group" "ai_agent" {
  name                = "ai-agent"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  ip_address_type     = "Public"
  dns_name_label      = "ai-agent"
  os_type            = "Linux"

  container {
    name   = "ai-agent-api"
    image  = "aiagent.azurecr.io/ai-agent:latest"
    cpu    = "4"
    memory = "16"
    gpu {
      count = 1
      sku   = "K80"
    }

    ports {
      port     = 8000
      protocol = "TCP"
    }

    environment_variables = {
      ENVIRONMENT = "production"
    }

    volume {
      name       = "models"
      mount_path = "/models"
      read_only  = true

      storage_account_name = azurerm_storage_account.models.name
      storage_account_key  = azurerm_storage_account.models.primary_access_key
      share_name          = azurerm_storage_share.models.name
    }
  }
}
```

## Production Configuration

### Environment Setup

```bash
# production.env
ENVIRONMENT=production
LOG_LEVEL=WARNING
DEBUG=false

# Model configuration
MODEL_PROVIDER=local
MODEL_PATH=/models/gpt-oss-20b
MODEL_DEVICE=cuda
MODEL_QUANTIZATION=8bit

# Database
REDIS_URL=redis://redis.internal:6379
POSTGRES_URL=postgresql://user:pass@postgres.internal:5432/aiagent

# Security
API_KEY_REQUIRED=true
CORS_ORIGINS=https://app.example.com
RATE_LIMIT_ENABLED=true

# Performance
MAX_WORKERS=8
CACHE_ENABLED=true
MIXED_PRECISION=true
```

### SSL/TLS Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.ai-agent.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    location / {
        proxy_pass http://ai-agent-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Monitoring & Observability

### Prometheus Metrics

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics
request_count = Counter('ai_agent_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('ai_agent_request_duration_seconds', 'Request duration')
active_sessions = Gauge('ai_agent_active_sessions', 'Number of active sessions')
model_inference_time = Histogram('ai_agent_model_inference_seconds', 'Model inference time')
gpu_utilization = Gauge('ai_agent_gpu_utilization_percent', 'GPU utilization')
memory_usage = Gauge('ai_agent_memory_usage_bytes', 'Memory usage')

@app.route('/metrics')
def metrics():
    return generate_latest()
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "AI Agent Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(ai_agent_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, ai_agent_request_duration_seconds)"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "ai_agent_gpu_utilization_percent"
          }
        ]
      },
      {
        "title": "Active Sessions",
        "targets": [
          {
            "expr": "ai_agent_active_sessions"
          }
        ]
      }
    ]
  }
}
```

### Logging Stack

```yaml
# docker-compose.logging.yml
services:
  elasticsearch:
    image: elasticsearch:8.10.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - es-data:/usr/share/elasticsearch/data

  logstash:
    image: logstash:8.10.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:8.10.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Scaling Strategies

### Horizontal Scaling

```python
# Load balancer configuration
from typing import List
import random

class LoadBalancer:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.current = 0
    
    def round_robin(self) -> str:
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server
    
    def least_connections(self, connections: Dict[str, int]) -> str:
        return min(connections, key=connections.get)
    
    def weighted_round_robin(self, weights: Dict[str, int]) -> str:
        servers = []
        for server, weight in weights.items():
            servers.extend([server] * weight)
        return random.choice(servers)
```

### Vertical Scaling

```bash
# GPU scaling script
#!/bin/bash

# Check current GPU utilization
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

if [ $GPU_UTIL -gt 80 ]; then
    echo "High GPU utilization detected. Scaling up..."
    # Add more GPU resources or migrate to larger instance
    aws ec2 modify-instance-attribute --instance-id i-1234567890abcdef0 \
        --instance-type g4dn.2xlarge
fi
```

## Security Hardening

### Security Checklist

```yaml
# security-checklist.yaml
security:
  network:
    - Use private subnets for application servers
    - Configure security groups with minimal access
    - Enable VPC flow logs
    - Use WAF for API protection
    
  authentication:
    - Implement JWT/OAuth2 authentication
    - Use API keys for service-to-service communication
    - Enable MFA for admin access
    - Rotate credentials regularly
    
  encryption:
    - Enable TLS 1.2+ for all communications
    - Encrypt data at rest
    - Use KMS for key management
    - Implement field-level encryption for sensitive data
    
  compliance:
    - Enable audit logging
    - Implement data retention policies
    - Configure backup encryption
    - Regular security scans
```

### Security Headers

```python
# Security middleware
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.cors import CORSMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*.ai-agent.example.com"]
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response
```

## Backup & Recovery

### Backup Strategy

```bash
# backup.sh
#!/bin/bash

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup databases
pg_dump $POSTGRES_URL > $BACKUP_DIR/postgres.sql
redis-cli --rdb $BACKUP_DIR/redis.rdb

# Backup vector indices
cp -r /data/indices $BACKUP_DIR/

# Backup configurations
cp -r /app/config $BACKUP_DIR/

# Compress and upload to S3
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
aws s3 cp $BACKUP_DIR.tar.gz s3://ai-agent-backups/

# Clean up old backups
find /backups -type d -mtime +7 -exec rm -rf {} +
```

### Disaster Recovery

```python
# disaster_recovery.py
import subprocess
import logging
from typing import Optional

class DisasterRecovery:
    def __init__(self, backup_location: str):
        self.backup_location = backup_location
        self.logger = logging.getLogger(__name__)
    
    def restore_from_backup(self, backup_id: str) -> bool:
        """Restore system from backup"""
        try:
            # Download backup
            subprocess.run([
                "aws", "s3", "cp",
                f"s3://ai-agent-backups/{backup_id}.tar.gz",
                "/tmp/restore.tar.gz"
            ], check=True)
            
            # Extract backup
            subprocess.run([
                "tar", "-xzf", "/tmp/restore.tar.gz", "-C", "/"
            ], check=True)
            
            # Restore databases
            subprocess.run([
                "psql", self.postgres_url, "<", f"/backups/{backup_id}/postgres.sql"
            ], shell=True, check=True)
            
            # Restore Redis
            subprocess.run([
                "redis-cli", "--rdb", f"/backups/{backup_id}/redis.rdb"
            ], check=True)
            
            self.logger.info(f"Successfully restored from backup {backup_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
```

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            aiagent/ai-agent:latest
            aiagent/ai-agent:${{ github.sha }}
          cache-from: type=registry,ref=aiagent/ai-agent:buildcache
          cache-to: type=registry,ref=aiagent/ai-agent:buildcache,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        uses: azure/k8s-deploy@v4
        with:
          manifests: |
            k8s/deployment.yaml
            k8s/service.yaml
          images: |
            aiagent/ai-agent:${{ github.sha }}
          namespace: production
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: python:3.10
  script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
    - pytest tests/ --cov=src
    - pylint src/
    - mypy src/
  coverage: '/TOTAL.*\s+(\d+%)$/'

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest
    - docker push $CI_REGISTRY_IMAGE:latest

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/ai-agent ai-agent=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - kubectl rollout status deployment/ai-agent
  only:
    - main
```

## Troubleshooting

### Common Deployment Issues

1. **Out of Memory**
   ```bash
   # Increase memory limits
   docker run -m 32g ai-agent:latest
   # Or in Kubernetes
   kubectl set resources deployment ai-agent --limits=memory=32Gi
   ```

2. **GPU Not Available**
   ```bash
   # Check GPU availability
   nvidia-smi
   # Install NVIDIA drivers
   sudo apt-get install nvidia-driver-470
   # Install nvidia-docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   ```

3. **Model Loading Failures**
   ```python
   # Check model files
   import os
   model_path = "/models/gpt-oss-20b"
   if not os.path.exists(model_path):
       print("Model not found. Downloading...")
       from scripts.download_models import download_gpt_oss
       download_gpt_oss(model_path)
   ```

4. **Connection Timeouts**
   ```yaml
   # Increase timeouts in nginx
   proxy_connect_timeout 300;
   proxy_send_timeout 300;
   proxy_read_timeout 300;
   ```

5. **SSL Certificate Issues**
   ```bash
   # Generate self-signed certificate for testing
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout /etc/ssl/private/ai-agent.key \
     -out /etc/ssl/certs/ai-agent.crt
   ```

### Health Check Script

```python
# healthcheck.py
import requests
import sys

def check_health():
    checks = {
        "API": "http://localhost:8000/health",
        "UI": "http://localhost:8501",
        "Redis": "redis://localhost:6379",
        "Postgres": "postgresql://localhost:5432"
    }
    
    for service, url in checks.items():
        try:
            if service in ["API", "UI"]:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    print(f"✓ {service}: OK")
                else:
                    print(f"✗ {service}: Failed (HTTP {resp.status_code})")
                    sys.exit(1)
            # Add Redis/Postgres checks
        except Exception as e:
            print(f"✗ {service}: Failed ({e})")
            sys.exit(1)
    
    print("All health checks passed!")

if __name__ == "__main__":
    check_health()
```

---

For more deployment examples and configurations, see the [examples/deployments](../examples/deployments) directory.