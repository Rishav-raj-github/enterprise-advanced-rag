# Acme Corp Platform Systems Architecture Specification
**Document ID:** ARCH-SPEC-2026-V1.0  
**Effective Date:** March 15, 2026  
**Author:** Principal Software Architect  

---

## 1. Microservices Ecosystem & Gateway Routing
The Acme platform is engineered on a highly-available, distributed microservices architecture deployed across multiple Kubernetes clusters in AWS and GCP.

```
                  ┌──────────────────────┐
                  │   Client Applications│
                  └──────────┬───────────┘
                             │ HTTPS / WSS
                             ▼
                  ┌──────────────────────┐
                  │ Kong API Gateway     │
                  └──────────┬───────────┘
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │Auth Service │  │Order Service│  │Inventory Svc│
     └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
            │                │                │
            ▼                ▼                ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │Redis Cache  │  │PostgreSQL DB│  │DynamoDB     │
     └─────────────┘  └─────────────┘  └─────────────┘
```

### 1.1 Ingress Control and Kong Gateway
All client traffic routes through a globally load-balanced **Kong API Gateway**. Kong is responsible for:
*   **JWT Authentication & Rate Limiting:** Enforces a tier-based rate limit:
    *   *Basic Tier:* Max 60 requests/minute.
    *   *Enterprise Tier:* Max 10,000 requests/minute.
*   **SSL Termination & CORS Policies:** Standardizes SSL cipher suites to TLS 1.3 only.
*   **Request Routing:** Dynamic routing based on URI prefixes (e.g., `/api/v1/orders/*` routes to the Order Service).

---

## 2. Database Tier & Distributed Caching
To maintain consistency and high throughput, the data layer utilizes a polyglot persistence model.

### 2.1 Core Relational Database (PostgreSQL Cluster)
The system of record for critical transactional transactions (orders, users, payments) is a multi-region **Amazon Aurora PostgreSQL Serverless v2** cluster.
*   **Replication Topology:** Single-writer instance in US-East-1 with two read-replicas spanning US-West-2 and EU-Central-1.
*   **Backup Strategy:** Automated continuous backups with a 35-day retention window. Cross-region snapshot copying is triggered daily at 01:00 UTC.

### 2.2 In-Memory Caching (Redis Enterprise Cluster)
To reduce DB read pressure, a multi-node **Redis Enterprise** cluster is positioned in front of the database.
*   **Cache Eviction Policy:** Volatile-LRU (Least Recently Used) with standard TTLs set to:
    *   *User Session Profiles:* 24 Hours.
    *   *Product Catalog Indices:* 1 Hour.
*   **Cache Synchronization:** Handled via write-through or cache-aside strategies depending on write frequency.

---

## 3. Asynchronous Messaging via Apache Kafka
Services communicate asynchronously using an enterprise **Apache Kafka** cluster hosted on Confluent Cloud.
*   **Primary Topics:**
    *   `orders.created`: Dispatched when a client completes a checkout flow.
    *   `inventory.allocated`: Emitted when warehouse stock is successfully locked.
    *   `payment.processed`: Dispatched when the Stripe webhook verifies invoice payment.
*   **Guarantees:** Producer idempotency is enabled (`enable.idempotence=true`), ensuring **Exactly-Once** delivery semantics.
