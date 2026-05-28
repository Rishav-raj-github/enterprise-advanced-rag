# Acme Corp Cloud Infrastructure Security Guide
**Document ID:** SEC-GUIDE-2025-V2.1  
**Effective Date:** September 1, 2025  
**Owner:** Director of Information Security  

---

## 1. Identity and Access Management (IAM) Policies
Security at Acme Corp starts with strict adherence to the **Principle of Least Privilege (PoLP)**. All access to cloud environments must follow the policies defined herein.

### 1.1 Multi-Factor Authentication (MFA)
MFA is strictly mandatory for all employees, contractors, and third-party integrations accessing any corporate asset.
*   **Hardware Tokens or Authenticator Apps:** SMS-based MFA is strictly prohibited due to SIM-swapping vulnerabilities. Only hardware security keys (e.g., YubiKeys) or registered mobile authenticator apps (e.g., Google Authenticator, Okta Verify) are permitted.
*   **Enforcement:** Access is automatically blocked if MFA registration is not completed within 24 hours of account creation.

### 1.2 IAM Roles and Identity Federation
*   **Okta Integration:** We federate all AWS, GCP, and SaaS access through Okta SSO.
*   **No Long-Lived API Keys:** Long-lived IAM User credentials and programmatic access keys are strictly banned in production environments. Developers and automation scripts must assume short-lived IAM Roles using **AWS STS** (Security Token Service) or GCP Workload Identity Federation.
*   **Session Durations:** Max session duration for console logins is capped at 8 hours. Programmatic CLI session tokens expire after 1 hour.

---

## 2. Data Encryption Standards
All data within Acme Corp's sphere of control must be encrypted both at rest and in transit.

### 2.1 Encryption at Rest
*   **S3 & Cloud Storage:** All Amazon S3 buckets must enforce Default Encryption using AWS KMS managed keys (SSE-KMS). Bucket policies must block any requests that do not specify encryption headers.
*   **Database Encryption:** All database volumes (Aurora, DynamoDB, RDS) must be encrypted using customer-managed keys (CMK) rotated annually.
*   **Key Management Service (KMS):** Cryptographic keys are managed strictly via AWS KMS or HashiCorp Vault. No private keys are to be stored in raw code repositories, configuration files, or docker environments.

### 2.2 Encryption in Transit
*   **TLS Protocol:** All public-facing and internal service-to-service communication must use TLS 1.3 or TLS 1.2 at a minimum. Ciphers utilizing weak keys (e.g., RC4, 3DES) are disabled at the load balancer level.
*   **mTLS (Mutual TLS):** Internal microservices communication within our Kubernetes service mesh (powered by Istio) strictly mandates mutual TLS (mTLS) in STRICT mode.

---

## 3. Threat Detection and Incident Response
*   **Continuous Auditing:** GuardDuty, AWS CloudTrail, and VPC Flow Logs are enabled across all regions. Logs are streamed in real-time to our central Splunk SIEM for threat detection.
*   **Incident Response Procedure:** In the event of a detected credential leak or unauthorized access:
    1.  **Isolation:** Immediately deactivate the affected IAM Role or credentials.
    2.  **Quarantine:** Isolate the associated computing instances or Kubernetes pods by changing their Security Groups/Network Policies.
    3.  **Audit:** Trigger an automated forensic audit of all actions performed by the compromised principal in the preceding 72 hours.
