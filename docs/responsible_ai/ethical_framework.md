# EcoSync Responsible AI & Ethical Governance Framework

## 1. Principles & Objectives

EcoSync is designed strictly as a **decision-support platform**, not an autonomous infrastructure controller. Because recommendations influence real-world mechanical equipment, operational costs, and environmental impacts, the system adheres to four core pillars of responsible AI:

```
                  ┌─────────────────────────────────────────┐
                  │       RESPONSIBLE AI GOVERNANCE         │
                  └────────────────────┬────────────────────┘
                                       │
         ┌──────────────────┬──────────┴──────────┬──────────────────┐
         ▼                  ▼                     ▼                  ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   TRANSPARENCY   │ │ ANTI-HALLUCINATE │ │ HUMAN AGENCY     │ │ PRIVACY & SAFETY │
│ - Evidence Chain │ │ - Grounded Facts │ │ - Advisory Only  │ │ - Zero PII       │
│ - Provenance Log │ │ - FACT/HYPOTHESIS│ │ - Engineering V&V│ │ - Aggregate Data │
└──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## 2. The Four Pillars

### 2.1 Transparency & Evidence Provenance
* **Evidence Visibility:** Every AI diagnosis and copilot output displays the exact tool execution trace, including timestamps, building identifiers, sensor readings, and baseline deltas.
* **Knowledge Provenance:** Technical remediation recommendations explicitly cite the authoritative source (e.g., *ASHRAE Standard 90.1-2019 Section 6.4*, *EPA WaterSense Commercial Guidelines*, *UN SDG Target 7.3*).
* **Provider State Disclosure:** The system explicitly states whether responses originate from **IBM Granite 3-3-8B** on watsonx.ai or the deterministic **Local grounded fallback engine**. The interface never misrepresents AI provenance.

### 2.2 Anti-Hallucination & Uncertainty Demarcation
To eliminate confabulation, EcoSync enforces strict output structure:
* **`[FACT]`** — Directly measured sensor data and verifiable historical observations (e.g., *"Building C recorded 31.81 m³ between 02:00 and 04:00"*).
* **`[HYPOTHESIS]`** — Machine learning diagnostic inferences and probable root causes (e.g., *"Statistical pattern suggests possible uncontained flow in commercial flushometer solenoid"*).
* **`[RECOMMENDATION]`** — Suggested human operational actions requiring physical verification (e.g., *"Inspect zone B2 mechanical room riser for valve bypass"*).

### 2.3 Human Agency & Safety
* **No Direct Actuation:** EcoSync cannot transmit control signals, override building management systems (BMS), or adjust physical valves or dampers.
* **Human-in-the-Loop Protocol:** All recommended interventions serve as prioritized triage checklists for licensed facility engineers and sustainability officers. Irreversible operational actions require explicit human authorization.

### 2.4 Data Privacy & Security
* **Zero Personally Identifiable Information (PII):** EcoSync consumes only aggregate physical utility telemetry (kWh, m³, kg) at the building level.
* **No Surveillance Feeds:** The platform does not ingest badge swipe records, facial recognition streams, individual room occupancy cameras, or individual computer telemetry.
* **Secure Cloud Integration:** When watsonx.ai is active, queries transmit only aggregated summary statistics and retrieved markdown guidelines, never proprietary infrastructure schematics.
