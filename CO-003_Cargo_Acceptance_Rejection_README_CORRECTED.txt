Cargo Acceptance Rejection Forecasting Dataset

Overview:
This dataset contains enterprise-style airline cargo acceptance operation records designed for machine learning, logistics analytics, and operational risk prediction.
The dataset simulates real-world cargo acceptance workflows followed by airline cargo terminals and freight
forwarding operations. 
It includes shipment screening, compliance verification, X-ray inspection outcomes, documentation checks, packaging validation, and shipper reliability information.
The dataset is designed specifically for predictive analytics and classification modeling in airline cargo operations.

Total Records: 22,000
Total Features: 21 Features + 1 Target Column


Data Split:
Training: 70% (16000)
Validation: 15% (3300)
Testing: 15% (3300)


Primary Use Case: Cargo Acceptance Rejection Prediction
This dataset is designed to support ML models that predict whether a cargo shipment will be:
• Accepted
OR
• Rejected
during cargo acceptance operations.


Target Variable:
acceptance_rejection_flag
0 = Accepted
1 = Rejected


Business Objective:
The goal is to help cargo operators:
• reduce acceptance delays
• detect risky shipments early
• improve compliance verification
• automate cargo screening decisions
• reduce dangerous goods violations
• improve operational efficiency


Dataset Characteristics:
Domain: Airline Cargo Operations
ML Type: Classification
Dataset Type: Tabular Structured Data
Records: 22,000
Features: 22
Numerical Features: 7
Categorical Features: 14
Target Columns: 1
Missing Values: No
Split Type: Train / Validation / Test



Feature Schema & Key Columns
1. Shipment & Routing Information
• record_id — Unique shipment acceptance ID
• timestamp — Cargo acceptance date and time
• airline — Airline handling the shipment
• origin — Shipment origin station
• destination — Shipment destination station


2. Cargo Handling & SHC Information
• SHC — Special Handling Code
• shipment_priority — Shipment service priority
• cargo_weight_kg — Cargo shipment weight
• packaging_condition — Cargo packaging quality
• packaging_seal_status — Seal integrity status


3. Security & Compliance Features
• documentation_status — Shipment documentation condition
• security_screening_status — Security clearance result
• xray_scan_result — X-ray screening outcome
• compliance_violation_count — Previous compliance issues
• inspection_duration_minutes — Inspection processing duration
• damage_history_count — Historical cargo damage count


4. Shipper & Customer Information
• shipper_company_name — Company sending cargo
• consignee_name — Cargo receiving company
• shipper_type — Type of shipper
• shipper_reliability_score — Historical reliability rating


5. Environmental & Operational Features
• weather_condition — Operational weather status


Machine Learning Logic & Embedded Patterns
High Rejection Probability Conditions
• Missing documentation
• Failed security screening
• Suspicious X-ray results
• Tampered packaging seals
• High compliance violations
• Low shipper reliability
• Dangerous cargo SHC categories


Recommended ML Algorithms
• CatBoost — Highly Recommended
• XGBoost — Recommended
• LightGBM — Recommended
• Random Forest — Good Baseline
• Logistic Regression — Baseline Model



Recommended Evaluation Metrics
• F1-Score — High Importance
• Recall — High Importance
• Precision — High Importance
• ROC-AUC — Recommended
• Accuracy — Secondary



Important Feature:

documentation_status	
security_screening_status
xray_scan_result	
packaging_seal_status	
compliance_violation_count	
SHC	
shipper_reliability_score	
packaging_condition



