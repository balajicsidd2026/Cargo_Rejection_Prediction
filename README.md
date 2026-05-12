Cargo Acceptance Rejection Forecasting:

1.Overview:
     The Cargo Acceptance Rejection Forecasting system is designed to predict the likelihood of cargo shipment rejection during acceptance operations using factors such as documentation status, packaging condition, and shipper compliance history. This Model is designed to predict whether a cargo shipment will be:
• Accepted
OR
• Rejected
during cargo acceptance operations.




2. Dataset Specification:
      Total Records: 22,000
      Total Features: 21 Features + 1 Target Column
   Data Split:
      Training: 70% (15400)
      Validation: 15% (3300)
      Testing: 15% (3300)



3. Model Selection:
	Multiple machine learning classification algorithms were implemented and compared to identify the best-performing model for cargo rejection prediction.
Implemented Models:
        • Logistic Regression
	• Decision Tree Classifier
	• Random Forest Classifier
	• K-Nearest Neighbors (KNN)
	• XGBoost Classifier
	• LightGBM Classifier
	• CatBoost Classifier
Final Selected Model:CatBoost Classifier
Reason for Selection:
• Achieved the best balance between Recall and F1-score
• Successfully detected high-risk rejected cargo shipments




4.Metrics for my model:
 4.1CatBoost Performance Metrics:
    • Accuracy  : 0.90
    • Precision : 0.54
    • Recall    : 0.84
    • F1-Score  : 0.66

 4.2Confusion Matrix:
    • True Positive  : 305
    • True Negative  : 2681
    • False Positive : 258
    • False Negative : 57



5. Application & Deployment :
	The final CatBoost model was deployed using Streamlit to create an interactive web application for real-time cargo rejection prediction. 
	The application allows users to enter shipment details such as packaging condition, documentation status, X-ray scan result, compliance violations, and shipper reliability information through a user-friendly interface. 
	Based on the provided operational inputs, the system predicts whether the cargo shipment is likely to be Accepted or Rejected and displays the rejection probability score. 
	The final CatBoost model was deployed using Streamlit Community Cloud (streamlit.io) to create an interactive web application for real-time cargo rejection prediction.
	





