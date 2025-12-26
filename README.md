# Airline Passenger Satisfaction Prediction ✈️

## 1. Description of the Problem
[cite_start]This project aims to predict airline passenger satisfaction (Satisfied vs. Neutral/Dissatisfied) based on various service attributes and flight details [cite: 1-14]. Understanding passenger satisfaction is crucial for airlines to minimize churn rates, improve service quality, and optimize operational costs. [cite_start]We approach this as a binary classification problem using advanced machine learning techniques [cite: 15-22].

## 2. Details of the Dataset and Preprocessing
**Dataset:** The complete dataset consists of **129,880 observations**. It is pre-split into a **Training Set of 103,904 entries** and a **Testing Set of 25,976 entries**.
* [cite_start]**Target Distribution:** The dataset is well-balanced with **56.7% Neutral/Dissatisfied** and **43.3% Satisfied** passengers, ensuring no significant class imbalance [cite: 98-105].

![Data Distribution](images/data_distrubition.png)

**Preprocessing Procedures:**
* [cite_start]**Handling Missing Values:** Missing values in the `Arrival Delay in Minutes` column were imputed using the **median** strategy to maintain data integrity [cite: 107-109].
* **Feature Selection:**
    * Irrelevant columns such as `id`, `Unnamed: 0`, and **`Gender`** were dropped to simplify the model.
    * [cite_start]`Departure Delay` was removed due to high multicollinearity (96%) with `Arrival Delay` [cite: 112-113].
* **Encoding:**
    * **Manual Mapping:** Applied to `Class` (Eco:0, Eco Plus:1, Business:2) and `satisfaction` (Neutral/Dissatisfied:0, Satisfied:1) to preserve ordinal meaning.
    * **Label Encoding:** Used for remaining categorical features (`Customer Type`, `Type of Travel`) to convert them into numeric format.

## 3. Model Details and Methodology
[cite_start]We evaluated multiple machine learning algorithms: **Logistic Regression, Random Forest, LightGBM, CatBoost, and XGBoost** [cite: 136-142].

**Experimental Approach (Feature Engineering Hypothesis):**
[cite_start]Before finalizing the model, we implemented a domain-driven feature aggregation strategy to reduce dimensionality [cite: 115-118]. We hypothesized that grouping specific services into macro-categories would simplify the model without losing accuracy.

* **Constructed Features:**
    1.  **Digital Score:** Average of *Inflight Wifi, Online Booking, Online Boarding*.
    2.  **Comfort Score:** Average of *Food & Drink, Seat Comfort, Entertainment, Leg Room, Cleanliness*.
    3.  **Staff Score:** Average of *Gate Location, On-board Service, Check-in, Baggage Handling*.
* [cite_start]**Experiment Result:** The model trained on these 3 composite scores (instead of the original 14 features) saw an accuracy drop from **96.45%** to **~92.00%** [cite: 119-120].
* **Conclusion:** The aggregation caused information loss. The model performs significantly better when it can access granular signals (e.g., distinguishing "Wifi" specifically from general "Digital" interaction). [cite_start]Therefore, the final model uses all original features [cite: 121-124].

**Optimization Strategy:**
[cite_start]We utilized **Optuna (Bayesian Optimization)** to efficiently search the hyperparameter space, maximizing model performance while managing computational resources [cite: 143-145].

## 4. Model Benchmarking & Selection (Performance Analysis)
[cite_start]Following the feedback regarding **Inference Time** during the project presentation, we conducted a comprehensive benchmark measuring Accuracy, Development Cost, and Real-Time Latency [cite: 184-191].

| Model | Accuracy | Optimization Time (Dev Cost) ⏳ | Inference Latency (Per Passenger) ⚡ | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **96.45%** | **~50 s** (Efficient) | **0.0042 ms** (Real-Time) | **🏆 SELECTED** |
| **CatBoost** | 96.12% | ~200 s (High Cost) | **0.0007 ms** (Fastest) | Too slow to optimize |
| **LightGBM** | 96.30% | ~40 s (Fastest) | 0.0047 ms | Lower Accuracy than XGB |
| **Random Forest**| 95.80% | ~60 s | 0.0058 ms | Slowest Inference |
| **Logistic Reg.**| ~87.50% | ~5 s | 0.0005 ms | Underfitting (Low Accuracy) |

### 💡 Why XGBoost? (Discussion on Inference Time)
During our detailed analysis, we observed an important trade-off:
1.  **CatBoost's Latency:** CatBoost achieved the fastest inference speed (**0.0007 ms**) due to its symmetric tree structure.
2.  **The Training Bottleneck:** However, CatBoost's training and optimization process was **4x slower** than XGBoost.
3.  **Final Decision:** We selected **XGBoost** as the production model because it offers the optimal balance. [cite_start]It provides the highest **Accuracy (96.45%)** and a latency of **0.0042 ms**, which is computationally negligible for real-time applications, while being significantly faster to train/retrain than CatBoost [cite: 181-183].

## 5. Model Results & Visualizations

### Feature Importance & SHAP Analysis 🧠
To go beyond simple feature importance, we utilized **SHAP (SHapley Additive exPlanations)** values to understand the *direction* and *magnitude* of each feature's impact on passenger satisfaction.

![SHAP Summary](images/shap_summary.png)

**Key Insights from the Plot:**
* **Top Driver - Type of Travel:** The most influential factor is `Type of Travel`. As seen in the plot, high values (pink/red) strongly correlate with satisfaction, indicating that Business travelers are easier to satisfy or generally happier than Personal travelers.
* **The "Wifi" Factor:** `Inflight wifi service` is the second most critical feature. High values (red dots) pull strongly to the right (positive impact), confirming that connectivity is a major satisfaction driver.
* **Negligible Impact of Catering:** Features like `Food and drink` are located at the very bottom with values clustered around zero. This scientifically proves that catering quality has minimal impact on the final satisfaction decision compared to digital services.

### Confusion Matrix
The model demonstrates balanced performance with high True Positive (10783) and True Negative (14276) rates, ensuring that both satisfied and dissatisfied customers are accurately identified.

![Confusion Matrix](images/confusion_matrix.png)

### Feature Importance (XGBoost Built-in)
Consistent with SHAP analysis, the built-in importance plot highlights **"Online boarding"** as the dominant feature, reinforcing the importance of the digital check-in experience.

![Feature Importance](images/feature_importance.png)

### Training Process (Learning Curve) 📉
The plot below demonstrates the model's learning progress over ~260 iterations.

![Training Curve](images/training_curve.png)

**Analysis of the Curve:**
* **Rapid Convergence:** The model achieves significant loss reduction within the first 50 iterations.
* **No Overfitting:** The **Train Loss (Blue)** and **Test Loss (Orange)** curves descend in parallel and remain close throughout the training process. The absence of a divergence between the two lines confirms that the model generalizes well to unseen data and is not memorizing the training set.

### Example Inference (Real-time Predictions) 🔍
To visualize the model's performance on individual passengers, we tested random samples from the unseen test set.

![Example Inference Results](images/sample_inference.png)

**Analysis of Sample:**
* The table above displays a random sample of 10 predictions.
* The model correctly classified **9 out of 10** passengers.
* **Error Analysis:** As seen in row 0 (Passenger ID 9408), the model predicted "Neutral/Dissatisfied" (0) while the actual status was "Satisfied" (1). Such occasional misclassifications are expected in stochastic models, yet the overall high accuracy (96.45%) remains robust for production use.

## 6. Business Insights & Recommendations
[cite_start]Based on the model's findings (SHAP values & Feature Importance), we propose the following strategies [cite: 267-276]:

1.  **Adopt a "Digital First" Strategy:** Wifi & Online Boarding impact satisfaction more than physical comfort. [cite_start]Airlines should prioritize IT infrastructure investments over seat upgrades [cite: 268-270].
2.  **Focus on "Personal Travelers":** Unlike business travelers, leisure travelers show a higher churn risk. [cite_start]Targeted loyalty campaigns should be launched specifically for personal travel [cite: 271-273].
3.  **Optimize Budget Allocation:** "Food and Drink" has negligible impact on passenger decisions. [cite_start]Catering budgets can be optimized to reallocate funds toward improving connectivity and digital services [cite: 274-276].

## 7. Instructions for Execution

### Prerequisites
Install the required libraries using the provided requirements file:
```bash
pip install -r requirements.txt