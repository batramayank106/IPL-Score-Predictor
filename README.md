# IPL Score Predictor & Results Prediction

A machine learning-based project that predicts cricket match outcomes in the Indian Premier League (IPL). This project uses historical IPL data (2008-2026) to predict first innings scores and second innings win probabilities.

---

## Features

### 1. First Innings Score Prediction
- Predicts the final total score based on:
  - Batting and bowling team
  - Stadium/Venue
  - Current over progress
  - Runs and wickets
  - Recent performance (last 5 overs)

### 2. Second Innings Win Probability
- ML-based win probability prediction for chase scenarios
- Calculates:
  - Required run rate
  - Current run rate
  - Projected final score
  - Runs and balls remaining

---

## Model Performance

### First Innings Score Prediction (Gradient Boosting Regressor)
| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | 19.72 runs |
| Root Mean Squared Error (RMSE) | 25.92 runs |
| Training Data | 179,149 samples |
| Test Data | 13,780 samples |

### Second Innings Win Probability (Random Forest Classifier)
| Metric | Value |
|--------|-------|
| Accuracy | 76.79% |
| Features Used | 9 (target, overs_left, runs_left, wickets_left, run rates, recent form) |

---

## Dataset Statistics
- **Total Records**: 197,222
- **After Cleaning**: 192,929 records
- **Features Encoded**: 81 features (teams + stadiums + numeric)
- **Years Covered**: 2008-2026

---

## Example Predictions

### First Innings Predictions

**Test 1: Mumbai Indians vs Royal Challengers Bangalore at Wankhede Stadium**
```
Input: 10.5 overs, 85 runs/2 wickets, 45 runs in last 5 overs
>>> Predicted Final Score: 103 to 119
```

**Test 2: Chennai Super Kings vs Delhi Capitals at MA Chidambaram Stadium**
```
Input: 13.2 overs, 110 runs/3 wickets, 35 runs in last 5 overs
>>> Predicted Final Score: 93 to 109
```

**Test 3: Sunrisers Hyderabad vs Punjab Kings at Rajiv Gandhi Stadium**
```
Input: 8.4 overs, 72 runs/1 wicket, 50 runs in last 5 overs
>>> Predicted Final Score: 108 to 124
```

**Test 4: Royal Challengers Bangalore vs Chennai Super Kings at M. Chinnaswamy Stadium**
```
Input: 15.0 overs, 145 runs/5 wickets, 40 runs in last 5 overs
>>> Predicted Final Score: 94 to 110
```

### Second Innings Predictions (Win Probability from ML Model)

**Test 5: Kolkata Knight Riders need 175 vs Mumbai Indians**
```
Input: 98/2 in 12.3 overs, target 175
>>> Win Probability: 47.5%
>>> Required Run Rate: 10.0
>>> Current Run Rate: 7.97
>>> Projected Score: 159
>>> Runs Left: 77 in 46 balls
```

**Test 6: Royal Challengers Bangalore need 210 vs Chennai Super Kings (Difficult Chase)**
```
Input: 95/1 in 10.0 overs, target 210
>>> Win Probability: 25.4%
>>> Required Run Rate: 11.50
>>> Current Run Rate: 9.50
```

**Test 7: Punjab Kings need 150 vs Delhi Capitals (Easy Chase)**
```
Input: 120/3 in 14.0 overs, target 150
>>> Win Probability: 98.0%
>>> Required Run Rate: 5.0
>>> Current Run Rate: 8.57
```

---

## Key Visualizations Generated

1. **Correlation Heatmap** (`correlation_heatmap.png`) - Shows feature correlations
2. **Feature Importance** (`feature_importance.png`) - Top 20 predictive features
3. **Actual vs Predicted** (`actual_vs_predicted.png`) - Model performance plots
4. **Year Analysis** (`year_analysis.png`) - Season-wise trends
5. **Residual Analysis** (`residual_analysis.png`) - Error distribution
6. **Model Comparison** (`model_comparison.png`) - MAE/RMSE across models
7. **Distribution Plots** - Score, runs, wickets distributions

---

## Usage

### First Innings Prediction
```python
final_score = predict_first_innings_score(
    batting_team='Mumbai Indians',
    bowling_team='Royal Challengers Bangalore',
    stadium='Wankhede Stadium',
    overs=10.5,
    runs=85,
    wickets=2,
    runs_in_prev_5=45,
    wickets_in_prev_5=1
)
print(f"Predicted Score: {final_score-8} to {final_score+8}")
```

### Second Innings Prediction
```python
result = predict_second_innings(
    target=175,
    batting_team='Kolkata Knight Riders',
    bowling_team='Mumbai Indians',
    overs=12.3,
    runs=98,
    wickets=2,
    runs_in_prev_5=40,
    wickets_in_prev_5=1
)
print(f"Win Probability: {result['win_probability']}%")
print(f"Required Run Rate: {result['required_run_rate']}")
print(f"Projected Score: {result['projected_score']}")
```

---

## Tech Stack

- **Language**: Python
- **Libraries**:
  - pandas, numpy - Data processing
  - scikit-learn - ML models (Gradient Boosting, Random Forest)
  - matplotlib, seaborn - Visualization
- **Data**: IPL match data (2008-2026 seasons)

---

## Project Structure

```
IPL Predictor/
├── ipl.csv                    # Raw dataset (197K+ records)
├── score_predictorandresults.ipynb    # Main analysis notebook
├── ipl_models.pkl            # Saved trained models
├── README.md                 # This file
└── test_notebook.py          # Test script
```

---

## Key Insights from Analysis

1. **Top Predictive Features**: Overs, current runs, wickets, wickets in last 5, runs in last 5
2. **Average IPL Score**: ~165-170 runs (T20 format)
3. **Run Rate Trend**: Increasing over seasons due to T20 evolution
4. **Stadium Impact**: Certain venues (Wankhede, M. Chinnaswamy) are high-scoring

---

## Future Improvements

- Add player-level features (batsman/bowler statistics)
- Incorporate weather and pitch condition data
- Implement deep learning models (LSTM for sequence)
- Create real-time prediction API
- Expand to other T20 leagues (BBL, CPL, etc.)

---

## How to Run

1. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
2. Open `score_predictorandresults.ipynb` in Jupyter/Colab
3. Run all cells to train models
4. Use prediction functions as shown in examples

---

## License

This project is for educational purposes and demonstrates ML applications in sports analytics.

---

*Built with Machine Learning for IPL Analytics*