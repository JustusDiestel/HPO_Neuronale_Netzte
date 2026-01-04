
### Ergebnisse der Hyperparameter-Optimierung

| Parameter | GA | Optuna |
|---|---:|---:|
| Accuracy | 0.9864 | 0.9864 |
| Aktivierung | relu | elu |
| Basis\_Einheiten | 32 | 128 |
| Batch\-Größe | 256 | 16 |
| Dropout\-Rate | 9.20e\-3 | 1.11e\-1 |
| Epochen | 66 | 76 |
| L2\-Gewichtsverfall | 9.46e\-4 | 1.19e\-3 |
| Lernrate | 2.94e\-3 | 1.77e\-4 |
| Schichten | 3 | 4 |
| Optimierer | adam | adam |
| Skaler | minmax | none |
| Breitenmuster | increasing | constant |

*Tabelle:* Hyperparameter und Testergebnisse für die beiden Optimierungsverfahren.