¿Qué predice la infidelidad? Un análisis estadístico y de Machine Learning

Análisis exploratorio y modelado predictivo sobre el dataset Fair's Affairs (Ray Fair, 1978), combinando pruebas estadísticas clásicas, ingeniería de variables y comparación de modelos de clasificación, con interpretabilidad vía SHAP.


Proyecto realizado con fines de práctica y portafolio. Los datos son autoreportes de encuestas de los años 60-70 (revistas Psychology Today y Redbook), por lo que están sujetos a sesgos de selección y deseabilidad social — no se interpretan como verdad poblacional, sino como ejercicio de análisis riguroso.



Objetivo

Identificar qué variables sociodemográficas y de relación se asocian con la infidelidad marital, y evaluar qué tan bien se puede predecir usando distintos enfoques de modelado, priorizando siempre el rigor estadístico sobre resultados superficiales.

Metodología

1. EDA univariado y bivariado


Identificación de variables discretas disfrazadas de continuas (age, yrs_married, children están en bins, no son mediciones exactas).
Pruebas de asociación según el tipo real de cada variable: Cramér's V para categóricas, Mann-Whitney U y Spearman para ordinales, Cochran-Armitage para tendencias monotónicas.
Todas las pruebas acompañadas de effect size, no solo p-value — con ~6,366 observaciones, casi cualquier asociación resulta "significativa", así que la magnitud del efecto es lo que realmente importa.


2. Hallazgo principal: efecto de supresión (paradoja de Simpson)
age muestra correlación positiva con infidelidad de forma aislada, pero se invierte a negativa al controlar por yrs_married en un modelo multivariado — ambas variables comparten ~0.89 de correlación entre sí (confirmado con matriz de correlación y análisis de colinealidad). Este hallazgo guió directamente la ingeniería de variables.

3. Feature engineering


edad_al_casarse = age − yrs_married (resuelve la colinealidad de forma interpretable)
hijos_por_ano = children / (yrs_married + 1)
satisfaccion_x_religiosidad = rate_marriage × religious (interacción con justificación teórica)


4. Modelado


XGBoost, Logistic Regression y SVM (kernel RBF), cada uno optimizado con Optuna (200-250 trials), incluyendo búsqueda de balanceo de clases (scale_pos_weight / class_weight) para cada modelo por igual.
Métrica principal: F1-score (no accuracy, que es engañoso dado el desbalance de clases ~68/32).
Validado contra un DummyClassifier como baseline, para confirmar que el rendimiento no es producto del desbalance de clases.


5. Interpretabilidad


SHAP (TreeExplainer, salida en probabilidad) sobre el modelo XGBoost final, para identificar qué variables impulsan cada predicción individual y el patrón agregado del modelo.


Resultados

<img width="365" height="97" alt="{E645C908-95CE-433A-95C8-43015C8880AB}" src="https://github.com/user-attachments/assets/39e09da7-2237-428c-81b4-46229377ecc6" />


Los tres modelos convergen prácticamente al mismo F1, a pesar de tener arquitecturas muy distintas (árboles, lineal, kernel no lineal). Esto es evidencia de que la relación entre las variables predictoras y la infidelidad es mayormente lineal/aditiva — no hay ganancia relevante de usar un modelo más complejo una vez que el feature engineering captura la estructura relevante de los datos. Es un hallazgo más valioso que si un solo modelo hubiera "ganado", porque demuestra dónde reside realmente el techo de rendimiento: en la señal disponible en los datos, no en la capacidad del algoritmo.

Variables más importantes según SHAP: satisfaccion_x_religiosidad, rate_marriage, hijos_por_ano y edad_al_casarse — tres de las cuatro son variables construidas específicamente para resolver la colinealidad detectada en el EDA, confirmando que absorbieron la señal que antes estaba dispersa entre age, yrs_married y children.

Stack técnico

pandas · numpy · scipy · statsmodels · scikit-learn · XGBoost · Optuna · SHAP · seaborn / matplotlib
