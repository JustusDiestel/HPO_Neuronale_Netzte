```mermaid
flowchart TD
    A[Start] --> B[Initialisierung einer Population<br/>von NN-Hyperparametern]
    
    B --> C[Fitnessbewertung:<br/>Training des NN<br/>und Accuracy auf Testdaten]
    
    C --> D[Sortierung der Population<br/>nach Fitness]
    
    D --> E[Elitismus:<br/>Bestes Individuum übernehmen]
    
    E --> F[Turnierselektion<br/>zur Elternauswahl]
    
    F --> G[Einpunkt-Crossover<br/>auf Hyperparametern]
    
    G --> H[Mutation einzelner Hyperparameter]
    
    H --> I[Erzeugung einer neuen Population]
    
    I --> J{Maximale Anzahl<br/>an Generationen erreicht?}
    
    J -- Nein --> C
    J -- Ja --> K[Ausgabe der besten<br/>Hyperparameterkonfiguration]
    
    K --> L[Ende]
````