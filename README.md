# Prompt Stability Evaluation System

Ein modulares System zur Evaluierung der Prompstabilität von LLM-Modellen mittels Monte-Carlo-Simulation.

## Struktur

Das System besteht aus mehreren modularen Komponenten:

### Module

- **`uuak.py`**: Hauptprogramm zum Ausführen von Tests mit verschiedenen Stilkombinationen
- **`data_storage.py`**: Abstrakte und konkrete Implementierungen für Datenspeicherung (JSON)
- **`stability_calculator.py`**: Berechnung der Prompstabilität mit verschiedenen Ähnlichkeitsmetriken
- **`evaluator.py`**: Orchestrierung der Datenauswertung und Berichtgenerierung
- **`evaluate.py`**: Kommandozeilen-Skript zur Auswertung gespeicherter Daten

## Verwendung

### 1. Tests ausführen und Daten sammeln

```bash
python uuak.py
```

Dies führt Tests mit verschiedenen Prompts aus und speichert die Ergebnisse automatisch in `test_data/test_results.json`.

### 2. Daten auswerten

```bash
# Vollständiger Bericht
python evaluate.py

# Nur Zusammenfassung
python evaluate.py --summary-only

# Spezifische Anzahl von Monte-Carlo-Iterationen
python evaluate.py --iterations 5000

# Andere Ähnlichkeitsmetrik verwenden
python evaluate.py --metric length

# Bericht in Datei speichern
python evaluate.py --output report.txt

# Nur einen spezifischen Prompt auswerten
python evaluate.py --prompt "Explain the theory of relativity."
```

## Architektur

### Abstraktionsebenen

1. **Storage Layer** (`data_storage.py`):
   - `DataStorage`: Abstrakte Basisklasse
   - `JSONDataStorage`: Konkrete JSON-Implementierung

2. **Calculation Layer** (`stability_calculator.py`):
   - `SimilarityMetric`: Abstrakte Basisklasse für Ähnlichkeitsmetriken
   - `JaccardSimilarity`, `LengthSimilarity`: Konkrete Metriken
   - `StabilityCalculator`: Hauptklasse für Stabilitätsberechnungen

3. **Evaluation Layer** (`evaluator.py`):
   - `TestResultAggregator`: Aggregiert Testdaten
   - `Evaluator`: Orchestriert Auswertung und Berichtgenerierung

## Prompstabilität

Die Prompstabilität misst, wie konsistent die Antworten eines Modells bei verschiedenen Stilvorgaben sind. 

- **Hohe Stabilität** (nahe 1.0): Das Modell gibt ähnliche Antworten trotz unterschiedlicher Stilvorgaben
- **Niedrige Stabilität** (nahe 0.0): Das Modell variiert stark in seinen Antworten

Die Berechnung erfolgt mittels Monte-Carlo-Simulation, die viele zufällige Stichproben der Antworten nimmt und die durchschnittliche Ähnlichkeit berechnet.

## Abhängigkeiten

- `ollama`: Für LLM-Interaktion
- `nltk`: Für Sentiment-Analyse
- `numpy`: Für statistische Berechnungen

