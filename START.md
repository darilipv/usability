# Schnellstart-Anleitung

## Schritt 1: Abhängigkeiten installieren

Stelle sicher, dass folgende Python-Pakete installiert sind:

```bash
pip install ollama nltk numpy
```

Falls `ollama` nicht über pip installiert werden kann, installiere es separat:
- Siehe: https://github.com/ollama/ollama

## Schritt 2: Tests ausführen und Daten sammeln

### Verfügbare Modelle anzeigen

Zuerst kannst du sehen, welche Modelle verfügbar sind:

```bash
python uuak.py --list-models
```

### Tests ausführen

**Standard (verwendet automatisch alle verfügbaren Modelle)**
```bash
python uuak.py
```
Dies ist die einfachste Option - das Programm erkennt automatisch alle installierten Ollama-Modelle und testet diese.

**Spezifische Modelle auswählen**
```bash
python uuak.py --models qwen2.5:0.5b qwen2.5:3b llama3.2:1b
```
Wenn du nur bestimmte Modelle testen möchtest, kannst du sie explizit angeben.

**Was passiert hier?**
- Das Programm führt Tests mit verschiedenen Prompts durch
- Für jeden Prompt wird eine zufällige Stilkombination generiert
- Jeder Agent (Modell) antwortet mehrmals auf den gleichen Prompt
- Alle Ergebnisse werden automatisch in `test_data/test_results.json` gespeichert

**Hinweis:** 
- Dies kann einige Zeit dauern, da echte LLM-Antworten generiert werden
- Je mehr Modelle du testest, desto länger dauert es
- Das Programm verwendet standardmäßig **alle verfügbaren Ollama-Modelle** automatisch

## Schritt 3: Daten auswerten

Nachdem die Tests abgeschlossen sind, führe die Auswertung aus:

```bash
python evaluate.py
```

**Was passiert hier?**
- Die gespeicherten Testdaten werden geladen
- Für jeden Prompt und jeden Agent wird die Prompstabilität berechnet
- Eine Monte-Carlo-Simulation wird durchgeführt (standardmäßig 1000 Iterationen)
- Ein detaillierter Bericht wird ausgegeben

## Optionen für die Auswertung

```bash
# Nur Zusammenfassung anzeigen
python evaluate.py --summary-only

# Mehr Iterationen für genauere Ergebnisse
python evaluate.py --iterations 5000

# Andere Ähnlichkeitsmetrik verwenden
python evaluate.py --metric length

# Bericht in Datei speichern
python evaluate.py --output report.txt

# Nur einen spezifischen Prompt auswerten
python evaluate.py --prompt "Explain the theory of relativity."
```

## Beispiel-Workflow

```bash
# 1. Tests ausführen (dauert einige Minuten)
python uuak.py

# 2. Auswertung mit Standardeinstellungen
python evaluate.py

# 3. Oder: Detaillierter Bericht in Datei
python evaluate.py --iterations 2000 --output stability_report.txt
```

## Troubleshooting

**Problem: "No test data found"**
- Lösung: Führe zuerst `python uuak.py` aus, um Daten zu generieren

**Problem: "ollama" Modul nicht gefunden**
- Lösung: Installiere ollama: `pip install ollama` oder siehe ollama-Dokumentation

**Problem: "nltk" Fehler**
- Lösung: Das Programm lädt automatisch die benötigten NLTK-Daten beim ersten Start

