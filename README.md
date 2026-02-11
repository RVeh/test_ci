
## 🚀 Direkt starten mit Binder - kann einige Zeit dauern

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RVeh/Hypothesentests/HEAD)

---
Im Menü `Run| Run all Cells` drücken, um die Programme zu starten.

# Hypothesentests – Modell, Setzung, Wirkung

Dieses Projekt enthält vier aufeinander aufbauende Programme
zur Darstellung und Untersuchung von Hypothesentests
am Beispiel des Binomialmodells.

Die Programme folgen einer einheitlichen Struktur:

**Modell → Setzung → (ggf.) Beobachtung**

Ziel ist nicht die rechnerische Durchführung,
sondern das Sichtbarmachen der zugrunde liegenden Struktur.

---

## Inhalte

### 1. Verteilung
Darstellung der Binomialverteilung
in einem σ-basierten Bereich um μ.

### 2. Ablehnungsbereich
Darstellung eines zweiseitigen Ablehnungsbereichs
für eine gegebene Setzung α.

### 3. Spiegelung
Veranschaulichung der Wirkung eines festen Ablehnungsbereichs
bei Variation des Modellparameters.

### 4. Powerfunktion
Darstellung der Powerfunktion
als Wahrscheinlichkeit \(P_p(X \in K)\)
in Abhängigkeit vom wahren Parameter p.

---

## Ausführung

Die Notebooks sind für die Nutzung mit Jupyter / Binder vorbereitet.

Beim ersten Start bitte **Run All Cells** ausführen.
Anschließend können die Parameter angepasst
und einzelne Zellen erneut ausgeführt werden.

---

## Struktur

Die zugrunde liegende Modulstruktur trennt konsequent:

- Modell (Verteilung)
- Geometrie (Ablehnungsbereich)
- Entscheidung (p-Wert)
- Darstellung (Plots)
- Stil (Layout)

Änderungen an Stil oder Setzungen wirken systemweit.
