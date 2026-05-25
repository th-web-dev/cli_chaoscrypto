To-Do Guide für Codex: Analyse & Evaluation von ChaosCrypto (BA2)

1\. Projektziel und Zweck

Das Ziel ist die wissenschaftliche Bewertung des in BA1 entwickelten Prototyps hinsichtlich **Security, Performance und Usability**24. Wir betrachten das System als Forschungsplattform4. Es geht nicht darum, Schwachstellen sofort zu beheben, sondern sie durch gezielte Testszenerien messbar und dokumentierbar zu machen3.

\--------------------------------------------------------------------------------

2\. Erhebung der Performance-Basis (Hauptteil II)

**Zweck:** Präzise Messung der Effizienz unter variierenden Bedingungen, um die Praxistauglichkeit zu bestimmen15.

*   **Task 2.1: Ressourcen-Split im Profiling**
    *   **Ziel:** Trennung der Zeitmessung für (1) Noise-Feld-Generierung, (2) Seed-Ableitung, (3) Keystream-Integration und (4) XOR-Verarbeitung56.*   **Anleitung:** Modifiziere die `benchmark`\-Logik, um diese Phasen einzeln in die `results.csv` zu schreiben78.*   **Task 2.2: Plattform-Divergenz-Test (Robustheit)**
    *   **Ziel:** Feststellen, ob Floating-Point-Rundungen auf verschiedenen Systemen (z. B. Native Windows vs. WSL/Linux) zu unterschiedlichen Keystreams führen59.*   **Anleitung:** Erstelle eine automatisierte Test-Suite, die für identische Parameter SHA-256-Hashes generiert und Abweichungen zwischen Laufzeitumgebungen loggt1011.*   **Task 2.3: Vergleich mit Industriestandards**
    *   **Ziel:** Einordnung der Performance gegenüber klassischen Verfahren1.*   **Anleitung:** Implementiere ein Benchmark-Modul, das den Durchsatz (bps) von ChaosCrypto direkt gegen eine Standard-AES-256-CTR-Verschlüsselung (z. B. via `cryptography`\-Library) misst5.

\--------------------------------------------------------------------------------

3\. Kryptoanalytisches Testing (Hauptteil I)

**Zweck:** Empirische Untersuchung der in der Literatur beschriebenen theoretischen Schwachstellen chaosbasierter Systeme am konkreten Prototyp1213.

*   **Task 3.1: Finite Precision & Periodizitäts-Analyse**
    *   **Ziel:** Messung, wann der deterministische Keystream aufgrund endlicher Gleitkommapräzision in periodische Zyklen verfällt12more\_horiz.*   **Anleitung:** Entwickle ein Test-Skript, das sehr lange Keystreams (z. B. >100 MB) generiert und nach Mustern oder Bit-Wiederholungen sucht916.*   **Task 3.2: Sensitivity / Avalanche-Effekt Test**
    *   **Ziel:** Messung der Sensitivität gegenüber minimalen Änderungen in den Eingabeparametern (Rule 3 nach Alvarez & Li)17.*   **Anleitung:** Ändere systematisch nur 1 Bit im Token oder den Koordinaten und berechne, wie viele Bits im resultierenden Keystream sich ändern (Idealwert: 50%)917.*   **Task 3.3: Attraktor-Rekonstruktions-Versuch**
    *   **Ziel:** Prüfung, ob aus Keystream-Fragmenten Rückschlüsse auf den internen Zustand (Lorenz-Parameter) möglich sind (Phase-Space Reconstruction)912.*   **Anleitung:** Implementiere ein Modul, das versucht, die Trajektorie des Lorenz-Systems basierend auf den quantisierten Output-Bytes zu plotten oder statistisch vorherzusagen1218.

\--------------------------------------------------------------------------------

4\. Algorithmische Diversität (Vergleichsszenarien)

**Zweck:** Testen, ob das Konzept der deterministischen Rekonstruktion allgemein auf chaotische Systeme anwendbar ist oder nur beim Lorenz-Attraktor funktioniert1920.

*   **Task 4.1: Implementierung des Rössler-Attraktors**
    *   **Ziel:** Ein zweites Chaossystem als Vergleichsobjekt hinzufügen2021.*   **Anleitung:** Implementiere die Differenzialgleichungen des Rössler-Systems als alternative `engine` im Prototyp, sodass Benchmarks und NIST-Tests für beide Systeme vergleichbar sind2022.

\--------------------------------------------------------------------------------

5\. Security-Funktionalitäts-Testing (Baseline)

**Zweck:** Ermittlung des Performance-Overheads, wenn grundlegende kryptographische Anforderungen (Integrität) erfüllt werden1323.

*   **Task 5.1: Integritäts-Messreihe (HMAC)**
    *   **Ziel:** Messung der Auswirkung einer Authentifizierung auf die Gesamtlaufzeit23.*   **Anleitung:** Erstelle eine Variante des `encrypt`\-Befehls, die zusätzlich einen HMAC (Hash-based Message Authentication Code) über den Chiffretext berechnet, und miss den Zeitverlust gegenüber der reinen XOR-Operation823.

\--------------------------------------------------------------------------------

6\. Usability-Datenerhebung

**Zweck:** Objektive Datenbasis für die Bewertung der Benutzerfreundlichkeit schaffen224.

*   **Task 6.1: CLI-Metriken Erfassung**
    *   **Ziel:** Messung der Komplexität des Workflows24.*   **Anleitung:** Erweitere das Logging, um die Anzahl der notwendigen Parameter pro Befehl (`init`, `encrypt`, `decrypt`) und die Rate fehlerhafter Eingaben während Test-Sessions zu protokollieren2425.

\--------------------------------------------------------------------------------

Wichtiger Hinweis für die Umsetzung:

Alle Tests müssen die Ergebnisse in das bestehende `analysis.csv` oder `results.csv` Format exportieren, um eine nahtlose Integration in die automatische **Report-Erzeugung** zu ermöglichen6more\_horiz.