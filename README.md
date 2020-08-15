# Poker Bot Competition

## Venv initialisieren

In Terminal auf Pfad /source navigieren. 

1. Venv generieren: python3 -m venv venv
2. Shell Script ausführen: sh run_evaluation.sh 
(dependencies werden dadurch in venv installiert)

## Ausführen

Zur Ausführung Agenten-Zusammenstellung wie gewünscht anpassen in runEvaluation.py. 
In Linie #50 kann die Anzahl Evaluationen angegeben werden.
Das Setup ist so eingerichtet, dass standardmässig add_dqn_bots aus configuration/CashGameConfig.py aufgerufen mit. Die Methode registriert sechs Ddqn und DdqnPer Agenten mit unterschiedlichen Konfigurationen. 

## Double Deep Q Network with Prioritized Experience Replay Overview

### Ablaufdiagramm DdqnPer

![Ablaufdiagramm DdqnPer](resources/Ablaufdiagramm_DDQNPer.png)


### MVBasePokerPlayer Ablauf

![Übersicht MVBasePokerPlayer Ablauf](resources/MVBasePokerPlayer.png)


## Resources

[Developer Guide](resources/Developer_Guide.pdf)
