# jeu-oralisation

Jeu d'oralisation à destination des orthophonistes — **application Python (macOS),
100 % locale, sans réseau ni API.**

L'enfant produit un son dans le micro pour faire avancer un objet (voiture, ballon,
fusée) jusqu'au drapeau d'arrivée.

- **Son libre** : n'importe quel son tenu fait avancer l'objet (travail du souffle,
  de la voix, de la durée).
- **Voyelle cible** : l'objet n'avance que si l'enfant prononce la bonne voyelle
  (A E I O U), reconnue par analyse des **formants** (voir plus bas).
- **Mot cible** : l'adulte tape un mot, il s'affiche en grand, l'enfant le dit ; un
  **feu d'artifice** salue la bonne réponse. Reconnaissance par **Whisper** local
  (`faster-whisper`, open source, sans réseau après le 1er téléchargement).

## Installation (macOS)

Pré-requis : Python 3.10+.

```bash
cd jeu-oralisation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python game.py
```

Au premier lancement, macOS demande l'autorisation d'accès au **micro** : accepte-la
(Réglages Système → Confidentialité et sécurité → Microphone).

## Utilisation

- **Scène** : Voiture / Ballon / Fusée.
- **Mode** : Son libre ou Son cible.
- **Cible** (mode Son cible) : choisis la voyelle A / E / I / O / U (clic ou touche
  clavier correspondante).
- **Sensibilité** (slider) : ajuste le seuil de détection du son (monte-la si le micro
  est peu réactif, baisse-la si l'environnement est bruyant).
- **Exigence** (slider, mode cible) : niveau de ressemblance minimal pour avancer.
- **Reste en place** : si activé (par défaut), le véhicule **conserve sa progression**
  quand le son s'arrête, et ne repart pas à zéro. Désactive-le pour travailler le son
  *tenu* (l'objet redescend alors si le son cesse).
- **Micro** : démarrer / arrêter.
- **ESPACE (maintenu)** : repli sans micro (modes son), pour tester le jeu ou
  récompenser manuellement.

### Mode « Mot cible »

1. Choisis le mode **Mot cible**, démarre le micro.
2. **Tape un mot** au clavier (il s'affiche en grand). « Effacer » remet à zéro.
3. L'enfant clique **« 🎤 Parler »** (ou touche **Entrée**) et dit le mot : l'app
   écoute ~2 s, transcrit localement, et lance un **feu d'artifice** si c'est bon.

**Bouton « Modèle »** : bascule la taille du modèle Whisper — `tiny` (rapide) →
`base` → `small` (plus précis, plus lent). L'adulte choisit selon le besoin. Chaque
taille se télécharge à son premier usage, puis reste en cache.

**Bouton « Image »** : récupère et affiche automatiquement un **pictogramme ARASAAC**
correspondant au mot (au-dessus du texte). Désactivable.

> Au tout premier usage de ce mode, `faster-whisper` télécharge le modèle (~75–460 Mo
> selon la taille) depuis Hugging Face — **connexion Internet requise une seule fois**
> par taille de modèle. Ensuite, tout fonctionne hors-ligne. (Modèle par défaut dans
> `speech.py` : `DEFAULT_MODEL`.)
>
> Whisper reste perfectible sur un **mot isolé** prononcé par un enfant ; la
> correspondance est volontairement tolérante (accents, petites erreurs).

### Pictogrammes ARASAAC

Les images proviennent d'[ARASAAC](https://arasaac.org) (Gouvernement d'Aragon),
sous licence **Creative Commons BY-NC-SA**. La 1re recherche d'un mot nécessite
Internet ; les images sont ensuite mises en cache dans `pictos_cache/` (hors-ligne).
Attribution requise en cas de diffusion : « Pictogrammes : ARASAAC (arasaac.org) ».

## Comment marche la reconnaissance (`recognition.py`)

Choix techniques pour rester **simple, local et universel** (sans calibrage par
enfant) :

1. **Formants par LPC** : on modélise le spectre par prédiction linéaire et on prend
   les racines du polynôme pour obtenir F1, F2, F3. C'est bien plus fiable que le
   « pic de FFT » (qui suit les harmoniques de la voix, pas les formants).
2. **Sélection des pics étroits** : les vrais formants ont une bande passante faible ;
   on écarte ainsi les résonances parasites.
3. **Échelle de Bark** : les formants sont convertis sur une échelle perceptuelle, ce
   qui atténue les différences homme / femme / enfant — d'où l'aspect « universel »
   sans calibrage.
4. **Plus proche voisin** : la voyelle retenue est celle dont les formants de
   référence (couvrant adulte masculin, adulte féminin et enfant) sont les plus
   proches ; F3 aide à distinguer les voyelles arrondies (O, U) des antérieures
   (I, E).

Limite connue : « U » (/y/, antérieur arrondi) reste la voyelle la plus délicate,
surtout pour les voix d'enfants très aiguës. A / E / I / O sont robustes.

## Tests

Sans micro, on valide la reconnaissance sur des voyelles synthétiques (3 profils de
voix) :

```bash
python test_recognition.py
```

## Empaqueter en application macOS (optionnel)

```bash
pip install pyinstaller
pyinstaller --noconsole --onefile --name "Jeu d'oralisation" game.py
```

L'exécutable est généré dans `dist/`.
