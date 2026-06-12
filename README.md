# jeu-oralisation

Jeu d'oralisation à destination des orthophonistes — **application Python (macOS),
100 % locale, sans réseau ni API.**

L'enfant produit un son dans le micro pour faire avancer un objet (voiture, ballon,
fusée) jusqu'au drapeau d'arrivée.

## Les 8 activités

- **Son libre** : n'importe quel son tenu fait avancer l'objet (voiture, ballon,
  fusée) — travail du souffle, de la voix, de la durée.
- **Son cible** : l'objet n'avance que si l'enfant produit le bon son —
  voyelles **A E I O U OU** (formants) ou consonnes tenues **S CH F** (fiables) /
  **V Z R L AN** (expérimentales, validées à l'oreille). Option **Non lecteur** :
  affiche le **picto-référent** du son (serpent=sss, vent=fff, moto=vvv,
  abeille=zzz, lion=rrr…) au lieu de la lettre.
- **Fusion** : fusion phonémique pour non-lecteurs — une image (*loup*) **+** un
  son-référent (*douche* = « ch ») **=** le mot à dire (*louche*), vérifié par
  Whisper. Bouton *Montrer* pour révéler la réponse.
- **Mot cible** : l'adulte tape un mot ou choisit une **liste**, le mot s'affiche
  en grand avec son **pictogramme ARASAAC** ; l'enfant le dit, un **feu
  d'artifice** salue la bonne réponse (Whisper local). Options : **Cacher le
  mot** (dénomination : l'enfant nomme l'image), **Mains libres** (écoute
  déclenchée à la voix, sans clic), passage automatique au mot suivant,
  **jetons** ★ vers un objectif.
- **Intensité** : garder la voix dans la zone verte (douce / moyenne / forte) —
  contrôle du volume, projection vocale.
- **Hauteur** : voix grave / médium / aiguë — la hauteur (F0) fait monter le
  ballon ; prosodie, mue, surdité appareillée.
- **Souffle** : tenir « aaaa » pendant 2/3/5/8 s pour gonfler le ballon — temps
  maximum de phonation.
- **Écoute** : l'app **prononce un mot** (synthèse vocale macOS, locale) et
  l'enfant clique la bonne image parmi 2–4 — discrimination auditive,
  vocabulaire.
- **Paires minimales** : deux images (*poule/boule*, *chou/joue*…), l'enfant doit
  dire le mot encadré ; si l'app entend l'autre mot de la paire, elle le dit —
  contrastes p/b, t/d, k/g, f/v, s/z, ch/j, s/ch, r/l.

Toutes les réussites sont consignées dans le **journal de séance** (bouton
« Exporter séance » → CSV dans `~/Documents/jeu-oralisation/seances/`).

### Menu pictogrammes 🖼

Dans les modes à images (Mot, Fusion, Paires, Écoute, Son cible non-lecteur), le
bouton **« 🖼 Pictos »** ouvre une fenêtre pour, sur le mot courant :

- voir l'**image actuelle** en cache ;
- choisir **une autre proposition ARASAAC** (clic sur une vignette) si la première
  ne convient pas ;
- **importer une image locale** (PNG/JPG…) via le sélecteur de fichiers.

La nouvelle image remplace celle en cache (`~/Library/Caches/jeu-oralisation/pictos/`)
et est réutilisée partout.

### Listes de mots

Les modes Mot cible et Écoute utilisent des listes éditables : de simples fichiers
`.txt` (un mot par ligne) dans `~/Documents/jeu-oralisation/listes/`. Des listes
d'exemple (animaux, son CH, son S, son R…) sont créées au premier lancement.

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
Internet ; les images sont ensuite mises en cache (hors-ligne) dans
`~/Library/Caches/jeu-oralisation/pictos/`.
Attribution requise en cas de diffusion : « Pictogrammes : ARASAAC (arasaac.org) ».

**Aucun compte ni clé d'API n'est nécessaire** : les endpoints de recherche et de
téléchargement sont publics (`api.arasaac.org` / `static.arasaac.org`).

En cas de souci, lance le diagnostic (affiche l'URL, l'ID trouvé et l'erreur
éventuelle) :

```bash
python pictos.py chat
```

Sur macOS, si tu vois une erreur `CERTIFICATE_VERIFY_FAILED`, l'app bascule
automatiquement en mode non vérifié pour récupérer l'image ; tu peux aussi exécuter
une fois *« Install Certificates.command »* (dans `/Applications/Python 3.x/`) ou
`pip install certifi`.

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

## Empaqueter en application macOS cliquable

Pour obtenir une **app `.app` lançable en un clic** (à glisser dans `/Applications`),
avec le **modèle Whisper embarqué** :

```bash
chmod +x build_mac.sh
./build_mac.sh base        # ou: ./build_mac.sh small  (plus précis, plus lourd)
```

Le script crée un environnement, pré-télécharge le modèle, puis lance PyInstaller via
`jeu-oralisation.spec`. Résultat : **`dist/Jeu d'oralisation.app`**.

> Au 1er lancement d'une app non signée : **clic droit → Ouvrir** (puis « Ouvrir »).
> L'app demande l'autorisation micro (déclarée dans le bundle).

### Caches (images et modèles)

Une app `.app` est en lecture seule : les caches sont donc écrits dans un dossier
**utilisateur persistant** (`~/Library/Caches/jeu-oralisation/`) :

- `pictos/` : pictogrammes ARASAAC téléchargés ;
- `whisper-models/` : modèles téléchargés à la volée (tailles non embarquées).

Le modèle choisi via `./build_mac.sh <taille>` est, lui, **embarqué dans l'app**
(dossier `models/whisper-<taille>/`, inclus par le `.spec`) → ce modèle fonctionne
**hors-ligne dès le premier lancement**. Les autres tailles, si l'adulte les
sélectionne, se téléchargent une fois puis sont mises en cache.

> Construire l'app embarque Python + pygame + Whisper (ctranslate2/onnxruntime) :
> compter **~0,5–1 Go** selon la taille du modèle. La construction se fait **sur un
> Mac** (PyInstaller produit un binaire pour la plateforme courante).
