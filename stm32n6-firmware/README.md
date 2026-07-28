# EMaGer sur STM32N6 (NUCLEO-N657X0-Q)

Outillage, configuration VS Code et squelette de firmware pour faire tourner un modèle
EMaGer sur un NUCLEO-N657X0-Q.

Le modèle lui-même vient de [EMaGerLib](https://github.com/SBIOML/EMaGerLib) :
`examples/deployment/export_model.py` produit un `.tflite` INT8 et le runtime C
(`emager_proto.c`). Ce dossier-ci s'occupe de tout ce qui vient après — installer la
chaîne d'outils, créer le projet, compiler le réseau pour le NPU, flasher, et vérifier
que la carte calcule bien la même chose que le PC.

> **Ce dossier est destiné à devenir son propre dépôt.** Il est stocké ici pour ne pas
> être perdu ; voir [Le sortir en dépôt autonome](#le-sortir-en-dépôt-autonome) en bas.

---

## Par où commencer

Les documents se lisent dans l'ordre. Le 1 et le 2 sont l'installation ; le reste ne
sert qu'une fois la carte reconnue.

| # | Document | Ce que tu en sors |
|---|---|---|
| 1 | [Installation Windows](docs/01-installation-windows.md) | tous les outils installés, carte détectée |
| 2 | [VS Code](docs/02-vscode.md) | compiler et déboguer depuis VS Code |
| 3 | [Créer le projet CubeMX](docs/03-projet-cubemx.md) | un projet CMake N6 qui compile |
| 4 | [Le modèle et le NPU](docs/04-modele-et-npu.md) | le réseau EMaGer compilé pour le Neural-ART |
| 5 | [Boot, signature et flash](docs/05-flash-et-boot.md) | le firmware qui démarre tout seul |
| 6 | [Dépannage](docs/06-depannage.md) | quand ça ne marche pas |

---

## Ce que le N6 change par rapport à un STM32 ordinaire

Trois choses, et elles expliquent la quasi-totalité des surprises :

1. **Pas de flash interne.** Le code et les données `const` vivent dans une flash OSPI
   externe. Le démarrage passe par un *FSBL* (first stage boot loader) et les images
   doivent être **signées** avant d'être programmées. Un STM32 classique n'a besoin
   d'aucune de ces étapes.
2. **Le réseau est compilé pour le NPU** (Neural-ART) par ST Edge AI Core, et piloté par
   le runtime **LL_ATON** — pas par l'API `ai_network_run()` de X-CUBE-AI classique. Le
   code d'exemple qu'on trouve pour un STM32F4/H7 ne se transpose pas tel quel.
3. **Deux modes de boot** sélectionnés par des cavaliers. Tant que tu développes, la
   carte est en *Development boot* et le code est chargé en RAM ; pour qu'elle démarre
   seule il faut basculer en *boot from flash*. Oublier ce cavalier est l'erreur nº 1.

Et une remarque honnête sur le choix de carte : à ~152 K MAC et 54 KiB, **le modèle
EMaGer n'a pas besoin d'un NPU.** Un Cortex-M4F avec CMSIS-NN le ferait tourner
largement en temps réel. Le N6 est une carte parfaitement valable, mais ce qui la
justifie n'est pas l'accélération du réseau — l'acquisition et le fenêtrage dominent
de loin. Si tu as le choix et que l'objectif est une prothèse embarquée, un L4/U5
serait plus adapté côté consommation.

---

## Structure

```
docs/          les six guides ci-dessus
app/           la couche applicative, indépendante du runtime d'inférence
  emager_app.{h,c}          MAV -> forward -> classification -> calibration
  emager_nn_port.h          le contrat que le runtime doit remplir
  emager_nn_port_stub.c     faux réseau, pour valider tout le reste d'abord
  emager_nn_port_llaton.c   l'implémentation Neural-ART (à compléter, voir doc 4)
scripts/       PowerShell : vérification de la chaîne d'outils, export, signature, flash
.vscode/       tâches, réglages, extensions recommandées
model/         là où atterrissent les artefacts générés (git-ignoré)
```

Ce qui **n'est pas** versionné ici : le projet CubeMX (`.ioc`, `Core/`, `Drivers/`,
`CMakeLists.txt`). Il est généré par CubeMX sur ta machine, pour ta version des outils,
et le livrer tout fait produirait un projet qui ne compile pas chez toi. La doc 3
explique comment le générer **dans ce dossier** ; `app/` et `model/` s'y greffent par
simple référence CMake, sans copie.

Une fois généré, `Core/` et le `.ioc` sont à committer (ils contiennent ton travail) ;
`Drivers/` et `Middlewares/` non (des centaines de Mo de code ST intact, régénérable).
Le `.gitignore` fourni applique déjà cette règle.

---

## Le sortir en dépôt autonome

Ce dossier n'a aucune dépendance sur le reste d'EMaGerLib (les artefacts du modèle sont
copiés dedans, pas référencés). Pour en faire un vrai dépôt :

```powershell
# depuis le dossier parent d'EMaGerLib
mkdir emager-stm32n6-firmware
Copy-Item -Recurse EMaGerLib\stm32n6-firmware\* emager-stm32n6-firmware\
cd emager-stm32n6-firmware
git init
git add .
git commit -m "chore: initial import from EMaGerLib"
```

Puis crée le dépôt sur GitHub (bouton **New** sur https://github.com/SBIOML, ou sur ton
compte) **sans** README ni .gitignore, et pousse :

```powershell
git remote add origin https://github.com/SBIOML/emager-stm32n6-firmware.git
git branch -M main
git push -u origin main
```

Si tu le crées sur ton compte perso, tu peux le transférer ensuite à SBIOML via
*Settings → General → Transfer ownership*.
