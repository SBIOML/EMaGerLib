# 6. Dépannage

Classé par symptôme. Les causes sont ordonnées par fréquence réelle, pas par
vraisemblance théorique — commence par le haut même si ça paraît trop bête.

---

## La carte n'est pas détectée / erreur de connexion

`STM32_Programmer_CLI -c port=SWD mode=HOTPLUG` échoue.

1. **Les cavaliers de boot.** La carte doit être en *Development boot*. C'est la cause
   nº 1, de loin. Change, puis **appuie sur RESET** — un changement de cavalier sans
   reset n'a aucun effet.
2. **Le câble.** Un câble USB-C fin ou long ne fournit pas assez de courant et produit
   des échecs intermittents qui ressemblent à un problème logiciel. Essaie celui fourni
   avec la carte, court et de bonne qualité.
3. **Le connecteur USB.** La carte en a plusieurs. Branche-toi sur celui du **ST-LINK**,
   pas sur le USB utilisateur.
4. **Le firmware du ST-LINK.** Mets-le à jour depuis l'interface STM32CubeProgrammer.
   Un ST-LINK ancien se connecte parfois mais échoue au moment d'écrire en flash
   externe.
5. **Le mode.** Essaie `mode=UR` (under reset) ou `mode=HOTPLUG` — l'un des deux passe
   souvent quand l'autre échoue.

---

## `Could not find an STM32CubeIDE installation` dans VS Code

L'extension cherche CubeCLT (ou CubeIDE) et ne le trouve pas.

Installe **STM32CubeCLT** ([doc 1](01-installation-windows.md)), puis renseigne son
chemin dans les paramètres — **Ctrl+,** → `stm32` pour voir les clés réellement
supportées par ta version de l'extension. Redémarre VS Code : l'extension ne relit pas
ces paramètres à chaud.

---

## VS Code ne compile pas / pas de kit CMake

- Vérifie que le projet a bien été généré avec **Toolchain = CMake** dans CubeMX. Avec
  « STM32CubeIDE », il n'y a pas de `CMakeLists.txt` et rien ne fonctionnera.
- Ouvre le dossier qui **contient** `CMakeLists.txt`, pas son parent.
- Si CMake Tools propose un kit Visual Studio, refuse : choisis celui pointant vers
  `arm-none-eabi-gcc` dans `C:\ST\STM32CubeCLT_<ver>\GNU-tools-for-STM32\bin`.

---

## Le debug démarre mais `SWO support is not available from the probe`

Attendu sur cette carte avec le serveur GDB ST-LINK. Ce n'est pas une erreur de
configuration.

Utilise l'**UART du VCP ST-LINK** pour tracer : la carte expose un port COM virtuel.
Redirige `printf` dessus (ou branche `emager_app_log`) et ouvre le port avec n'importe
quel terminal série. C'est plus fiable que le SWO ici.

---

## Ça marche en debug mais la carte ne démarre pas seule

Le cas classique du N6. Dans l'ordre :

1. **Les cavaliers** sont-ils passés en *boot from flash* ? Avec RESET après.
2. **L'image est-elle signée ?** Une image non signée est refusée par la ROM, sans
   message. Vérifie que tu as flashé le `_signed.bin`, pas le `.bin` brut.
3. **As-tu flashé l'image que tu viens de compiler ?** L'erreur la plus fréquente est de
   reflasher un binaire signé précédemment. `scripts\sign-and-flash.ps1` refuse de
   flasher un fichier signé plus ancien que le binaire compilé.
4. **Le FSBL est-il programmé** à `0x70000000` ?
5. **Les adresses** : FSBL `0x70000000`, application `0x70100000`.

---

## `stedgeai` introuvable

Le pack X-CUBE-AI est installé sous
`%USERPROFILE%\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\<version>\Utilities\windows\`.
Le nom du dossier de version change à chaque mise à jour, ce qui casse les chemins
codés en dur.

`scripts\check-toolchain.ps1` le localise automatiquement. Si le script ne trouve rien,
c'est que le pack n'est pas installé : CubeMX → *Manage embedded software packages*.

---

## Le modèle ne se compile pas pour le NPU

- Vérifie que tu fournis bien `model_int8.tflite` et **pas** `model_float32.tflite` ni
  `model.onnx`. Le Neural-ART veut de l'INT8.
- *Compression* doit être sur **none** : le modèle est déjà quantifié.
- Si une couche est refusée, lis `<name>_generate_report.txt`. Un repli sur le
  Cortex-M55 est normal pour ce modèle (voir [doc 4](04-modele-et-npu.md)) ; un refus
  franc de compiler est autre chose.

---

## Le self-test échoue

Lis les deux compteurs **séparément**, ils n'ont pas la même signification.

### `n_embed_ok` est bon, `n_class_ok` est mauvais

Le réseau va bien. Le problème est dans les prototypes :

- la carte a-t-elle déjà été calibrée ? Dans ce cas c'est **normal** — la calibration
  remplace les prototypes, donc le même embedding donne une autre classe. Relance le
  test contre une copie fraîche de `EMAGER_FACTORY_PROTOTYPES` ;
- un blob de prototypes périmé a-t-il été rechargé depuis la flash après un changement
  de modèle ?
- les drapeaux `valid` sont-ils tous à 1 ?

### `n_embed_ok` est mauvais

Le chemin réseau. Par ordre de probabilité :

1. **Le MAV ne correspond pas à l'entraînement.** Taille de fenêtre (200), incrément
   (10), **ordre des canaux** de la grille 4×16. C'est la cause la plus fréquente et la
   plus facile à rater : une grille transposée donne des embeddings plausibles mais
   faux.
2. **La quantification d'entrée.** Utilise `emager_quantize_input()`, pas des constantes
   recopiées à la main. Les scales viennent du `.tflite` réel et changent à chaque
   réexport.
3. **Le runtime mal câblé** : mauvais tenseur d'entrée/sortie, layout inattendu,
   buffers non fournis.
4. **Les vecteurs ne correspondent pas au modèle flashé.** `emager_test_vectors.h`
   n'est valable que pour l'export exact qui l'a produit. Après un réentraînement ou un
   réexport, régénère-les.

`worst_vector` te donne l'indice du pire cas — regarde sa classe, un échec concentré sur
un seul geste oriente vers les données, un échec réparti vers le chemin de calcul.

---

## Le modèle est lent

Avant d'optimiser, mesure **où** part le temps. Le réseau fait ~152 K MAC : sur cette
carte il n'est presque certainement pas le coupable. Instrumente séparément
l'acquisition, le calcul du MAV et l'inférence — dans la plupart des portages EMaGer,
les deux premiers dominent largement.

Si l'inférence elle-même est lente, lis le rapport de génération : la part du graphe qui
retombe sur le Cortex-M55 (padding circulaire, BatchNorm flottante) est l'explication la
plus probable.

---

## Rien de tout ça

Reviens à un état connu : projet CubeMX neuf, LED qui clignote, sans modèle
([doc 3](03-projet-cubemx.md)). Si ça ne marche pas, le problème est dans l'outillage.
Si ça marche, réintroduis les morceaux un par un — port du réseau, puis self-test, puis
l'application. Déboguer trois couches à la fois ne converge pas.
