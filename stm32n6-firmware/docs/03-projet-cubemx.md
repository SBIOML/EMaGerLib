# 3. Créer le projet CubeMX

Le projet n'est pas versionné dans ce dépôt : il dépend de ta version de CubeMX, de la
HAL et de X-CUBE-AI, et un projet généré ailleurs ne compile généralement pas tel quel.
Tu le génères ici en dix minutes, puis tu y greffes les fichiers de `app/`.

---

## Générer

1. CubeMX → **File → New Project** → onglet **Board Selector** → cherche
   `NUCLEO-N657X0-Q` → **Start Project**.
2. CubeMX propose d'initialiser les périphériques par défaut de la carte : **oui**. Ça
   configure les horloges, la LED, le VCP UART et l'OSPI — tout ce qu'il faut pour
   démarrer.
3. S'il manque le package `STM32Cube_FW_N6`, CubeMX propose de le télécharger. Accepte
   (quelques centaines de Mo).

### Ce qu'il faut vérifier dans l'onglet *Pinout & Configuration*

- **USART/VCP** activé — c'est ton seul canal de trace utilisable, le SWO ne
  fonctionnant pas ici (voir [doc 2](02-vscode.md)). Note le numéro d'USART relié au
  VCP ST-LINK.
- **LED utilisateur** en GPIO sortie — indispensable pour un premier test qui ne dépend
  de rien d'autre.
- L'**horloge** : laisse la configuration par défaut de la carte au début. Le N6 a un
  arbre d'horloge complexe et le tuner avant que quoi que ce soit ne fonctionne est une
  perte de temps.

### Onglet *Project Manager*

| Champ | Valeur |
|---|---|
| Project Name | `emager-n6` (ou ce que tu veux, **sans espace ni accent**) |
| Project Location | **le dossier qui contient ce dépôt** — ex. `C:\dev`, de sorte que le projet atterrisse dans `C:\dev\emager-n6\` |
| **Toolchain / IDE** | **CMake** |

Génère le projet **dans le dossier du firmware lui-même**, pas à côté : `app/`,
`docs/`, `scripts/` et le projet CubeMX cohabitent alors dans un seul dépôt, et le
`.gitignore` fourni est déjà réglé pour ça (il ignore `Drivers/` et `Middlewares/`,
régénérables, mais conserve `Core/` et le `.ioc`, qui contiennent ton travail).

Le choix **CMake** est le point critique de toute cette page : l'extension VS Code ne
sait travailler qu'avec ça. Si tu choisis STM32CubeIDE, tu devras tout regénérer.

Dans *Code Generator*, coche **Generate peripheral initialization as a pair of .c/.h
files per peripheral**. Ça évite qu'une regénération future n'écrase un gros `main.c`
où tu aurais mis ton code.

Puis **GENERATE CODE**.

---

## Où mettre ton code

CubeMX **réécrit** les fichiers qu'il génère à chaque regénération, mais il préserve ce
qui se trouve entre ses marqueurs :

```c
/* USER CODE BEGIN 2 */
    emager_app_init();
/* USER CODE END 2 */
```

Tout ce qui est en dehors de ces blocs disparaîtra. C'est la raison d'être de `app/` :
le code applicatif vit dans ses propres fichiers, que CubeMX ne touche jamais, et
`main.c` ne contient que des appels dans les blocs `USER CODE`.

**Ne copie rien dans `Core/Src/`.** Les fichiers restent dans `app/` et c'est CMake qui
va les chercher. Un fichier copié devient un doublon qu'on oublie de mettre à jour ; un
fichier référencé ne peut pas diverger.

---

## Déclarer les nouveaux fichiers à CMake

Le `CMakeLists.txt` généré liste les sources explicitement. Ajoute les tiennes dans le
bloc utilisateur prévu :

```cmake
# USER CODE BEGIN Sources
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    app/emager_app.c
    app/emager_nn_port_llaton.c   # ou emager_nn_port_stub.c au début -- UN SEUL des deux
    model/emager_proto.c
    model/emager_selftest.c
)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    app
    model
)
# USER CODE END Sources
```

`emager_proto.c` et `emager_selftest.c` viennent d'EMaGerLib et atterrissent dans
`model/` — le [document 4](04-modele-et-npu.md) explique comment les récupérer avec le
modèle.

> Compile **un seul** port : `emager_nn_port_stub.c` **ou** `emager_nn_port_llaton.c`.
> Les deux définissent `emager_nn_init` / `emager_nn_forward` et les inclure ensemble
> donne une erreur de symbole dupliqué à l'édition de liens. Commence par le stub — il
> permet de valider le fenêtrage, le MAV et la calibration sans NPU fonctionnel.

---

## Premier test : la LED

Avant tout code EMaGer, valide la chaîne complète. Dans `main.c` :

```c
/* USER CODE BEGIN WHILE */
while (1)
{
    HAL_GPIO_TogglePin(LED_GREEN_GPIO_Port, LED_GREEN_Pin);
    HAL_Delay(500);
}
/* USER CODE END WHILE */
```

Compile (**Ctrl+Shift+B**), lance le debug (**F5**), pose un point d'arrêt dans la
boucle. Si la LED clignote et que le point d'arrêt est atteint : outillage validé.

Tant que ce test ne passe pas, ne charge pas le modèle. Déboguer le NPU et la chaîne
d'outils en même temps est une très mauvaise façon de passer un après-midi.

Ensuite : [le modèle et le NPU](04-modele-et-npu.md).
