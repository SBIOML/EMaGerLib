# 2. VS Code

Objectif : compiler, flasher et déboguer sans quitter l'éditeur.

Un point à comprendre avant tout, parce qu'il explique la moitié des questions qu'on se
pose : **l'extension STM32 (v2.x) ne crée pas de projet.** Contrairement à ce qu'on
attend, elle ne remplace pas CubeIDE. Elle prend un projet **CMake généré par CubeMX**
et lui ajoute de quoi compiler et déboguer. Le projet vient de CubeMX ; VS Code est
l'interface.

Donc l'ordre est : CubeMX génère → VS Code ouvre. Jamais l'inverse.

---

## Extensions

Ouvre VS Code dans ce dossier ; il proposera les extensions listées dans
`.vscode/extensions.json`. Sinon, installe-les à la main :

| Extension | Identifiant | Rôle |
|---|---|---|
| STM32 VS Code Extension | `stmicroelectronics.stm32-vscode-extension` | importe le projet CubeMX, debug ST-LINK |
| C/C++ | `ms-vscode.cpptools` | IntelliSense, moteur de debug |
| CMake Tools | `ms-vscode.cmake-tools` | configure/compile |

`Cortex-Debug` (`marus25.cortex-debug`) n'est **pas** nécessaire : l'extension ST fournit
son propre type de debug. Ne l'installe que si tu veux utiliser OpenOCD ou J-Link à la
place — deux configurations de debug concurrentes sur le même projet créent surtout de
la confusion.

---

## Indiquer où sont les outils

L'extension ST doit trouver CubeCLT, CubeMX et CubeProgrammer. Normalement elle les
détecte seule. Si elle se plaint (« Could not find… »), renseigne-les :

**Ctrl+Shift+P** → *Preferences: Open User Settings (JSON)* → ajoute, en adaptant les
versions :

```jsonc
{
  "stm32-for-vscode.cubeCLTPath": "C:\\ST\\STM32CubeCLT_1.18.0",
  "stm32-for-vscode.cubeMXPath":  "C:\\Program Files\\STMicroelectronics\\STM32Cube\\STM32CubeMX\\STM32CubeMX.exe",
  "stm32-for-vscode.cubeProgrammerPath": "C:\\Program Files\\STMicroelectronics\\STM32Cube\\STM32CubeProgrammer\\bin"
}
```

> Les noms exacts des clés varient selon la version de l'extension. Le moyen fiable :
> **Ctrl+,** puis tape `stm32` dans la recherche des paramètres — tu verras les clés
> réellement supportées par la version que tu as, avec leur description. Ne recopie pas
> des clés trouvées sur un forum sans les avoir vues dans cette liste.

---

## Ouvrir le projet

Une fois le projet CubeMX généré ([doc 3](03-projet-cubemx.md)) :

1. **File → Open Folder** sur le dossier qui contient le `CMakeLists.txt` généré.
2. **Ctrl+Shift+P** → `STM32: Import CMake project` (le libellé exact varie ;
   cherche `STM32` dans la palette et prends l'entrée d'import).
3. L'extension détecte la cible, propose un kit CMake et écrit une configuration de
   debug.

Si CMake Tools demande un *kit*, choisis celui de CubeCLT
(`arm-none-eabi-gcc` dans `C:\ST\STM32CubeCLT_<ver>\GNU-tools-for-STM32\bin`). Ne
choisis surtout pas un compilateur Visual Studio : il compilerait pour ton PC.

---

## Compiler

- **Ctrl+Shift+B** lance la tâche de build par défaut.
- Ou palette → `CMake: Build`.

Les tâches fournies dans `.vscode/tasks.json` couvrent le cycle complet :

| Tâche | Ce qu'elle fait |
|---|---|
| `Build` | compile (CMake + Ninja) |
| `Clean` | supprime `build/` |
| `Sign firmware` | signe le `.bin` — **obligatoire sur N6** |
| `Flash (external OSPI)` | signe puis programme la flash externe |
| `Check toolchain` | relance la vérification du document 1 |
| `Export model + test vectors` | régénère le modèle depuis EMaGerLib |

Elles appellent les scripts de `scripts\`. Ouvre-les : ce sont quelques lignes chacun,
et il vaut mieux comprendre la commande que la lancer en aveugle — surtout celles qui
écrivent dans la flash.

---

## Déboguer

**F5** démarre le ST-LINK GDB server et s'arrête au début de `main()`.

Trois choses propres au N6 qui n'ont pas d'équivalent sur un STM32 ordinaire :

**1. La carte doit être en *Development boot*.** En mode « boot from flash », le
processeur exécute la flash externe et le port de debug n'est pas ouvert de la même
façon. Si F5 échoue avec une erreur de connexion, vérifie les cavaliers **avant** de
soupçonner VS Code — voir [doc 5](05-flash-et-boot.md).

**2. Debug en RAM vs debug depuis la flash externe.** Pendant le développement, le plus
rapide est de charger le binaire directement en RAM : pas de signature, pas de
programmation de flash, quelques secondes par itération. Le revers est que ça ne teste
pas le chemin de boot réel — un firmware qui marche en RAM peut très bien ne pas
démarrer seul. Fais les deux : itère en RAM, valide en flash avant de conclure.

**3. Le SWO ne fonctionne pas** avec le serveur GDB ST-LINK sur cette carte (« SWO
support is not available from the probe »). Ce n'est pas une mauvaise configuration de
ta part. Pour tracer, utilise l'**UART du VCP ST-LINK** — la carte expose un port COM
virtuel, et un simple `printf` redirigé dessus est plus fiable que le SWO ici. Le
squelette applicatif fournit un point d'accroche pour ça
(`emager_app_log` dans [`app/emager_app.h`](../app/emager_app.h)).

---

## Vérifier que la boucle complète fonctionne

Avant d'écrire du code EMaGer, valide la chaîne sur un projet vide : un `main()` qui
fait clignoter la LED. Si Build + F5 + point d'arrêt fonctionnent, l'outillage est bon
et tout problème ultérieur vient de ton code ou du modèle. Sauter cette étape, c'est
s'exposer à déboguer simultanément l'IA, le NPU et la chaîne d'outils.

Ensuite : [créer le projet CubeMX](03-projet-cubemx.md).
