# 1. Installation sous Windows

Pour un **NUCLEO-N657X0-Q**, sur **Windows 10/11 64 bits**, avec droits administrateur.

L'ordre compte. Chaque étape se termine par une vérification — fais-la, une erreur
détectée ici coûte deux minutes, la même erreur détectée au moment de flasher coûte
une soirée.

> Les numéros de version indiqués sont les **minimums connus** pour le N6. Prends
> toujours la version la plus récente proposée ; le N6 est un produit récent et le
> support s'améliore à chaque révision.

---

## 0. Avant de commencer

**Le câble.** Le N6 consomme beaucoup. Utilise un **câble USB-C de qualité, court**
(celui fourni avec la carte de préférence). Un câble de charge fin ou long provoque des
chutes de tension qui se manifestent en erreurs de connexion ST-LINK apparemment
aléatoires — on cherche un problème de driver pendant des heures alors que c'est le
câble.

**L'emplacement du projet.** Mets ton code dans un chemin **court, local, sans espaces
et hors OneDrive** — par exemple `C:\dev\emager-n6`. Trois raisons, toutes vécues :

- les projets embarqués imbriquent profondément (`Drivers/STM32N6xx_HAL_Driver/...`) et
  Windows coupe à 260 caractères ;
- OneDrive synchronise pendant la compilation et verrouille des fichiers objets ;
- des outils de la chaîne gèrent mal les espaces dans les chemins.

Active aussi les chemins longs, une fois pour toutes (PowerShell **administrateur**) :

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**Compte ST.** Tous les téléchargements ST demandent un compte (gratuit). Crée-le
maintenant : https://www.st.com — ça évite d'être interrompu au milieu.

---

## 1. STM32CubeProgrammer (≥ 2.18)

**En premier**, parce qu'il installe le driver ST-LINK dont tout le reste dépend, et
qu'il contient deux outils indispensables au N6 : l'outil de signature et les
*external loaders* pour la flash OSPI.

Téléchargement : https://www.st.com/en/development-tools/stm32cubeprog.html

Pendant l'installation, **accepte l'installation du driver ST-LINK** (case cochée par
défaut). C'est l'étape que les gens sautent.

### Vérification

Branche la carte sur le connecteur USB **ST-LINK** (celui côté ST-LINK, pas le USB
utilisateur), puis :

```powershell
& "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe" --version
& "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe" -c port=SWD mode=HOTPLUG
```

La deuxième commande doit afficher un identifiant de carte et un `ST-LINK FW`. Si elle
échoue, ne continue pas — va voir [le dépannage](06-depannage.md), c'est presque
toujours le cavalier de boot, le câble, ou le firmware ST-LINK.

### Mets à jour le firmware du ST-LINK

Le N6 exige un firmware ST-LINK récent. Lance l'interface graphique
STM32CubeProgrammer → **Firmware upgrade** (bouton dans le panneau de connexion) →
**Upgrade**. Fais-le même si la connexion fonctionne : un ST-LINK trop ancien se
connecte mais échoue plus tard, au moment de programmer la flash externe.

### Ajoute-le au PATH

Beaucoup de commandes de ces guides supposent `STM32_Programmer_CLI` accessible
directement. En PowerShell **administrateur**, de façon permanente :

```powershell
$p = "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin"
[Environment]::SetEnvironmentVariable("Path",
  [Environment]::GetEnvironmentVariable("Path","Machine") + ";$p", "Machine")
```

Ferme et rouvre le terminal, puis `STM32_Programmer_CLI --version` doit répondre.

---

## 2. STM32CubeMX (dernière version)

Le générateur de projet. C'est aussi lui qui installe X-CUBE-AI et le package firmware
N6, donc il n'est pas optionnel même si tu comptes tout faire en ligne de commande.

Téléchargement : https://www.st.com/en/development-tools/stm32cubemx.html

CubeMX a besoin de **Java** ; l'installeur récent l'embarque, mais si CubeMX refuse de
démarrer, c'est là qu'il faut regarder.

### Vérification

Lance CubeMX. Dans **Help → Manage embedded software packages**, tu dois voir une
entrée `STM32N6`. Si elle n'existe pas, ta version de CubeMX est trop ancienne pour la
carte — mets-la à jour avant d'aller plus loin.

---

## 3. STM32CubeCLT (≥ 1.18)

La chaîne d'outils en ligne de commande : `arm-none-eabi-gcc`, `gdb`, **CMake**,
**Ninja**, et le **ST-LINK GDB server**. C'est ce que l'extension VS Code utilise —
sans lui, VS Code ne peut ni compiler ni déboguer.

Téléchargement : https://www.st.com/en/development-tools/stm32cubeclt.html

Installe-le à l'emplacement proposé (`C:\ST\STM32CubeCLT_<version>`). L'installeur
propose d'ajouter les outils au PATH : **accepte**.

### Vérification

Ouvre un **nouveau** terminal (le PATH n'est pas rechargé dans les anciens) :

```powershell
arm-none-eabi-gcc --version
cmake --version
ninja --version
```

Les trois doivent répondre. Si `cmake` renvoie une version installée par ailleurs
(Visual Studio, Chocolatey), ce n'est pas grave tant qu'elle est ≥ 3.22 — mais en cas
de comportement bizarre, c'est un suspect.

---

## 4. X-CUBE-AI (≥ 10.0) — le compilateur pour le NPU

C'est le pack qui contient **ST Edge AI Core** et le **compilateur Neural-ART**, celui
qui transforme un `.tflite` en code pour le NPU du N6.

Dans CubeMX : **Help → Manage embedded software packages → STMicroelectronics →
X-CUBE-AI** → coche la dernière version (≥ 10.0.0) → **Install Now**.

> Attention au vocabulaire : X-CUBE-AI ≥ 10 est bâti sur ST Edge AI Core. Sur un STM32
> classique il génère des noyaux logiciels ; sur le N6 il invoque le compilateur
> Neural-ART. Même pack, deux comportements — la doc que tu trouves en ligne parle
> souvent de l'un en croyant parler de l'autre.

### Vérification

Le pack s'installe sous
`C:\Users\<toi>\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\<version>\`.
Tu dois y trouver un dossier `Utilities\windows\` contenant `stedgeai.exe`.

```powershell
& "$env:USERPROFILE\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\<version>\Utilities\windows\stedgeai.exe" --version
& "...\stedgeai.exe" --tools-version
```

Si `stedgeai` répond, la partie IA est prête. (Le chemin exact varie selon la version ;
`scripts\check-toolchain.ps1` le cherche pour toi.)

**Alternative** : ST Edge AI Core existe aussi en installeur autonome
(https://www.st.com/en/development-tools/stedgeai-core.html), utile si tu veux compiler
des modèles sans ouvrir CubeMX. Les deux peuvent coexister ; assure-toi juste de savoir
lequel tu appelles.

---

## 5. Le package firmware STM32Cube_FW_N6

Les drivers HAL, les fichiers de démarrage, les exemples et le **FSBL** de référence.

Le plus simple : ne l'installe pas à la main. Quand tu créeras le projet
([doc 3](03-projet-cubemx.md)) en sélectionnant la carte NUCLEO-N657X0-Q, CubeMX
proposera de le télécharger. Accepte.

---

## 6. Visual Studio Code

Si ce n'est pas déjà fait : https://code.visualstudio.com

La configuration des extensions est traitée dans [le document 2](02-vscode.md).

---

## 7. L'environnement Python d'EMaGerLib

Nécessaire pour produire le modèle. À faire dans ton clone d'EMaGerLib :

```powershell
pip install -e ".[deploy]"
```

### Vérification

```powershell
python examples\deployment\export_model.py --help
```

---

## Vérification globale

Le script fourni contrôle tout d'un coup et te dit précisément ce qui manque :

```powershell
cd stm32n6-firmware
powershell -ExecutionPolicy Bypass -File scripts\check-toolchain.ps1
```

Il vérifie la présence et la version de chaque outil, trouve `stedgeai`, liste les
*external loaders* disponibles pour la flash OSPI, et tente une connexion à la carte.

Quand tout est vert, passe au [document 2](02-vscode.md).

---

## Récapitulatif

| Outil | Version min. | Pourquoi |
|---|---|---|
| STM32CubeProgrammer | 2.18 | driver ST-LINK, flash OSPI, `STM32_SigningTool_CLI` |
| STM32CubeMX | dernière | génère le projet, installe X-CUBE-AI |
| STM32CubeCLT | 1.18 | gcc, gdb, CMake, Ninja, serveur GDB ST-LINK |
| X-CUBE-AI | 10.0 | ST Edge AI Core + compilateur Neural-ART |
| STM32Cube_FW_N6 | — | HAL, FSBL de référence (via CubeMX) |
| VS Code | — | l'éditeur |
| Python + `.[deploy]` | — | produit le `.tflite` depuis EMaGerLib |
