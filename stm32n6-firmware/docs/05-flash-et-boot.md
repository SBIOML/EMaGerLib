# 5. Boot, signature et flash

Le sujet où le N6 ne ressemble à aucun autre STM32. À lire en entier avant la première
programmation de la flash externe.

---

## Pourquoi c'est différent

Le STM32N6 **n'a pas de flash interne**. Il possède beaucoup de RAM (plusieurs Mo) et
une interface OSPI vers une flash externe. Conséquences en chaîne :

- au démarrage, la ROM de boot **copie** une image depuis la flash externe vers la RAM
  interne, puis l'exécute ;
- cette image doit être **signée**, sinon la ROM la refuse ;
- programmer la flash externe demande un **external loader** (`.stldr`), un petit
  programme que STM32CubeProgrammer charge en RAM pour piloter l'OSPI ;
- et un **cavalier** choisit entre « ouvrir le port de debug » et « démarrer depuis la
  flash ».

Aucune de ces étapes n'existe sur un F4 ou un H7.

---

## Les deux modes de boot

La carte a deux cavaliers, **JP1** et **JP2**, câblés sur BOOT0 et BOOT1.

| Mode | Quand l'utiliser |
|---|---|
| **Development boot** | développement, debug, **et toute programmation de la flash** |
| **Boot from flash** | la carte démarre seule sur ton firmware |

> ⚠️ **Vérifie la correspondance exacte cavalier ↔ position sur ta carte** : la
> sérigraphie du PCB et le manuel utilisateur de la carte (UM du NUCLEO-N657X0-Q) font
> foi. Les sources en ligne se contredisent sur le sens, et une inversion se voit
> immédiatement (aucune connexion, ou aucun démarrage) sans rien casser. Note la bonne
> combinaison ici une fois que tu l'as établie :
>
> ```
> Development boot : JP1 = ____   JP2 = ____
> Boot from flash  : JP1 = ____   JP2 = ____
> ```

Le réflexe à prendre : **avant de flasher, mets la carte en Development boot**, et
appuie sur RESET après avoir changé un cavalier. Un changement de cavalier n'a aucun
effet tant qu'il n'y a pas eu de reset ou de coupure d'alimentation.

---

## Le FSBL

En boot depuis la flash, la ROM charge d'abord un **FSBL** (first stage boot loader),
qui initialise l'OSPI et les horloges puis lance ton application.

Les adresses conventionnelles en flash externe :

| Élément | Adresse |
|---|---|
| FSBL | `0x70000000` |
| Application signée | `0x70100000` |

Le FSBL n'est pas à écrire toi-même : le package `STM32Cube_FW_N6` et les projets
*getting started* de ST en fournissent un utilisable tel quel. Tu le programmes **une
fois** ; ensuite tu ne reprogrammes que l'application.

---

## Signer

Sans signature, la ROM refuse l'image et la carte ne démarre pas — silencieusement.

`STM32_SigningTool_CLI` est livré avec STM32CubeProgrammer :

```powershell
STM32_SigningTool_CLI -bin build\emager-n6.bin -nk -t ssbl -hv 2.3 -o build\emager-n6_signed.bin
```

| Option | Sens |
|---|---|
| `-nk` | *no key* — pas de clé, signature factice pour le développement |
| `-t ssbl` | type d'image : *second stage boot loader*, c'est-à-dire l'application |
| `-hv 2.3` | version d'en-tête |

> `-nk` convient au développement. Pour un produit réel, la chaîne de boot sécurisée du
> N6 attend de vraies clés ECC — un autre sujet, à ne pas traiter à la fin d'un projet.
>
> `-hv 2.3` est la valeur employée par les exemples ST actuels. Si l'outil la refuse,
> `STM32_SigningTool_CLI --help` liste les versions acceptées par ta version.

---

## Programmer

Il faut l'external loader correspondant à la flash de la carte. Sur le
NUCLEO-N657X0-Q :

```
MX25UM51245G_STM32N6570-NUCLEO.stldr
```

livré dans `...\STM32CubeProgrammer\bin\ExternalLoader\`.

```powershell
$EL = "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\ExternalLoader\MX25UM51245G_STM32N6570-NUCLEO.stldr"

# le FSBL, une seule fois
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el "$EL" -hardRst -w fsbl_signed.bin 0x70000000

# l'application, à chaque itération
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el "$EL" -hardRst -w build\emager-n6_signed.bin 0x70100000
```

`scripts\sign-and-flash.ps1` enchaîne signature et programmation, et refuse de
continuer si le fichier signé n'a pas été régénéré — l'erreur classique étant de
reflasher l'image précédente après avoir recompilé.

Vérifie que le loader présent sur ta machine porte bien ce nom : les références en
ligne mentionnent aussi un `MX66UW1G45G`, qui correspond à une autre carte N6. Liste
le dossier plutôt que de deviner :

```powershell
Get-ChildItem "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\ExternalLoader\*N6*"
```

---

## Le cycle de développement

Ne signe et ne flashe pas à chaque compilation. La boucle rapide :

1. **Development boot**, F5 dans VS Code → chargement en RAM, quelques secondes.
2. Itère.
3. Quand le comportement est bon : signe, flashe, bascule en **boot from flash**,
   RESET, et vérifie que la carte démarre **seule**.

L'étape 3 n'est pas facultative. Un firmware qui fonctionne chargé en RAM par le
debugger peut très bien ne pas démarrer depuis la flash — initialisation d'horloge
différente, dépendance à un état laissé par le debugger, image mal signée. Le découvrir
le jour de la démonstration est un grand classique.

---

## Les prototypes de l'utilisateur

`emager_proto.c` permet de recalibrer le modèle sur la carte ; le résultat est un bloc
d'environ 1,8 ko qu'il faut conserver entre deux mises sous tension. Sur le N6, ce bloc
ira dans la **même flash externe** que le firmware, ce qui crée deux pièges :

1. **Reflasher le firmware peut effacer les prototypes.** Réserve-leur un secteur
   dédié, loin de la zone applicative, et n'efface pas la puce entière (`-e all`) par
   habitude.
2. **Des prototypes périmés avec un nouveau réseau échouent silencieusement** — pas
   d'erreur, juste de mauvaises prédictions. Stocke une version ou un hash de
   `emager_model_data.h` à côté du bloc et jette le bloc s'il ne correspond pas.

Le second point est le plus vicieux : rien ne signale la panne. Traite-le en même temps
que tu écris la persistance, pas après.

En cas de problème : [dépannage](06-depannage.md).
