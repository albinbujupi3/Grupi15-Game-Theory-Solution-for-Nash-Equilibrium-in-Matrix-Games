# 🎯 Nash Equilibrium Solver – Matrix Games

### Game Theory Solution for Nash Equilibrium in Matrix Games  
**Grupi 15 – Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike**



## 📘 Përshkrimi i Projektit

Ky projekt implementon një sistem të plotë për gjetjen e ekuilibrit të Nash-it në lojëra me matrica (2 lojtarë, me strategji të fundme). Sistemi përfshin:

### Metoda të ndryshme zgjidhjeje:
- ✔ **Fictitious Play** (metodë iterative)
- ✔ **Linear Programming (LP)** për lojëra zero-sum
- ✔ **Kontrollues të devijimeve të njëanshme** për verifikimin e ekuilibrit (equilibrium checker)

### Funksionalitete të tjera:
- GUI interaktiv me Tkinter
- CLI (Command Line Interface)
- Alexa tests (pytest) për siguri dhe korrektësi

Ky projekt është i dizajnuar për përdorim akademik dhe demonstrim të koncepteve të teorisë së lojërave.


## 📂 Struktura e Projektit

```

.
├── core/
│   ├── equilibrium_checker.py
│   ├── payoff_matrix.py
│   ├── solver_iterative.py
│   ├── solver_lp.py
├── utils/
│   ├── matrix_examples.py
│   ├── matrix_loader.py
│   ├── plotting.py
├── tests/
│   ├── test_checker.py
│   ├── test_iterative.py
│   ├── test_lp.py
├── gui.py
├── main.py
└── README.md

```



## 🚀 Si të ekzekutohet projekti

### 1️⃣ Instalimi i varësive
Në terminal:

```

pip install -r requirements.txt

```

Varësitë kryesore:
- numpy
- scipy
- matplotlib
- tkinter (vjen me Python)



### 2️⃣ Ekzekutimi i CLI
Për të lëshuar solver-in për një shembull të gatshëm:

```

python main.py --example prisoners_dilemma

```

Për të zgjedhur metodën:

```

python main.py --method lp   # për zero-sum me Linear Programming
python main.py --method fp   # fictitious play

```



## 🧠 Algoritmet e implementuara

### ✔ Fictitious Play (Iterative Solver)
Simulon mësimin iterativ mes lojtarëve dhe konvergon te ekuilibri te shumë lojëra jo-zero-sum.


### ✔ Linear Programming Solver (Zero-Sum)
Gjen strategjinë optimale të lojtarëve në lojëra zero-sum duke përdorur formën kanonike të LP.



### ✔ Equilibrium Checker
Verifikon nëse një çift strategjish (p, q) është Nash duke llogaritur devijimin e mundshëm të njëanshëm.



## 🧪 Testet (Pytest)

Projekti ka një paketë të plotë testimesh:

### ▶️ Testet LP
Verifikojnë se solver-i LP prodhon strategji optimale për *matching pennies*.


### ▶️ Testet Iterative
Kontrollojnë nëse *fictitious play* konvergon te strategjitë prite për *Prisoner’s Dilemma*.



### ▶️ Testet Checker
Sigurojnë që *equilibrium_checker* identifikon korrekt strategjitë që janë ose nuk janë NE.

### Ekzekutimi i testeve:

```

pytest -v

```



## 👥 Autorët

Grupi 15 – UP FIEK 2025

- **Albin Bujupi**
- **Enes Spahiu**
- **Dion Haradinaj**

