hchessai ♟️
hchessai (Human Chess AI) is an experimental UCI-compatible chess engine built around a trained large-scale neural network model. The goal of this project is not only playing strength, but human-like decision making and style.
________________________________________
🚀 Project Goals
•	Build a neural-network-based chess engine from scratch
•	Support full UCI protocol compatibility
•	Explore human-like play patterns instead of purely optimal moves
•	Provide a flexible framework for experimentation with:
o	Vision Transformers (ViT)
o	Policy + Value networks
o	Learned evaluation instead of handcrafted heuristics
________________________________________
🧠 Architecture Overview
The engine consists of the following core components:
•	Model: Vision Transformer trained on chess positions
•	Policy Head: Predicts move probabilities
•	Value Head: Estimates position evaluation
•	Inference Engine: Selects moves based on model output
•	UCI Interface: Communicates with chess GUIs
________________________________________
📦 Current Features
•	✅ Trained neural network model (PyTorch)
•	✅ Working Python-based UCI engine
•	✅ Checkpoint loading (.pt files)
•	🚧 EXE packaging (in progress)
•	🚧 Search improvements (planned)
________________________________________
📂 Project Structure
project/
├─ engine/
│  ├─ main.py
│  ├─ uci_loop.py
│  ├─ model.py
│  ├─ search.py
│  └─ utils/
├─ checkpoints/
│  └─ chess_vit_latest.pt
├─ tools/
│  └─ smoke_test.py
├─ build/
├─ dist/
└─ README.md
________________________________________
▶️ Running the Engine (Python)
python engine/main.py
Then interact via UCI:
uci
isready
position startpos
go movetime 100
quit
________________________________________
🧪 Testing
Run a simple smoke test:
python tools/smoke_test.py
The engine should: - respond to UCI commands - return a legal bestmove - not crash
________________________________________
🏗️ Building the EXE
Using PyInstaller:
pyinstaller \
  --onedir \
  --name hchessai \
  --add-data "checkpoints/chess_vit_latest.pt;checkpoints" \
  --collect-all torch \
  engine/main.py
Output:
dist/hchessai/hchessai.exe
________________________________________
⚙️ UCI Options (Planned)
•	ModelPath
•	Threads
•	MoveOverhead
•	UseGPU
•	Deterministic
________________________________________
📚 References & Inspiration
This project builds upon ideas and tooling from the following open-source work:
•	AlphaZero-style training concepts
•	Leela Chess Zero (LC0)
•	PyTorch ecosystem
•	Vision Transformer research
(Exact repositories and citations will be added as the project stabilizes.)
________________________________________
🧭 Roadmap
☐	Stable inference pipeline
☐	Stronger move selection / search
☐	Reliable EXE build
☐	Performance optimization
☐	Human-style tuning (blunders, intuition, etc.)
________________________________________
⚠️ Disclaimer
This is an experimental research project. The engine may:
•	play weak or inconsistent moves
•	crash in edge cases
•	behave non-deterministically
Use for experimentation and development purposes.
________________________________________
🧑‍💻 Author
Created as part of an independent AI + chess engine project.
________________________________________
📜 License
TBD
