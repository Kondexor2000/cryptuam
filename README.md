# AI Developer Portfolio

This repository contains a portfolio of four independent projects developed for educational purposes at UAM (Adam Mickiewicz University) in Poznań, Poland. Each project demonstrates different aspects of Python programming, machine learning, and data processing.

## Projects

### Project 1: ECG Arrhythmia Detection
A machine learning project for detecting arrhythmias in ECG signals using Random Forest classification.

**Files:** cardio.py, secure_backup.py, cardio_tests.py, backend.py

**Installation:**
```bash
git clone https://github.com/Kondexor2000/cryptuam.git
cd cryptuam
# Copy cardio.py, secure_backup.py, cardio_tests.py to your project directory
```

**Run:**
```bash
python cardio_tests.py  # Run tests
python cardio.py
python secure_backup.py
python backend.py       # Start API backend
```

**HTML test panel:**
```bash
http://127.0.0.1:5000/
```

**API:**
```bash
POST /api/project1/ecg/predict
Body: {"beat": [0.1, 0.2, ... 200 samples total]}
```

### Project 2: Document-Based Chatbot
An AI-powered chatbot that answers questions based on indexed documents using FAISS and transformers.

**Files:** index.py, chat.py, chat_tests.py

**Installation:**
```bash
git clone https://github.com/Kondexor2000/cryptuam.git
cd cryptuam
# Copy docs, docs_not, index.py, chat.py, chat_tests.py to your project directory
```

**Run:**
```bash
python index.py  # Index documents
python chat_tests.py  # Run tests
python chat.py   # Start chatbot
```

### Project 3: Syllabus Chatbot
A specialized chatbot for educational content, with secure backup functionality.

**Files:** index_sylabus.py, rczar.py, chat_sylabus.py, chat_sylabus_tests.py, backend.py

**Installation:**
```bash
git clone https://github.com/Kondexor2000/cryptuam.git
cd cryptuam
# Copy sylabus, index_sylabus.py, rczar.py, chat_sylabus.py, chat_sylabus_tests.py to your project directory
```

**Run:**
```bash
python index_sylabus.py  # Index syllabus documents
python chat_sylabus_tests.py  # Run tests
python chat_sylabus.py   # Start syllabus chatbot
python backend.py        # Start API backend
```

**HTML test panel:**
```bash
http://127.0.0.1:5000/
```

**API:**
```bash
POST /api/project3/syllabus/ask
Body: {"question": "Jakie sa efekty uczenia sie?"}
```

### Project 4: Energy Consumption Simulator
A simulation tool for predicting energy usage in devices using machine learning regression.

**Files:** energy.py, energy_tests.py, backend.py

**Installation:**
```bash
git clone https://github.com/Kondexor2000/cryptuam.git
cd cryptuam
# Copy energy.py, energy_tests.py to your project directory
```

**Run:**
```bash
python energy_tests.py  # Run tests
python energy.py
python backend.py       # Start API backend
```

**HTML test panel:**
```bash
http://127.0.0.1:5000/
```

**API:**
```bash
POST /api/project4/energy/predict
Body: {"devices": 5, "hours": 10, "efficiency": 0.6, "eco_mode": 0}
```

## Tech Stack

- Python
- Libraries: NumPy, Pandas, Scikit-learn, TensorFlow, Transformers, FAISS, wfdb, matplotlib, cryptography, pqcrypto

## Contact

- Email: k.kosciecha20@gmail.com

