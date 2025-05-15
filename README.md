# 🛡️ Quantum Kernel 기반 NSL-KDD 침입 탐지

이 프로젝트는 Qiskit의 QuantumKernel과 QSVC를 활용하여 NSL-KDD 네트워크 데이터셋 상에서 사이버 공격 탐지 모델을 구현합니다.

## 📁 구성
- `data/`: NSL-KDD 데이터셋
- `notebooks/`: 실험용 Jupyter 노트북
- `src/`: 핵심 코드 (전처리, 커널, 모델)
- `results/`: 실험 결과

## 🛠️ 실행
```bash
pip install -r requirements.txt
python run_pipeline.py

