#!/bin/bash
echo "Face Swap 설치를 시작합니다..."
python -m pip install -r requirements.txt
python download_models.py
echo "설치가 완료되었습니다!"

# setup.bat (Windows 용)
@echo off
echo Face Swap 설치를 시작합니다...
python -m pip install -r requirements.txt
python download_models.py
echo 설치가 완료되었습니다!
