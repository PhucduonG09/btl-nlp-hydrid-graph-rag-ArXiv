"""Shared blind review entrypoint; no assistant suggestions or model API calls."""
import os,runpy
from pathlib import Path
os.environ['REVIEW_SHARED_MODE']='1'
runpy.run_path(str(Path(__file__).resolve().parent/'review_answers.py'),run_name='__main__')
