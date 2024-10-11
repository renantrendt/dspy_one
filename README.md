# DSPy - this code make chatgpt ask the same question for all chunks and at the end create a cohesive answer with source citation (file + page)
1. run python3 -m venv dspy_env
2. run source dspy_env/bin/activate
3. Choose this python workspace Python 3.12.4 or less
4. run pip install requirements.txt
6. create a file .env with the content: API_KEY=
7. drop files inside the folder data
8. Ask your question on the line 82 of file main.py 
9. run python3 main.py
10. optional you can change the size of the output line 43
