import os

os.system("python3 data/ivf.py")
for i in range(50):
    os.system("python3 data/rabitq.py")
    os.system("python3 script/index_batch.py")
    os.system("python3 script/search_batch.py")
