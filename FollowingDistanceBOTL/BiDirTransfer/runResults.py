import subprocess
import sys

numSs = [int(sys.argv[2])]#,int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])]
port = sys.argv[1]

for i in range(1,2):
    #subprocess.call(['python3', 'controller.py',str(i),str(numS1),str(port),str(weightType),str(cullThresh)])
    #subprocess.call(['python3', 'controller.py',str(i),str(numS2),str(port),str(weightType),str(cullThresh)])
    for numS1 in numSs:
        subprocess.call(['python3', 'controller.py',str(i),str(numS1),str(port),str('OLS'),str(0.0),str(-1)])
        subprocess.call(['python3', 'controller.py',str(i),str(numS1),str(port),str('OLSFE'),str(0.0),str(-1)])
        subprocess.call(['python3', 'controller.py',str(i),str(numS1),str(port),str('OLSFEMI'),str(0.2),str(0.95)])

