
import subprocess
import sys

numSs = [int(sys.argv[2])]#,int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])]
port = sys.argv[1]
dT = str(sys.argv[3])
numS1 = numSs[0]

windowLens = [50]#[20,30,40,50,60,70,80,90,100,120,150,180,200]
thresholds = [0.7]#[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]

for windowLen in windowLens:
    for thresh in thresholds:
        subprocess.call(['python3', 'controller.py',str(1),str(numS1),str(port),str('OLS'),str(0.0),str(-1),str(dT),str(windowLen),str(thresh)])
        subprocess.call(['python3', 'controller.py',str(1),str(numS1),str(int(port)+300),str('OLSFE'),str(0.0),str(-1),str(dT),str(windowLen),str(thresh)])
        subprocess.call(['python3', 'controller.py',str(1),str(numS1),str(int(port)+600),str('OLSFEMI'),str(0.2),str(0.95),str(dT),str(windowLen),str(thresh)])

