import subprocess
import sys

numSs = [int(sys.argv[2])]#,int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])]
port = sys.argv[1]
numS1 = numSs[0]

windowLens = [80]#[40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]
adwinParams = [0.02]#[0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]

for windowLen in windowLens:
    for adwinParam in adwinParams:
        subprocess.call(['python3', 'controller.py',str(6),str(numS1),str(port),str('OLS'),str(0.0),str(-1),str(windowLen),str(adwinParam)])
        subprocess.call(['python3', 'controller.py',str(6),str(numS1),str(port),str('OLSFE'),str(0.0),str(-1),str(windowLen),str(adwinParam)])
        subprocess.call(['python3', 'controller.py',str(6),str(numS1),str(port),str('OLSFEMI'),str(0.2),str(0.95),str(windowLen),str(adwinParam)])
