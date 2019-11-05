import subprocess
import sys

numSs = [int(sys.argv[2])]#,int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])]
port = sys.argv[1]
numS1 = numSs[0]

windowLens = [14]#[5,7,10,14,18,21,30]#,7,9,12,14,18,21,25,30]#,120,150,180,200]
adwinDeltas = [0.02]#[0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]

for windowLen in windowLens:
    windowLen = windowLen *48
    for adDelt in adwinDeltas:
        subprocess.call(['python3', 'controller.py',str(1),str(numS1),str(port),str('OLS'),str(0.0),str(-1),str(windowLen),str(adDelt),str(0.5),str(0.5)])
        subprocess.call(['python3', 'controller.py',str(1),str(numS1),str(port),str('OLSFE'),str(0.0),str(-1),str(windowLen),str(adDelt),str(0.5),str(0.5)])
        subprocess.call(['python3', 'controller.py',str(1),str(numS1),str(port),str('OLSFEMI'),str(0.2),str(0.95),str(windowLen),str(adDelt),str(0.5),str(0.5)])

