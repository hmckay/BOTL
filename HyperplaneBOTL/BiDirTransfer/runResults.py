import subprocess
import sys

numSs = [int(sys.argv[2])]#,int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])]
port = sys.argv[1]

for i in range(11,12):
    #subprocess.call(['python3', 'controller.py',str(i),str(numS1),str(port),str(weightType),str(cullThresh)])
    #subprocess.call(['python3', 'controller.py',str(i),str(numS2),str(port),str(weightType),str(cullThresh)])
    for numS1 in numSs:
        # subprocess.call(['python3', 'controller.py',str(13),str(numS1),str(port),str('OLSKPAC'),str(0.0),str(-1)])
        # subprocess.call(['python3', 'controller.py',str(16),str(numS1),str(port),str('OLSPAC2'),str(0.0),str(-1)])
        # subprocess.call(['python3', 'controller.py',str(16),str(numS1),str(port),str('OLSKPAC2'),str(0.0),str(-1)])
        # subprocess.call(['python3', 'controller.py',str(13),str(numS1),str(port),str('OLSPAC'),str(0.0),str(-1)])
        subprocess.call(['python3', 'controller.py',str(1),str(numS1),str(port),str('OLSCL2'),str(0.0),str(-1),str('Sudden')])
        subprocess.call(['python3', 'controller.py',str(1),str(numS1),str(port),str('OLSCL'),str(0.0),str(-1),str('Sudden')])
        # subprocess.call(['python3', 'controller.py',str(1),str(numS1),str(port),str('OLS'),str(0.0),str(-1),str('Sudden')])
        # subprocess.call(['python3', 'controller.py',str(1),str(numS1),str(port),str('OLSFE'),str(0.0),str(-1),str('Sudden')])
        # subprocess.call(['python3', 'controller.py',str(1),str(numS1),str(port),str('OLSFEMI'),str(0.2),str(0.95),str('Sudden')])

