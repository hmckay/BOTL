import sys
import subprocess

# numS1 = sys.argv[1]
# numS2= sys.argv[3]
# port = sys.argv[2]
# weightType = str(sys.argv[3])
# cullThresh = float(sys.argv[4])
def run(ident,driftType,offset):
    subprocess.call(['python', 'controller.py',str(ident),str(driftType),str(offset),str('OLS')])
    subprocess.call(['python', 'controller.py',str(ident),str(driftType),str(offset+300),str('OLSFE')])
    subprocess.call(['python', 'controller.py',str(ident),str(driftType),str(offset+600),str('OLSFEMI')])
    #controller.mainrun(sourceFPs,'OLSFE')
    #controller.mainrun(sourceFPs,'OLSFEMI')

def getFPs(ident,driftType,offset):
    s = dict()
    FPs = dict()
    tos = dict()
    froms = dict()
    for i in range(1,6):
        s[(i+offset)]='TARGET'+str(i)+str(driftType)+str(ident+1)
        FPs[(i+offset)]='../../HyperplaneData/KDDDatasets/Datastreams/TARGET'+str(i)+'MultiConcept'+str(driftType)+str(ident+1)+'.csv'
        froms[(i+offset)]=1
        if driftType == 'Sudden':
            tos[(i+offset)]=10000
        else:
            tos[(i+offset)]=11900
    return s, FPs, froms, tos



def main():
    # drift = str(sys.argv[1])
    drift = ['Sudden']#,'Gradual','GradualRand']

    for d in drift:
        for ident in range(0,1):
            run(ident,d,(2000+(ident*1000)))
            #run(0.6,s,sourceFPs,Froms,Tos)






if __name__ == '__main__':main()
# for i in range(1,31):

    # subprocess.call(['python', 'controller.py',str(i),str(numS1),str(port),str('OLSFEMI'),str(0.4),str(0.65)])
    # # subprocess.call(['python', 'controller.py',str(i),str(numS2),str(port),str('OLSFEMI'),str(0.4),str(0.65)])

