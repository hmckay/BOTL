import subprocess
import sys
from optparse import OptionParser

DEFAULT_PRED=0
CD_TYPE=0
MAX_WINDOW=0
MODEL_HIST_THRESHOLD_ACC=0
MODEL_HIST_THRESHOLD_PROB=0
runID=0
numStreams=0
ADWIN_DELTA=0
socketOffset=0
weightType=0
CThresh=0
MThresh=0
PAThresh=0
PCAVar=0
LEARNER_TYPE = ''
CONNECTIVITY = 0
REPLACEMENT=0

def main():
    global DEFAULT_PRED
    global CD_TYPE
    global MAX_WINDOW
    global MODEL_HIST_THRESHOLD_ACC
    global MODEL_HIST_THRESHOLD_PROB
    global runID
    global numStreams
    global ADWIN_DELTA
    global socketOffset
    global weightType
    global CThresh
    global MThresh
    global PAThresh
    global PCAVar
    global LEARNER_TYPE
    global CONNECTIVITY
    global REPLACEMENT

    parser = OptionParser(usage="usage: prog options",version="BOTL v2.0")
    parser.add_option("-d","--domain",default = "Following",dest="DEFAULT_PRED",help="domain: Following, Heating, Sudden, Gradual")
    parser.add_option("-t","--type",default = "RePro",dest= "CD_TYPE",help="Concept Drift Type: RePro, ADWIN, AWPro")
    parser.add_option("-w","--window",default = "90",dest="MAX_WINDOW",help="Window size (default = 90)")
    parser.add_option("-r","--ReProAcc",default = "0.5",dest="MODEL_HIST_THRESHOLD_ACC",help="RePro drift threshold")
    parser.add_option("-p","--ReProProb",default = "0.5",dest="MODEL_HIST_THRESHOLD_PROB",help="RePro recur prob")
    parser.add_option("-i","--runid",default = "1",dest="runID",help="RunID")
    parser.add_option("-n","--numStreams",default = "1",dest="numStreams",help="Number of streams")
    parser.add_option("-z","--ADWINDelta",default = "0.02",dest="ADWIN_DELTA",help="ADWIN confidence value")
    # parser.add_option("-e","--ReProThresh",default = "0.1",dest="THRESHOLD",help="RePro error threshold")
    parser.add_option("-s","--socket",default = "3000",dest="socketOffset",help="Socket Offset")
    parser.add_option("-e","--ensemble",default = "OLS",dest="weightType",help="Weight Type (OLS, OLSFE, OLSFEMI,...)")
    parser.add_option("-c","--perfCull",default = "0.0",dest="CThresh",help="Performance culling parameter")
    parser.add_option("-m","--miCull",default = "2",dest="MThresh",help="Mutual Information culling parameter")
    parser.add_option("-a","--paCull",default = "2",dest="PAThresh",help="Principal Angle culling parameter")
    parser.add_option("-v","--variance",default = "0.05",dest="PCAVar",help="Keep prinicpal components that capture this uch variance")
    parser.add_option("-l","--learner",default="SVR",dest="learnerType",help="Type of model used by the underlying CDD")
    parser.add_option("--connectivity",default="0",dest="connectivity",help="How connected domains are, defailt = 0 means fully connected")
    parser.add_option("--replacement",default="0",dest="replacement",help="If domains can replace models with better conceptuall similar models. No=0, Yes=1")

    # thresh = [0.5]
    (options,args) = parser.parse_args()
    
    DEFAULT_PRED=str(options.DEFAULT_PRED)
    CD_TYPE=str(options.CD_TYPE)
    MAX_WINDOW=str(options.MAX_WINDOW)
    MODEL_HIST_THRESHOLD_ACC=str(options.MODEL_HIST_THRESHOLD_ACC)
    MODEL_HIST_THRESHOLD_PROB=str(options.MODEL_HIST_THRESHOLD_PROB)
    runID=str(options.runID)
    numStreams=str(options.numStreams)
    ADWIN_DELTA=str(options.ADWIN_DELTA)
    socketOffset=str(options.socketOffset)
    weightType=str(options.weightType)
    CThresh=str(options.CThresh)
    MThresh=str(options.MThresh)
    PAThresh=str(options.PAThresh)
    PCAVar=str(options.PCAVar)
    LEARNER_TYPE = str(options.learnerType)
    CONNECTIVITY = str(options.connectivity)
    REPLACEMENT = str(options.replacement)

    MThresh = 0.2
    CThresh = 0.2
    PAThresh = 0.4
    for i in range(1,11):#31):
        args = ["python3","controller2.py",
                "--domain",str(DEFAULT_PRED),
                "--type",str(CD_TYPE),
                "--learner",str(LEARNER_TYPE),
                "--window",str(MAX_WINDOW),
                "--ReProAcc",str(MODEL_HIST_THRESHOLD_ACC),
                "--ReProProb",str(MODEL_HIST_THRESHOLD_PROB),
                "--runid",str(i),
                "--numStreams",str(numStreams),
                "--ADWINDelta",str(ADWIN_DELTA),
                "--socket",str(int(socketOffset)),#+20*i),
                "--ensemble",str(weightType),
                "--perfCull",str(CThresh),
                "--miCull",str(MThresh),
                "--paCull",str(PAThresh),
                # "--perfCull",str(CThresh),
                # "--miCull",str(MThresh),
                # "--paCull",str(PAThresh),
                "--variance",str(PCAVar),
                "--connectivity",str(CONNECTIVITY),
                "--replace",str(REPLACEMENT)]
        subprocess.call(args)



if __name__ == '__main__':main()


