import controller
import sys
import subprocess

# numS1 = sys.argv[1]
# numS2= sys.argv[3]
# port = sys.argv[2]
# weightType = str(sys.argv[3])
# cullThresh = float(sys.argv[4])
def run():
    startDates = {2000: "2014-01-01", 2001:"2015-01-01", 2002:"2014-09-01",2003:"2015-01-01",2004:"2014-01-01"}
    endDates = {2000: "2015-03-31",2001:"2015-12-31",2002:"2015-03-30",2003:"2015-09-30",2004:"2015-06-30"}
    subprocess.call(['python', 'controller.py',str('OLS'),str(1)])
    subprocess.call(['python', 'controller.py',str('OLSFE'),str(2)])
    subprocess.call(['python', 'controller.py',str('OLSFEMI'),str(3)])
    # controller.mainrun(startDates,endDates,'OLS')
    # startDates = {3000: "2014-01-01", 3001:"2015-01-01", 3002:"2014-09-01",3003:"2015-01-01",3004:"2014-01-01"}
    # endDates = {3000: "2015-03-31",3001:"2015-12-31",3002:"2015-03-30",3003:"2015-09-30",3004:"2015-06-30"}
    # controller.mainrun(startDates,endDates,'OLSFE')
    # startDates = {4000: "2014-01-01", 4001:"2015-01-01", 4002:"2014-09-01",4003:"2015-01-01",4004:"2014-01-01"}
    # endDates = {4000: "2015-03-31",4001:"2015-12-31",4002:"2015-03-30",4003:"2015-09-30",4004:"2015-06-30"}
    # controller.mainrun(startDates,endDates,'OLSFEMI')




def main():
    for i in range(1,2):
        run()






if __name__ == '__main__':main()
# for i in range(1,31):

    # subprocess.call(['python', 'controller.py',str(i),str(numS1),str(port),str('OLSFEMI'),str(0.4),str(0.65)])
    # # subprocess.call(['python', 'controller.py',str(i),str(numS2),str(port),str('OLSFEMI'),str(0.4),str(0.65)])

