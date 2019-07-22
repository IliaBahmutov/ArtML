import numpy #for numpy storage
import os #to find files
import time #for time to complete
import sklearn
import pickle

#Load Test Data

running = True
selection = 1

while running:

    if (selection == 21):
        print ("done")
        exit()


    elif (selection == 1):
        testdata = numpy.load("Data/UkiORBTestingDataV.npy")
        testdata = testdata.reshape(173,(10*256))
        correctnum = 0 
        results = numpy.zeros(shape = (20,173))

    elif (selection == 2):
        testdata = numpy.load("Data/LReORBTestingDataV.npy")
        testdata = testdata.reshape(172,(10*256))
        correctnum = 1 
        results = numpy.zeros(shape = (20,172))

    elif (selection == 3):
        testdata = numpy.load("Data/MinORBTestingDataV.npy")
        testdata = testdata.reshape(89,(10*256))
        correctnum = 2
        results = numpy.zeros(shape = (20,89))

    elif (selection == 4):
        testdata = numpy.load("Data/HReORBTestingDataV.npy")
        testdata = testdata.reshape(186,(10*256))
        correctnum = 3 
        results = numpy.zeros(shape = (20,186))

    elif (selection == 5):
        testdata = numpy.load("Data/EreORBTestingDataV.npy")
        testdata = testdata.reshape(197,(10*256))
        correctnum = 4
        results = numpy.zeros(shape = (20,197))

    elif (selection == 6):
        testdata = numpy.load("Data/PopORBTestingDataV.npy")
        testdata = testdata.reshape(208,(10*256))
        correctnum = 5
        results = numpy.zeros(shape = (20,208))

    elif (selection == 7):
        testdata = numpy.load("Data/CFPORBTestingDataV.npy")
        testdata = testdata.reshape(92,(10*256))
        correctnum = 6
        results = numpy.zeros(shape = (20,92))

    elif (selection == 8):
        testdata = numpy.load("Data/RocORBTestingDataV.npy")
        testdata = testdata.reshape(282,(10*256))
        correctnum = 7 
        results = numpy.zeros(shape = (20,282))

    elif (selection == 9):
        testdata = numpy.load("Data/CubORBTestingDataV.npy")
        testdata = testdata.reshape(317,(10*256))
        correctnum = 8
        results = numpy.zeros(shape = (20,317))

    elif (selection == 10):
        testdata = numpy.load("Data/NAPORBTestingDataV.npy")
        testdata = testdata.reshape(355,(10*256))
        correctnum = 9
        results = numpy.zeros(shape = (20,355))

    elif (selection == 11):
        testdata = numpy.load("Data/NreORBTestingDataV.npy")
        testdata = testdata.reshape(358,(10*256))
        correctnum = 10
        results = numpy.zeros(shape = (20,358))

    elif (selection == 12):
        testdata = numpy.load("Data/AExORBTestingDataV.npy")
        testdata = testdata.reshape(380,(10*256))
        correctnum = 11
        results = numpy.zeros(shape = (20,380))

    elif (selection == 13):
        testdata = numpy.load("Data/BORBTestingDataV.npy")
        testdata = testdata.reshape(589,(10*256))
        correctnum = 12
        results = numpy.zeros(shape = (20,589))

    elif (selection == 14):
        testdata = numpy.load("Data/ANMORBTestingDataV.npy")
        testdata = testdata.reshape(633,(10*256))
        correctnum = 13
        results = numpy.zeros(shape = (20,633))

    elif (selection == 15):
        testdata = numpy.load("Data/SymORBTestingDataV.npy")
        testdata = testdata.reshape(637,(10*256))
        correctnum = 14
        results = numpy.zeros(shape = (20,637))

    elif (selection == 16):

        testdata = numpy.load("Data/PImORBTestingDataV.npy")
        testdata = testdata.reshape(946,(10*256))
        correctnum = 15
        results = numpy.zeros(shape = (20,946))


    elif (selection == 17):
        testdata = numpy.load("Data/ExpORBTestingDataV.npy")
        testdata = testdata.reshape(981,(10*256))
        correctnum = 16
        results = numpy.zeros(shape = (20,981))

    elif (selection == 18):
        testdata = numpy.load("Data/RomORBTestingDataV.npy")
        testdata = testdata.reshape(964,(10*256))
        correctnum = 17
        results = numpy.zeros(shape = (20,964))

    elif (selection == 19):
        testdata = numpy.load("Data/RelORBTestingDataV.npy")
        testdata = testdata.reshape(1542,(10*256))
        correctnum = 18
        results = numpy.zeros(shape = (20,1542))

    elif (selection == 20):
        testdata = numpy.load("Data/ImpORBTestingDataV.npy")
        testdata = testdata.reshape(1912,(10*256))
        correctnum = 19
        results = numpy.zeros(shape = (20,1912))


    #Load All SVM's
    UKISVM = pickle.load(open("1UKIORBProb.DT", "rb"))
    LReSVM = pickle.load(open("2LReORBProb.DT", "rb"))
    MinSVM = pickle.load(open("3MinORBProb.DT", "rb"))
    HRenSVM = pickle.load(open("4HRenORBProb.DT", "rb"))
    ERenSVM = pickle.load(open("5ERenORBProb.DT", "rb"))
    PopSVM = pickle.load(open("6PopORBProb.DT", "rb"))
    CFPSVM = pickle.load(open("7CFPORBProb.DT", "rb"))
    RocSVM = pickle.load(open("8RocORBProb.DT", "rb"))
    CubSVM = pickle.load(open("9CubORBProb.DT", "rb"))
    NAPSVM = pickle.load(open("10NAPORBProb.DT", "rb"))
    NRSVM = pickle.load(open("11NRORBProb.DT", "rb"))
    AESVM = pickle.load(open("12AEORBProb.DT", "rb"))
    BSVM = pickle.load(open("13BORBProb.DT", "rb"))
    ANMSVM = pickle.load(open("14ANMORBProb.DT", "rb"))
    SymSVM = pickle.load(open("15SymORBProb.DT", "rb"))
    PISVM = pickle.load(open("16PIORBProb.DT", "rb"))
    ESVM = pickle.load(open("17EORBProb.DT", "rb"))
    RomSVM = pickle.load(open("18RomORBProb.DT", "rb"))
    RelSVM = pickle.load(open("19RelORBProb.DT", "rb"))
    ImpSVM = pickle.load(open("20ImpORBProb.DT", "rb"))

    results[0] = UKISVM.predict_proba(testdata)[:,1]
    results[1] = LReSVM.predict_proba(testdata)[:,1]
    results[2] = MinSVM.predict_proba(testdata)[:,1]
    results[3] = HRenSVM.predict_proba(testdata)[:,1]
    results[4] = ERenSVM.predict_proba(testdata)[:,1]
    results[5] = PopSVM.predict_proba(testdata)[:,1]
    results[6] = CFPSVM.predict_proba(testdata)[:,1]
    results[7] = RocSVM.predict_proba(testdata)[:,1]
    results[8] = CubSVM.predict_proba(testdata)[:,1]
    results[9] = NAPSVM.predict_proba(testdata)[:,1]
    results[10] = NRSVM.predict_proba(testdata)[:,1]
    results[11] = AESVM.predict_proba(testdata)[:,1]
    results[12] = BSVM.predict_proba(testdata)[:,1]
    results[13] = ANMSVM.predict_proba(testdata)[:,1]
    results[14] = SymSVM.predict_proba(testdata)[:,1]
    results[15] = PISVM.predict_proba(testdata)[:,1]
    results[16] = ESVM.predict_proba(testdata)[:,1]
    results[17] = RomSVM.predict_proba(testdata)[:,1]
    results[18] = RelSVM.predict_proba(testdata)[:,1]
    results[19] = ImpSVM.predict_proba(testdata)[:,1]

    labels = ["UKI","LRe","Min","HRen","ERen","Pop","CFP","Roc","Cub","NAP","NR","AE","B","ANM","Sym","PI","ES","RoM","Rel","Imp"]

    resultsmax = results.argmax(axis=0)
    correct = list(resultsmax).count(correctnum)
    print(correct)
    selection = selection +1;


 
    