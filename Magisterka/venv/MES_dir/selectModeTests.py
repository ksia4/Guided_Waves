from MES_dir import readData as rd
import numpy as np
import matplotlib.pyplot as plt


#Klasa opisująca pojedynczy punkt krzywej dyspersji, czyli k i omega
class Point:

    def __init__(self,w = complex(0),k = float(0)):
        self.w = w.real
        self.k = k

    #Funkcja do wypisywania współrzędnych punktu
    def printCoor(self):
        print("w = ", self.w, "k = ", self.k)

    #Funkcja, która oblicza dystans między dwoma punktami Chyyba jednak nie będzie potrzebna
    def calcDist(self,point2):
        return np.sqrt((self.k-point2.k)**2+(self.w+point2.w)**2)

#Klasa przechowująca pojedynczy mode czyli po prostu uporządkowany zbiór punktów
class Mode:
    def __init__(self):
        self.points = []

    #Funkcja dodająca kolejny punkt do danego modu
    def addPoint(self, point):
        self.points.append(point)
    #Funkcja usuwająca punkt (podajemy punkt, nie indeks)
    def delPoint(self,point):
        pk = point.k
        pw = point.w
        for ind, todel in enumerate(self.points):
            if(todel.k == pk and todel.w == pw):
                self.points.pop(ind)
                return

    #Funkcja, która usuwa część wspólną listy punktów tego modu i podanej listy punktów
    def delDuplicats(self,pointlist):
        for dupl in pointlist:
            self.delPoint(dupl)

    #Sortowanie wszystkich punktów modu po omegach
    def quicksort(self, pocz, koniec):
        if(pocz + 1 >= koniec):
            return
        i = pocz - 1
        j = koniec
        ktory = int((koniec + pocz)/2)
        pivot = self.points[ktory].w
        self.points[koniec],self.points[ktory] = self.points[ktory],self.points[koniec]
        while(i<=j):
            i += 1
            while(self.points[i].w < pivot):
                i += 1
            j -= 1
            while(self.points[j].w > pivot):
                j -= 1
            if(i<j):
                self.points[i],self.points[j] = self.points[j],self.points[i]
        self.points.insert(j+1, self.points[koniec])
        self.points.pop(koniec+1)
        if(j > pocz):
            self.quicksort(pocz, j)
        if(j+2 < koniec):
            self.quicksort(i,koniec)


    #Znajduje kąt pomiędzy wektorami: 1. stworzonym przez ostatnie dwa punkty 2. Stworzony przez ostatni punkt modu i potencjalny punkt modu
    def findAngle(self, Ppoint):
        if(len(self.points)<2):
            return float("inf")
        else:
            ind2 = len(self.points)-1
            ind1 = ind2-1
            x1 = self.points[ind1].w
            x2 = self.points[ind2].w
            y1 = self.points[ind1].k
            y2 = self.points[ind2].k
            #self vx i self vy czyli wektor z dwóch ostatnich punktów dodanych do listy skierowany w stronę ostatniego dodanego punktu
            svx = x2 - x1
            svy = y2 - y1
            #wektor stworzony przez ostatni punkt dodany do listy i punkt Ppoint
            nvx = Ppoint.w - x2
            nvy = Ppoint.k - y2
            dotprod = (svx * nvx) + (svy * nvy) #iloczyn skalarny tych dwóch wektorów
            svlen = np.sqrt(svx**2 + svy**2)
            nvlen = np.sqrt(nvx**2 + nvy**2)
            print("kąt pomiędzy: \npoczątek:")
            print(self.points[ind1].printCoor())
            print("koniec")
            print(self.points[ind2].printCoor())
            print("a wektorem ktory ma koniec: ")
            print(Ppoint.printCoor())
            return np.arccos(dotprod/(svlen*nvlen))

    #Funkcja która z pośród wektora punktów zwróci indeks tego, który tworzy najmniejszy kąt z tym już istniejącym i ten punkt będziemy dodawać do modu
    def findSmallestAngle(self, vPoints, dist=100000000):
        angle = float("inf")
        angind = float("inf")
        last = len(self.points) - 1
        for ind, Ppoint in enumerate(vPoints):
            if(abs(Ppoint.w - self.points[last].w) > dist): #40000
                continue
            temp = np.abs(self.findAngle(Ppoint)) #abs bo nie jestem pewna czy kąt może być ujemny, ale chyyba może a znak tutaj nie ma znaczenia (chyba)
            if(temp < angle):
                angle = temp
                angind = ind
        #print(angind)
        #print(angle)
        return angind
#        if angind < float("inf"):
#            return self.findSmallestAngle(vPoints, dist+20000)
#        else:
#            return angind

    #Funkcja, która zwraca listę punktów które mogą być następnymi punktami w danym modzie (wyszukuje po prostu wszystkie punkty o podanym k, bo chyba wszystkie mody to funkcje rosnące... a przynajmniej tak mi się zdaje)
    def findPointsWithK(self, k):
        PotentialPoints = []
        for potPoint in self.points:
            if(potPoint.k == k):
                PotentialPoints.append(potPoint)
        return PotentialPoints


class Data:
    def __init__(self):
        self.modeTable = []
    def addMode(self,mode):
        self.modeTable.append(mode)

#Wczytujemy kolejne k, dla których mamy omegi
kvect = np.array(rd.read_kvect('../eig/kvect'))
k_v = kvect * 1e3

#AllModes jest obiektem typu Mode, który przechowuje wszyskit nieprzydzielone jeszcze do żadnego modu obiekty
AllPoints = Mode()

#Czytamy z pliku wszystkie omegi i robimy z nich pointy o współrzędnych omega i k
for ind in range(426): #Jak to sparametryzować? :/ to jest liczba wierszy w tym omega
    #temp = np.array(rd.read_complex_omega('../eig/omega', ind))/(2 * np.pi)
    temp = np.array(rd.read_complex_omega('../eig/omega', ind))
    for p in range (len(temp)):
        AllPoints.addPoint(Point(temp[p], k_v[p]))

#ModeTable to obiekt klasy data, który przechowuje listę wszystkich modów
ModeTable = Data()
#obiekt Mode, w którym są punktu z najmniejszym K czyli pierwsze punkty kolejnych modów
MinKTable = Mode()
#drugie punkty kolejnych modów
MinKTable2 = Mode()
mink = min(k_v)
#Wyszukiwanie pierwszych dwóch punktów kolejnych modów (punktów o najmniejszym i prawie najmniejszym k)
for wszystko in AllPoints.points:
    if(wszystko.k == mink):
        MinKTable.addPoint(wszystko)
    elif(wszystko.k == k_v[1]):
        MinKTable2.addPoint(wszystko)

#usuwanie punktów które już znalazły swój mod
AllPoints.delDuplicats(MinKTable.points)
AllPoints.delDuplicats(MinKTable2.points)

#sortowanie po omegach
AllPoints.quicksort(0, len(AllPoints.points) - 1)
MinKTable.quicksort(0, len(MinKTable.points)-1)
MinKTable2.quicksort(0, len(MinKTable.points)-1)

#dodajemy pierwsze dwa punkty do odpowiednich modów
for ind, m in enumerate(MinKTable.points):
    ModeTable.addMode(Mode())
    ModeTable.modeTable[ind].addPoint(m)
    ModeTable.modeTable[ind].addPoint(MinKTable2.points[ind])

#Segregowanie punktów do modów
for i in range(2, len(k_v)):
    actk = k_v[i]
    potentialPoints = AllPoints.findPointsWithK(actk)
    test = np.array(potentialPoints)
    AllPoints.delDuplicats(potentialPoints)
    j = 0
    print(i)
    for mod in ModeTable.modeTable:
        print(j)
        j += 1
        print("ostatnie dwa punkty to:")
        print(mod.points[i-1].printCoor())
        print(mod.points[i-2].printCoor())
        ind = mod.findSmallestAngle(potentialPoints)
        print("I wygrał punkt: ")
        print(ind)
        potentialPoints[ind].printCoor()
#        print(ind)
        mod.addPoint(potentialPoints[ind])
        if(len(potentialPoints) > 3):
            potentialPoints.pop(ind)
#    print(len(AllPoints.points))

#ModeTable.modeTable[0].points[0].printCoor()
#ModeTable.modeTable[0].points[1].printCoor()

plt.figure(1)
for i in range(50):
    dziady = []
    for p in ModeTable.modeTable[i].points:
        dziady.append(p.w/(2000 * np.pi))
    plt.plot(dziady, k_v, markersize=8)

plt.xlabel("Frequency [kHz]")
plt.ylabel("Wavenumber [rad/m]")
plt.xlim([0, 500])#600
plt.ylim([0, 400])#2000
plt.show()

