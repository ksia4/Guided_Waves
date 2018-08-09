from MES_dir import readData as rd
import numpy as np
import matplotlib.pyplot as plt
zmiana = 2000*np.pi
import test

#Klasa opisująca pojedynczy punkt krzywej dyspersji, czyli k i omega
class Point:

    def __init__(self, w=complex(0), k=float(0)):
        self.w = w.real /zmiana #wartość w khz
        self.wkat_real_part = w.real
        self.wkat_complex = w
        self.k = k

    #Funkcja do wypisywania współrzędnych punktu
    def printCoor(self):
        print("w = ", self.w, "k = ", self.k)

#Klasa przechowująca pojedynczy mode czyli po prostu uporządkowany zbiór punktów
class Mode:
    def __init__(self):
        self.points = []
        self.minOmega=float('inf') #wartość w khz
        self.min_omega_kat = float('inf') #wartość w rad/s
        self.allOmega=[]
        self.all_omega_khz = []

    #Funkcja dodająca kolejny punkt do danego modu
    def addPoint(self, point):
        self.points.append(point)
        if point.w < self.minOmega:
            self.minOmega=point.w
            self.min_omega_kat = point.wkat_real_part
        self.allOmega.append(point.wkat_complex)
        self.all_omega_khz.append(point.w)
    #Funkcja usuwająca punkt (podajemy punkt, nie indeks)
    def delPoint(self,point):
        pk = point.k
        pw = point.w
        for ind, todel in enumerate(self.points):
            if(todel.k == pk and todel.w == pw):
                self.points.pop(ind)
                return
        for ind, todel in enumerate(self.allOmega):
            if(todel.wkat_complex == point.wkat_complex):
                self.allOmega.pop(ind)
                break
        for ind, todel in enumerate(self.all_omega_khz):
            if(todel.w == pw):
                self.all_omega_khz.pop(ind)
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
        self.points[koniec], self.points[ktory] = self.points[ktory], self.points[koniec]
        while(i<=j):
            i += 1
            while(self.points[i].w < pivot):
                i += 1
            j -= 1
            while(self.points[j].w > pivot):
                j -= 1
            if(i<j):
                self.points[i], self.points[j] = self.points[j], self.points[i]
        self.points.insert(j+1, self.points[koniec])
        self.points.pop(koniec+1)
        if(j > pocz):
            self.quicksort(pocz, j)
        if(j+2 < koniec):
            self.quicksort(i, koniec)


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
            nvx = Ppoint.w - x2#svx
            nvy = Ppoint.k - y2#svy
            dotprod = (svx * nvx) + (svy * nvy) #iloczyn skalarny tych dwóch wektorów
            svlen = np.sqrt(svx**2 + svy**2)
            nvlen = np.sqrt(nvx**2 + nvy**2)
            return np.arccos(dotprod/(svlen*nvlen))

    #Funkcja która z pośród wektora punktów zwróci indeks tego, który tworzy najmniejszy kąt z tym już istniejącym i ten punkt będziemy dodawać do modu
    def findSmallestAngle(self, vPoints, dist=60000/zmiana):
        angle = float("inf")
        angind = float("inf")
        last = len(self.points) - 1
        for ind, Ppoint in enumerate(vPoints):
            if(abs(Ppoint.w - self.points[last].w) > dist): #40000
                continue
            temp = self.findAngle(Ppoint)
            if(temp < angle):
                angle = temp
                angind = ind

        if angind == float("inf"):

            return self.findSmallestAngle(vPoints, dist+20000/zmiana)
        else:
            return angind

    #Funkcja, która zwraca listę punktów które mogą być następnymi punktami w danym modzie (wyszukuje po prostu wszystkie punkty o podanym k, bo chyba wszystkie mody to funkcje rosnące... a przynajmniej tak mi się zdaje)
    def findPointsWithK(self, k):
        PotentialPoints = []
        for potPoint in self.points:
            if(potPoint.k == k):
                PotentialPoints.append(potPoint)
        return PotentialPoints

    def findPoint(self, points, omega):
        P1 = points[0]
        P2 = points[1]
        a = (P1.k - P2.k)/(P1.w - P2.w)
        b = P1.k - a * P1.w

        return a * omega + b

    def findPointWithGivenK(self, points, k):
        P1 = points[0]
        P2 = points[1]
        a = (P1.k - P2.k)/(P1.w - P2.w)
        b = P1.k - a * P1.w

        if a == 0:
            return b
        else:
            return (k-b)/a #w kHz

    def findPointWithGivenK_rad_s(self, points, k):
        P1 = points[0]
        P2 = points[1]
        a = (P1.k - P2.k)/(P1.wkat_complex - P2.wkat_complex)
        b = P1.k - a * P1.wkat_complex

        if a == 0:
            return b
        else:
            return (k-b)/a #rad/s

    def findKWithGivenOmega_kHz(self, omega_kHz):
        point1 = Point()
        point2 = Point()
        if omega_kHz < self.minOmega:
            return float(0)
        if omega_kHz > self.points[-1].w:
            point1 = self.points[-2]
            point2 = self.points[-1]
        for ind in range(len(self.points) - 1):
            if self.points[ind].w == omega_kHz:
                return self.points[ind].k
            if self.points[ind].w > omega_kHz:
                continue
            if self.points[ind].w < omega_kHz and self.points[ind+1].w >omega_kHz:
                point1 = self.points[ind]
                point2 = self.points[ind+1]
                break
        a = (point1.k - point2.k)/(point1.w - point2.w)
        b = point1.k - a * point1.w
        return a* omega_kHz + b


class Data:
    def __init__(self):
        self.modeTable = []
    def addMode(self,mode):
        self.modeTable.append(mode)


class SelectedMode:
    def __init__(self, kvect_path, omega_path, rows=426):
        self.eig_path = kvect_path
        self.omega_path = omega_path
        self.rows = rows
        self.AllModes = Data()
        self.k_v=[]

    def selectMode(self):
        #Wczytujemy kolejne k, dla których mamy omegi
        kvect = np.array(rd.read_kvect(self.eig_path)) #'../eig/kvect'
        self.k_v = kvect * 1e3

        #AllModes jest obiektem typu Mode, który przechowuje wszyskit nieprzydzielone jeszcze do żadnego modu obiekty
        AllPoints = Mode()

        #Czytamy z pliku wszystkie omegi i robimy z nich pointy o współrzędnych omega i k
        for ind in range(426): #Jak to sparametryzować? :/ to jest liczba wierszy w tym omega
            #temp = np.array(rd.read_complex_omega('../eig/omega', ind))/(2 * np.pi)
            temp = np.array(rd.read_complex_omega(self.omega_path, ind)) #'../eig/omega'
            for p in range (len(temp)):
                AllPoints.addPoint(Point(temp[p], self.k_v[p]))

        #obiekt Mode, w którym są punktu z najmniejszym K czyli pierwsze punkty kolejnych modów
        MinKTable = Mode()
        #drugie punkty kolejnych modów
        MinKTable2 = Mode()
        mink = min(self.k_v)
        #Wyszukiwanie pierwszych dwóch punktów kolejnych modów (punktów o najmniejszym i prawie najmniejszym k)
        for wszystko in AllPoints.points:
            if(wszystko.k == mink):
                MinKTable.addPoint(wszystko)
            elif(wszystko.k == self.k_v[1]):
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
            self.AllModes.addMode(Mode())
            self.AllModes.modeTable[ind].addPoint(m)
            self.AllModes.modeTable[ind].addPoint(MinKTable2.points[ind])

        #Segregowanie punktów do modów
        for i in range(2, len(self.k_v)):
            actk = self.k_v[i]
            potentialPoints = AllPoints.findPointsWithK(actk)
            test = np.array(potentialPoints)
            AllPoints.delDuplicats(potentialPoints)
            j = 0

            for mod in self.AllModes.modeTable:
                j += 1
                ind = mod.findSmallestAngle(potentialPoints)
                mod.addPoint(potentialPoints[ind])
                if(len(potentialPoints) > 3):
                    potentialPoints.pop(ind)

    def plot_modes(self,num_of_modes):
        plt.figure(1)
        for i in range(num_of_modes):
            dziady = []
            for p in self.AllModes.modeTable[i].points:
                dziady.append(p.w)
            plt.plot(dziady, self.k_v, markersize=3)
        plt.xlabel("Frequency [kHz]", fontsize=15)
        plt.ylabel("Wavenumber [rad/m]", fontsize=15)
        plt.xlim([0, 100])#600
        plt.ylim([0, 400])#2000
        plt.xticks(size=13)
        plt.yticks(size=13)
        plt.show()

    def getMode(self, number):
        return self.AllModes.modeTable[number]




if __name__ == "__main__":
    Mody = SelectedMode('../eig/kvect', '../eig/omega')
    Mody.selectMode()
    Mody.plot_modes(50)
